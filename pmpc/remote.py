import os, gc, random, pickle, signal, sys, time, traceback  # noqa: E401
from argparse import ArgumentParser
from multiprocessing import Process, Value, set_start_method, get_start_method
from typing import Optional, Callable, Union, Any, Dict, List, Tuple
from socket import gethostname, gethostbyname
import psutil
from copy import copy

import zmq, zstandard, numpy as np  # noqa: E401
import cloudpickle as serializer
#import dill as serializer
from tqdm import tqdm

try:
    import redis
except ModuleNotFoundError:
    redis = None

from . import SOLVE_KWS
from .scp_mpc import solve as solve_, tune_scp as tune_scp_, lqp_generate_problem_matrices
from .problem_struct import Problem

SUPPORTED_METHODS = dict(
    solve=solve_, tune_scp=tune_scp_, lqp_generate_problem_matrices=lqp_generate_problem_matrices
)
DEFAULT_PORT = 65535 - 7117
DEFAULT_HOSTNAME = "localhost"
COMPRESSION_MODULE = zstandard
HOSTNAME = gethostname()
PID = os.getpid()
REDIS_CONFIG = dict()
if os.getenv("REDIS_HOST", None) is not None:
    REDIS_CONFIG["host"] = gethostbyname(os.getenv("REDIS_HOST"))
if os.getenv("REDIS_PORT", None) is not None:
    REDIS_CONFIG["port"] = int(os.getenv("REDIS_PORT"))
if os.getenv("REDIS_PASSWORD", None) is not None:
    REDIS_CONFIG["password"] = os.getenv("REDIS_PASSWORD")

####################################################################################################

REGISTERED_FUNCTIONS_CACHE = dict()


class RegisteredFunction:
    """A function carrying its own hash to be referenced instead of recompiled."""

    def __init__(self, fn: Callable) -> None:
        self.fn, self.hash = fn, serializer.dumps(fn)

    def __call__(self, *args, **kwargs):
        if self.hash not in REGISTERED_FUNCTIONS_CACHE:
            REGISTERED_FUNCTIONS_CACHE[self.hash] = self
            return self.fn(*args, **kwargs)
        else:
            return REGISTERED_FUNCTIONS_CACHE[self.hash].fn(*args, **kwargs)


## calling utilities ###############################################################################
def call(
    method: str,
    hostname: Optional[str] = None,
    port: Optional[int] = None,
    blocking: bool = True,
    *args,
    **kwargs,
) -> Union[Any, Callable]:
    """Call a remote function from a list of pre-approved functions in this
    library; blocking or not."""
    hostname = hostname if hostname is not None else DEFAULT_HOSTNAME
    port = port if port is not None else DEFAULT_PORT
    msg2send = serializer.dumps(
        (sys.path, COMPRESSION_MODULE.compress(serializer.dumps((method, args, kwargs))))
    )
    if blocking:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect(f"tcp://{hostname}:{str(port)}")
        sock.send(msg2send)
        return serializer.loads(COMPRESSION_MODULE.decompress(sock.recv()))
    else:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{hostname}:{str(port)}")
        sock.send(msg2send)

        def fn():
            if sock.poll(1e-4) == zmq.POLLIN:
                msg = sock.recv()
                return serializer.loads(COMPRESSION_MODULE.decompress(msg))
            else:
                return "NOT_ARRIVED_YET"

        fn.sock, fn.ctx = sock, ctx
        return fn


def solve(*args, **kw):
    return call("solve", solve.hostname, solve.port, solve.blocking, *args, **kw)


solve.hostname = DEFAULT_HOSTNAME
solve.port = DEFAULT_PORT
solve.blocking = True


def tune_scp(*args, **kw):
    return call("tune_scp", tune_scp.hostname, tune_scp.port, tune_scp.blocking, *args, **kw)


tune_scp.hostname = DEFAULT_HOSTNAME
tune_scp.port = DEFAULT_PORT
tune_scp.blocking = True


# server utils #####################################################################################

SERVERS = dict()


def start_server(
    port: int = DEFAULT_PORT, verbose: bool = False, redis_config: Optional[Dict[str, Any]] = None
):
    if port in SERVERS.keys():
        raise RuntimeError("PMPC server on this port already exits")
    if verbose:
        print(f"Starting PMPC server on port: {port:d}")
    SERVERS[port] = Server(port, redis_config)


def precompilation_call():
    p = Problem(N=20, xdim=4, udim=2, x0=np.random.randn(4), res_tol=-1.0, max_it=5)
    A = np.random.randn(p.xdim, p.xdim) + 0.4 * np.eye(p.xdim)
    A = np.tile(A, (p.N, 1, 1))
    B = np.tile(np.array([[0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.3, 0.0]]).T, (p.N, 1, 1))

    def linear_dynamics_f_fx_fu_fn(x, u, p=None):
        xp = (A @ x[..., None])[..., 0] + (B @ u[..., None])[..., 0]
        return xp, A, B

    p.f_fx_fu_fn = linear_dynamics_f_fx_fu_fn
    p.u_l, p.u_u = -10 * np.ones((p.N, p.udim)), 10 * np.ones((p.N, p.udim))
    solver_settings = copy(p.solver_settings)
    for solver in ["ecos", "osqp"]:
        for smooth in [True, False]:
            p.solver_settings = dict(solver_settings, solver=solver)
            if smooth and solver in ["ecos"]:
                p.solver_settings = dict(p.solver_settings, smooth_cstr="logbarrier", smooth_alpha=1e1)
            try:
                solve_(**p)
            except RuntimeError:
                pass # Mosek might be missing a license

    #Q, R, x0 = np.eye(2)[None, ...], np.eye(1)[None, ...], np.zeros(2)
    #f_fx_fu_fn = lambda x, u: (  # noqa: E731
    #    np.zeros((1, 2)),
    #    np.eye(2)[None, ...],
    #    np.ones((2, 1))[None, ...],
    #)
    #args = (f_fx_fu_fn, Q, R, x0)
    #solve_(*args, max_it=1, verbose=True)

    # blocking = True
    # call("solve", gethostbyname(hostname), port, blocking, *args, max_it=1, verbose=True)


####################################################################################################


def get_redis_connection(redis_config: Optional[Dict[str, Any]] = None) -> Optional["redis.Redis"]:
    if redis is None:
        return None
    try:
        redis_config = redis_config if redis_config is not None else copy(REDIS_CONFIG)
        if "host" in redis_config:
            redis_config["host"] = gethostbyname(redis_config["host"])
        rconn = redis.Redis(**redis_config)
        rconn.keys("")  # query for connection status
    except redis.ConnectionError:
        print(f"Could not connect to redis at {redis_config}")
        rconn = None
    return rconn


def set_redis_status(rconn: Optional["redis.Redis"], port: int) -> None:
    if redis is not None and rconn is not None:
        try:
            redis_key = f"pmpc_worker_{HOSTNAME}_{PID}/{HOSTNAME}:{port}"
            rconn.set(redis_key, f"{HOSTNAME}:{port}")
            rconn.expire(redis_key, 60)  # in seconds
        except redis.ConnectionError:
            pass


def unset_redis_status(rconn: Optional["redis.Redis"], port: int) -> None:
    if redis is not None and rconn is not None:
        try:
            redis_key = f"pmpc_worker_{HOSTNAME}_{PID}/{HOSTNAME}:{port}"
            rconn.delete(redis_key)
        except redis.ConnectionError:
            pass


## server routine ##################################################################################


def _server(
    exit_flag, status_flag, port=DEFAULT_PORT, redis_config: Optional[Dict[str, Any]] = None
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # bind to the zmq socket #######################################################################
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{port}")
    except zmq.error.ZMQError:
        print(f"Could not bind to port {port:d}")
        status_flag.value = time.time() - 1e5  # indicate failure
        return

    # precompile ###################################################################################
    print("Starting precompilation...")
    status_flag.value = time.time() + 60.0  # allow 60 seconds for precompilation
    precompilation_call()
    print("Precompilation done")

    rconn, redis_update = get_redis_connection(redis_config), time.time() - 10.0
    while not exit_flag.value:
        status_flag.value = time.time()
        if time.time() - redis_update > 10.0:
            set_redis_status(rconn, port)
            redis_update = time.time()

        # attempt to receive a message #############################################################
        is_msg_there = sock.poll(100)  # in milliseconds
        if is_msg_there:
            msg = sock.recv()
        else:
            continue

        # we have received a message ###############################################################
        try:
            syspath, data = serializer.loads(msg)
        except (pickle.UnpicklingError, EOFError):
            continue

        for path in syspath:
            if path not in sys.path:
                sys.path.append(path)
        error_str = ""
        try:
            method, args, kwargs = serializer.loads(COMPRESSION_MODULE.decompress(data))
        except (
            pickle.UnpicklingError,
            EOFError,
            TypeError,
            COMPRESSION_MODULE.ZstdError,
            ModuleNotFoundError,
        ):
            method = "UNSUPPORTED"
            error_str = traceback.format_exc()
            print(error_str)

        # method is a string and message unpacking was successful ##################################
        if method in SUPPORTED_METHODS:
            try:
                ret = SUPPORTED_METHODS[method](*args, **kwargs)
                compressed = COMPRESSION_MODULE.compress(serializer.dumps(ret))
                sock.send(compressed)
                continue
            except Exception:
                error_str = traceback.format_exc()
                print(error_str)

        # always respond ###########################################################################
        sock.send(COMPRESSION_MODULE.compress(serializer.dumps(error_str)))
        gc.collect()
        time.sleep(1e-3)
    unset_redis_status(rconn, port)
    sock.close()


class Server:
    def __init__(self, port=DEFAULT_PORT, redis_config: Optional[Dict[str, Any]] = None):
        self.port = port
        self.process, self.exit_flag, self.status_flag = None, None, None
        self.redis_config = redis_config
        self.start()

    def start(self):
        self.exit_flag, self.status_flag = Value("b", False), Value("d", time.time())
        self.process = Process(
            target=_server, args=(self.exit_flag, self.status_flag, self.port, self.redis_config)
        )
        self.old_signal_handler = signal.signal(signal.SIGINT, self.sighandler)
        self.process.start()

    def stop(self):
        if self.process is not None:
            self.exit_flag.value = True
            self.process.join()
            self.process.close()
            self.process, self.exit_flag, self.status_flag = None, None, None

    def is_alive(self):
        return time.time() - self.status_flag.value < 60.0

    def kill(self):
        if self.process is not None:
            self.process.kill()
            unset_redis_status(get_redis_connection(self.redis_config), self.port)

    def sighandler(self, signal, frame):
        self.stop()
        self.old_signal_handler(signal, frame)


# alternative & parallel solution interface ########################################################


def solve_problem(
    problem: Dict[str, Any], solve_fn: Callable = solve
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve a single optimal control problem defined by as keyword only arguments.

    Args:
        problem (Dict[str, Any]): Problem specification, positional arguments provided as keywords.
        solve (Callable, optional): Solve function. Defaults to in-process (non-remote) MPC solve.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: Solution to the optimal control problem.
    """
    problem = copy(problem)
    f_fx_fu_fn = problem["f_fx_fu_fn"]
    args = problem["Q"], problem["R"], problem["x0"]
    problem = {k: v for (k, v) in problem.items() if k not in {"f_fx_fu_fn", "Q", "R", "x0"}}
    problem.setdefault("verbose", True)
    if "extra_cstrs_fn_np" in problem:
        problem["extra_cstrs_fn"] = problem["extra_cstrs_fn_np"]
    return solve_fn(f_fx_fu_fn, *args, **problem)


def solve_problem_remote(
    args: Tuple[Dict[str, Any], int]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve a single problem by a call to a remote server, blocking.

    Args:
        args (Tuple[Dict[str, Any], int]): Tuple of problem, specifying the
        problem and port to expect the server on.

    Returns:
        _Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: A solution to a single
        problem obtained from the remote call to the server.
    """
    problem, hostname, port, blocking = args
    return solve_problem(
        problem,
        solve_fn=lambda *args, **kwargs: call("solve", hostname, port, blocking, *args, **kwargs),
    )


WORKER_ADDRS = []
LAST_WORKER_SCAN = time.time() - 60.0


def rescan_workers(force: bool = False, redis_config: Optional[Dict[str, Any]] = None):
    global LAST_WORKER_SCAN, WORKER_ADDRS
    if redis is not None and (force or time.time() - LAST_WORKER_SCAN > 10.0):
        try:
            redis_config = redis_config if redis_config is not None else copy(REDIS_CONFIG)
            if "host" in redis_config:
                redis_config["host"] = gethostbyname(redis_config["host"])
            rconn = redis.Redis(**redis_config)
            rconn.keys("")  # query for connection status
        except redis.ConnectionError:
            print(f"Could not connect to redis at {redis_config}")
            return
        worker_keys = rconn.keys("pmpc_worker*")
        WORKER_ADDRS = [
            tuple(worker_key.decode("utf8").split("/")[1].split(":")) for worker_key in worker_keys
        ]
        if len(WORKER_ADDRS) == 0:
            print("Could not find any PMPC workers registers in redis.")
        LAST_WORKER_SCAN = time.time()


def solve_problems(
    problems: List[Dict[str, Any]],
    verbose: bool = False,
    randomize_assignment: bool = True,
    rescan: bool = True,
    max_solve_time: float = 20.0,
    redis_config: Optional[Dict[str, Any]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Solve problems in parallel by calls to remote workers on WORKER_ADDRS.

    Args:
        problems (List[Dict[str, Any]]): A list of SCP problems defined as keyword dictionaries.
        verbose (bool, optional): Whether to print tqdm updates. Defaults to False.
        randomize_assignment (bool, optional): Use workers in a random order. Defaults to True.
        rescan (bool, optional): Whether to consult redis for workers. Defaults to True.
        max_solve_time (float, optional): Maximum solution time past which worker is deemed dead.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]: Solutions list from calls to scp_solve.
    """

    if rescan or len(WORKER_ADDRS) == 0:
        rescan_workers(force=False, redis_config=redis_config)

    workers = {worker_addr: None for worker_addr in WORKER_ADDRS}  # worker map
    pending, results = list(range(len(problems))), dict()  # in and out job/result queue
    pbar = tqdm(range(len(problems)), disable=not verbose)  # status bar
    while len(results) < len(problems):
        workers_items = list(workers.items())
        if randomize_assignment:  # randomize worker order
            random.shuffle(workers_items)
        for worker_addr, v in workers_items:  # for each worker collect result or assign new job
            if v is not None:
                idx, results_fn, solve_start_time = v
                ret = results_fn()
                if ret == "NOT_ARRIVED_YET":  # the worker is not finished yet
                    if time.time() - solve_start_time > max_solve_time:  # check if worker is dead
                        print(f"PMPC Worker {worker_addr} is broken, we will use another worker.")
                        # worker is broken
                        pending.append(idx)
                        results_fn.sock.close()
                        results_fn.ctx.destroy()
                        del workers[worker_addr]
                    continue  # is working, leave it alone
                results[idx], workers[worker_addr], v = ret, None, None  # is now free
                pbar.update(1)
            if v is None and len(pending) > 0:  # worker is done, we can assign it a new job
                idx = pending.pop()
                if problems[idx] is None:  # if the fed problem is None, just skip solving it
                    results[idx] = None
                else:
                    workers[worker_addr] = (
                        idx,
                        solve_problem_remote((problems[idx], *worker_addr, False)),
                        time.time(),
                    )
        if len(workers) == 0:  # all workers died
            print("All PMPC workers deemed dead, rescanning all PMPC workers.")
            rescan_workers(force=True)
            workers = {worker_addr: None for worker_addr in WORKER_ADDRS}
        time.sleep(1e-3)
    return [results[i] for i in range(len(problems))]


## module level access #############################################################################
def main():
    """Main routine."""
    #try:
    #    if get_start_method() == "fork":
    #        set_start_method("spawn")
    #except RuntimeError:
    #    pass

    parser = ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=DEFAULT_PORT, help="TCP port on which to start the server"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--worker-num",
        "-n",
        type=int,
        help="Number of workers to start, 0 means number equal to physical CPU cores.",
        default=1,
    )
    parser.add_argument(
        "--resurrect", type=bool, default=True, help="Whether to resurrect dead workers."
    )
    parser.add_argument("--redis-host", type=str, help="Redis host", default=None)
    parser.add_argument("--redis-port", type=str, help="Redis port", default=None)
    parser.add_argument("--redis-password", type=str, help="Redis password", default=None)
    args = parser.parse_args()
    if args.worker_num == 0:
        args.worker_num = psutil.cpu_count(logical=False)

    redis_config = copy(REDIS_CONFIG)
    if args.redis_host is not None:
        redis_config["host"] = args.redis_host
    if args.redis_port is not None:
        redis_config["port"] = args.redis_port
    if args.redis_password is not None:
        redis_config["password"] = args.redis_password

    for i in range(args.worker_num):
        start_server(args.port + i, verbose=args.verbose, redis_config=redis_config)

    try:
        while True:
            server_items = list(SERVERS.items())
            for port, server in server_items:
                if not server.is_alive():
                    print(f"Killing server on port {port}")
                    server.kill()
                    if args.resurrect:
                        new_port = max(list(SERVERS.keys())) + 1
                        print(f"Restarting dead server on port {new_port}")
                        start_server(
                            new_port,
                            verbose=args.verbose,
                            redis_config=redis_config,
                        )
                    del SERVERS[port]
            time.sleep(1.0)
    except KeyboardInterrupt:
        for port, server in SERVERS.items():
            print(f"Stopping server on port {port}")
            server.stop()

if __name__ == "__main__":
    main()
