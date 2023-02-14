import os
import gc
import gzip
import pickle
import signal
import sys
import time
import traceback
from argparse import ArgumentParser
from multiprocessing import Process, Value
from typing import Optional
from socket import gethostname

import cloudpickle as cp
import zmq
import zstandard

try:
    import redis
except ModuleNotFoundError:
    redis = None

from .scp_mpc import solve as solve_
from .scp_mpc import tune_scp as tune_scp_

SUPPORTED_METHODS = dict(solve=solve_, tune_scp=tune_scp_)
DEFAULT_PORT = 65535 - 7117
DEFAULT_HOSTNAME = "localhost"
COMPRESSION_MODULE = zstandard
HOSTNAME = gethostname()
PID = os.getpid()


## calling utilities ###########################################################
def call(
    method: str,
    hostname: Optional[str] = None,
    port: Optional[int] = None,
    blocking: bool = True,
    *args,
    **kwargs,
):
    hostname = hostname if hostname is not None else DEFAULT_HOSTNAME
    port = port if port is not None else DEFAULT_PORT
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{hostname}:{str(port)}")
    msg2send = cp.dumps((sys.path, COMPRESSION_MODULE.compress(cp.dumps((method, args, kwargs)))))
    sock.send(msg2send)
    if blocking:
        return cp.loads(COMPRESSION_MODULE.decompress(sock.recv()))
    else:

        def fn():
            try:
                msg = sock.recv(flags=zmq.NOBLOCK)
                return cp.loads(COMPRESSION_MODULE.decompress(msg))
            except zmq.ZMQError:
                return "NOT_ARRIVED_YET"

        return fn


solve = lambda *args, **kw: call("solve", solve.hostname, solve.port, solve.blocking, *args, **kw)
solve.hostname = DEFAULT_HOSTNAME
solve.port = DEFAULT_PORT
solve.blocking = True
tune_scp = lambda *args, **kw: call(
    "tune_scp", tune_scp.hostname, tune_scp.port, tune_scp.blocking, *args, **kw
)
tune_scp.hostname = DEFAULT_HOSTNAME
tune_scp.port = DEFAULT_PORT
tune_scp.blocking = True


################################################################################
## server utilities ############################################################
def start_server(port: int = DEFAULT_PORT, verbose: bool = False):
    if not hasattr(start_server, "servers"):
        start_server.servers = dict()
    if port in start_server.servers.keys():
        raise RuntimeError("PMPC server on this port already exits")
    if verbose:
        print(f"Starting PMPC server on port: {port:d}")
    start_server.servers[port] = Server(port)


################################################################################
## server routine ##############################################################
def _server(exit_flag, port=DEFAULT_PORT, **kw):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    redis_update = time.time() - 10.0
    if redis is not None:
        try:
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_password = os.environ.get("REDIS_PASSWORD", None)
            if redis_password:
                rconn = redis.Redis(host=redis_host, password=redis_password)
            else:
                rconn = redis.Redis(host=redis_host)
        except redis.ConnectionError:
            print(f"Could not connect to redis at {redis_host} with password {redis_password}.")
            rconn = None

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    while not exit_flag.value:
        time.sleep(1e-3)
        is_msg_there = sock.poll(100)  # in milliseconds
        if is_msg_there:
            msg = sock.recv()
        else:
            if redis is not None and rconn is not None and time.time() - redis_update > 10.0:
                try:
                    redis_key = f"pmpc_worker_{HOSTNAME}_{PID}/{HOSTNAME}:{port}"
                    rconn.set(redis_key, f"{HOSTNAME}:{port}")
                    rconn.expire(redis_key, 60)
                except redis.ConnectionError:
                    pass
                redis_update = time.time()
            continue

        try:
            syspath, data = cp.loads(msg)
        except (pickle.UnpicklingError, EOFError):
            continue

        for path in syspath:
            if path not in sys.path:
                sys.path.append(path)
        error_str = ""
        try:
            method, args, kwargs = cp.loads(COMPRESSION_MODULE.decompress(data))
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
        if method in SUPPORTED_METHODS:
            try:
                ret = SUPPORTED_METHODS[method](*args, **kwargs)
                compressed = COMPRESSION_MODULE.compress(cp.dumps(ret))
                sock.send(compressed)
                continue
            except Exception as e:
                error_str = traceback.format_exc()
                print(error_str)

        # always respond
        sock.send(COMPRESSION_MODULE.compress(cp.dumps(error_str)))
        gc.collect()
    if rconn is not None:
        rconn.delete(redis_key)
    sock.close()


class Server:
    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.exit_flag = Value("b", False)
        self.process = Process(target=_server, args=(self.exit_flag, port))
        self.old_signal_handler = signal.signal(signal.SIGINT, self.sighandler)
        self.process.start()

    def stop(self):
        if self.process is not None:
            self.exit_flag.value = True
            self.process.join()
            self.process.close()
            self.process, self.exit_flag = None, None

    def sighandler(self, signal, frame):
        self.stop()
        self.old_signal_handler(signal, frame)


################################################################################
## module level access #########################################################
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=DEFAULT_PORT, help="TCP port on which to start the server"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--worker-num", "-n", type=int, help="Number of workers to start", default=1
    )
    args = parser.parse_args()
    assert args.worker_num > 0

    if args.worker_num == 1:
        start_server(args.port, verbose=args.verbose)
    else:
        for i in range(args.worker_num):
            start_server(args.port + i, verbose=args.verbose)
    while True:
        time.sleep(1)
################################################################################
