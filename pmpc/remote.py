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

import cloudpickle as cp
import zmq
import zstandard

from .scp_mpc import solve as solve_
from .scp_mpc import tune_scp as tune_scp_

SUPPORTED_METHODS = dict(solve=solve_, tune_scp=tune_scp_)
DEFAULT_PORT = 65535 - 7117
COMPRESSION_MODULE = zstandard
# COMPRESSION_MODULE = gzip


## calling utilities ###########################################################
def call(method: str, port: Optional[int] = None, blocking: bool = True, *args, **kwargs):
    port = port if port is not None else DEFAULT_PORT
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://localhost:%s" % str(port))
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


solve = lambda *args, **kw: call("solve", solve.port, *args, **kw)
solve.port = DEFAULT_PORT
tune_scp = lambda *args, **kw: call("tune_scp", tune_scp.port, *args, **kw)
tune_scp.port = DEFAULT_PORT

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
def server_(exit_flag, port=DEFAULT_PORT, **kw):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:%s" % str(port))
    while not exit_flag.value:
        time.sleep(1e-3)
        is_msg_there = sock.poll(100)  # in milliseconds
        if is_msg_there:
            msg = sock.recv()
        else:
            continue

        try:
            syspath, data = cp.loads(msg)
        except (pickle.UnpicklingError, EOFError):
            continue

        # t = time.time()
        for path in syspath:
            if path not in sys.path:
                sys.path.append(path)
        # print("Updating path took %9.4e s" % (time.time() - t))

        try:
            method, args, kwargs = cp.loads(COMPRESSION_MODULE.decompress(data))
        # except (pickle.UnpicklingError, EOFError, TypeError, zstd.ZstdError):
        except (pickle.UnpicklingError, EOFError, TypeError, COMPRESSION_MODULE.ZstdError):
            method = "UNSUPPORTED"
        if method in SUPPORTED_METHODS:
            try:
                ret = SUPPORTED_METHODS[method](*args, **kwargs)
                compressed = COMPRESSION_MODULE.compress(cp.dumps(ret))
                sock.send(compressed)
                continue
            except Exception as e:
                traceback.print_exc()

        # always respond
        sock.send(COMPRESSION_MODULE.compress(cp.dumps(None)))
        gc.collect()
    sock.close()


class Server:
    def __init__(self, port=DEFAULT_PORT):
        self.exit_flag = Value("b", False)
        self.process = Process(target=server_, args=(self.exit_flag, port))
        self.old_signal_handler = signal.signal(signal.SIGINT, self.sighandler)
        self.process.start()

    def stop(self):
        if self.process is not None:
            self.exit_flag.value = True
            self.process.join()
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
    args = parser.parse_args()

    start_server(args.port, verbose=args.verbose)
    while True:
        time.sleep(1)
################################################################################
