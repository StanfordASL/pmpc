import os, pickle, sys, pdb, math, time, signal, traceback
from multiprocessing import Process, Value

#import zmq, cloudpickle as cp, zstandard as zstd
import gzip
import zmq, cloudpickle as cp

from .scp_mpc import solve as solve_, tune_scp as tune_scp_

PORT = 7117117


## calling utilities ###########################################################
def call(method, port, *args, **kwargs):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://localhost:%s" % str(port))
    msg2send = cp.dumps(
        (sys.path, gzip.compress(cp.dumps((method, args, kwargs))))
    )
    sock.send(msg2send)
    return cp.loads(gzip.decompress(sock.recv()))


solve = lambda *args, **kw: call("solve", solve.port, *args, **kw)
solve.port = PORT
tune_scp = lambda *args, **kw: call("tune_scp", tune_scp.port, *args, **kw)
tune_scp.port = PORT

################################################################################
## server utilities ############################################################
def start_server(port=PORT):
    if not hasattr(start_server, "servers"):
        start_server.servers = dict()
    if port in start_server.servers.keys():
        raise RuntimeError("PMPC server on this port already exits")
    start_server.servers[port] = Server(port)


################################################################################
## server routine ##############################################################
def server_(exit_flag, port=PORT, **kw):
    supported_methods = dict(solve=solve_, tune_scp=tune_scp_)
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
            method, args, kwargs = cp.loads(gzip.decompress(data))
        #except (pickle.UnpicklingError, EOFError, TypeError, zstd.ZstdError):
        except (pickle.UnpicklingError, EOFError, TypeError, gzip.ZstdError):
            method = "UNSUPPORTED"
        if method in supported_methods:
            try:
                ret = supported_methods[method](*args, **kwargs)
                sock.send(gzip.compress(cp.dumps(ret)))
                continue
            except Exception as e:
                traceback.print_exc()

        # always respond
        sock.send(gzip.compress(cp.dumps(None)))
    sock.close()


class Server:
    def __init__(self, port=PORT):
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
    start_server(PORT if len(sys.argv) <= 1 else sys.argv[1])
    while True:
        time.sleep(1)
################################################################################
