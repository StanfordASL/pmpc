import os, pickle, sys, pdb, math, time, signal
from multiprocessing import Process, Value

import zmq, cloudpickle as cp, zstandard as zstd

from .scp_mpc import solve as solve_, tune_scp as tune_scp_

PORT = 7117117


## calling utilities ###########################################################
def call(method, *args, **kwargs):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://localhost:%s" % str(PORT))
    msg2send = cp.dumps(
        (sys.path, zstd.compress(cp.dumps((method, args, kwargs))))
    )
    sock.send(msg2send)
    return cp.loads(zstd.decompress(sock.recv()))


solve = lambda *args, **kw: call("solve", *args, **kw)
tune_scp = lambda *args, **kw: call("tune_scp", *args, **kw)

################################################################################
## server utilities ############################################################
def start_server(background=False):
    if hasattr(start_server, "server"):
        raise RuntimeError("One PMPC server alread exits")
    if background:
        start_server.server = Server()
    else:
        exit_flag = Value("b", False)
        server_(exit_flag)


def stop_server():
    if not hasattr(start_server, "server"):
        return
    start_server.server.stop()
    delattr(start_server, "server")


################################################################################
## server routine ##############################################################
def server_(exit_flag, **kw):
    supported_methods = dict(solve=solve_, tune_scp=tune_scp_)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:%s" % str(PORT))
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
            method, args, kwargs = cp.loads(zstd.decompress(data))
            if method in supported_methods:
                ret = supported_methods[method](*args, **kwargs)
                sock.send(zstd.compress(cp.dumps(ret)))
                continue
        except (pickle.UnpicklingError, EOFError, TypeError, zstd.ZstdError):
        #except (pickle.UnpicklingError, EOFError, TypeError):
            pass

        # always respond
        sock.send(zstd.compress(cp.dumps(None)))
    sock.close()


class Server:
    def __init__(self):
        self.exit_flag = Value("b", False)
        self.process = Process(target=server_, args=(self.exit_flag,))
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
    start_server(background=True)
    while True:
        time.sleep(1)
################################################################################
