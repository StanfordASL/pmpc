import os, pickle, sys, pdb, math, time
import zmq, cloudpickle as cp

from . import *


PORT = 7117


def server_main():
    supported_methods = dict(solve=solve)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:%s" % str(PORT))
    while True:
        try:
            msg = sock.recv()
            syspath, data = cp.loads(msg)
            t = time.time()
            for path in syspath:
                if path not in sys.path:
                    sys.path.append(path)
            print("Updating path took %9.4e s" % (time.time() - t))
            method, args, kwargs = cp.loads(data)
            if method in supported_methods:
                ret = supported_methods[method](*args, **kwargs)
                sock.send(cp.dumps(ret))
        except KeyboardInterrupt as e:
           sock.send(cp.dumps(None))


def remote(method, *args, **kwargs):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://localhost:%s" % str(PORT))
    sock.send(cp.dumps((sys.path, cp.dumps((method, args, kwargs)))))
    return cp.loads(sock.recv())
