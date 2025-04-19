
import zmq
import time
import json
import argparse

# fn = open('spawner.log', 'a')

def request_execution(b,f):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:5000')
    socket.send_pyobj([b,f])
    
    message = socket.recv()    
    socket.close()
    context.term()
    return message

def parse_arguments():
    parser = argparse.ArgumentParser(description="Spawn a container using Adrias framework")
    parser.add_argument("-b",    "--bench",      type = str,   default=None,       help="Name of benchmark")
    parser.add_argument("-f",    "--family",     type = str,   default=None,       help="Family of benchmark")
    args=vars(parser.parse_args())
    return args

if __name__== "__main__":
    
    args   = parse_arguments()
    bench  = args["bench"]
    family = args["family"]
    m=request_execution(bench,family)
    print(m.decode("utf-8"))