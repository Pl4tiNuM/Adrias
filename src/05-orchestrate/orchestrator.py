import subprocess
import time
import threading
import numpy as np
import logging
import zmq
import random
import itertools

class SystemOrchestratorDaemon:
    def __init__(self,syspred_daemon,scheduler):
        self.syspred_daemon = syspred_daemon
        self.t        = threading.Thread(name="orch", target=self.subscribe)
        self.t.daemon = True
        self.t.start()
        self.rr=itertools.cycle(["local","remote"])
        self.scheduler= scheduler
        self.b_BE = 0.8
        self.b_LC = 1.0
        logging.info("[Orchestrator] Initialized is %s mode" % self.scheduler)

    def orchestrate(self,bench,family):
        if (family == "ibench"):
            mode=random.choice(["local","remote"])
            return mode+",0"
        logging.info("[Orchestrator] Calling inference")
        t_local,t_remote = self.syspred_daemon.infer(bench)
        if (t_local == -1):
            mode=random.choice(["local"])
            logging.info("[Orchestrator] Got -1 as prediction. Choosing random mode %s" % mode)
            return mode+",0"
        else:
            logging.info("[Orchestrator] Predicted t_local:  %f" % t_local)
            logging.info("[Orchestrator] Predicted t_remote: %f" % t_remote)
            if (family == "spark"):
                if (t_local <= self.b_BE*t_remote):
                    mode="local"
                    logging.info("[Orchestrator] beta=%f - Choosing %s" % (self.b_BE,mode))
                    return mode+","+str(t_local)
                else:
                    mode="remote"
                    logging.info("[Orchestrator] beta=%f - Choosing %s" % (self.b_BE,mode))
                    return mode+","+str(t_remote)
            else:
                if (t_local <= self.b_LC*t_remote):
                    mode="local"
                    logging.info("[Orchestrator] beta=%f - Choosing %s" % (self.b_LC,mode))
                    return mode+","+str(t_local)
                else:
                    mode="remote"
                    logging.info("[Orchestrator] beta=%f - Choosing %s" % (self.b_LC,mode))
                    return mode+","+str(t_remote)                

    def terminate(self):
        logging.info("[Orchestrator] Unbinding socket")
        self.socket.unbind('tcp://127.0.0.1:5000')
        logging.info("[Orchestrator] Closing socket")
        self.socket.close()
        logging.info("[Orchestrator] Terminating context")
        self.context.term()
        logging.info("[Orchestrator] Terminated")

        
    def subscribe(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://127.0.0.1:5000')
        logging.info("[Orchestrator] Up and running listening on tcp://127.0.0.1:5000")

        while True:
            logging.info("[Orchestrator] Waiting for new container requests")

            b,f = self.socket.recv_pyobj()

            start = time.time()
            logging.info("[Orchestrator] New container submitted for deployment: %s-%s" % (b,f))
            if (self.scheduler == "Round-Robin"):
                mode=next(self.rr)
            elif (self.scheduler == "All-Local"):
                mode="local"
            elif (self.scheduler == "All-Remote"):
                mode="remote"
            elif (self.scheduler == "Random"):
                mode=random.choice(["local","remote"])
            else:
                mode=self.orchestrate(b,f)
            logging.info("[Orchestrator] Chosen mode: %s" % mode)
            logging.info("[Orchestrator] Time elapsed: %s" % (time.time()-start))
            self.socket.send_string(mode)