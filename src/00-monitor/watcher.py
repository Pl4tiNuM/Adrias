#!/usr/bin/python3
import sys
sys.path.insert(1, '/nfs_homes/dmasouros/project/src/04-predict/')
sys.path.insert(1, '/nfs_homes/dmasouros/project/src/05-orchestrate/')

import dockerinfo
import numpy as np
import time
import argparse
import signal
import logging
import argparse
import coloredlogs

from daemons import TimeStamperDaemon
from daemons import SystemPerfmonDaemon
from daemons import ContainerPerfmonDaemon
from predictor import SystemPredictorDaemon
from orchestrator import SystemOrchestratorDaemon

class Watcher:
    def __init__(self,sys_interval,con_interval,output,scheduler) -> None:
        signal.signal(signal.SIGINT, lambda signal, frame: self._signal_handler() )
        self.terminated = False
        self.active_containers     = []
        self.active_pids           = []
        self.active_daemons        = []
        self.sys_monitor_interval  = int(sys_interval)
        self.con_monitor_interval  = int(con_interval)
        self.output_folder         = output
        self.scheduler             = scheduler
        logging.info("[Watcher] Initialized")
    
    def _signal_handler(self):
        logging.critical("[Watcher] Catched SIGINT... Cleaning up, please be patient")
        logging.critical("[Watcher] Killing perf subprocesses")
        self.terminated = True
        for idx,_ in enumerate(self.active_daemons):
            self.active_daemons[idx].terminate()
        self.sysperf_daemon.terminate()
        self.syspred_daemon.terminate()
        self.sysorch_daemon.terminate()

    def cleanup(self,current_containers):
        current_pids = []
        for container in current_containers:
            pid = dockerinfo.pid_ps(container)
            if pid != 0:
                current_pids.append(pid)
        if (sorted(self.active_pids) != sorted(current_pids)):
            logging.info("[Watcher] Current  running containers: " + ' '.join(sorted(current_containers)))
            logging.info("[Watcher] Previous running containers: " + ' '.join(sorted(self.active_containers)))
            logging.info("[Watcher] Current  running PIDs: " + ' '.join(sorted(current_pids)))
            logging.info("[Watcher] Previous running PIDs: " + ' '.join(sorted(self.active_pids)))

        # Find finished containers (current containers - previous active containers)
        finished_containers = list(set(self.active_containers) - set(current_containers))
        if finished_containers:
            logging.info("[Watcher] Going to terminate daemons for: " + ' '.join(finished_containers))
        else:
            return

        for container in finished_containers:
            for daemon in self.active_daemons:
                if daemon.parent == container:
                    logging.info("[Watcher] Terminating %s" % daemon.name)
                    daemon.terminate()

        # Update active containers with current running containers (previous active - finished)
        self.active_containers = list(set(self.active_containers) - set(finished_containers))
        logging.info("[Watcher] Updated self.active_containers with: " + ' '.join(self.active_containers))

        finished_pids = list(set(self.active_pids) - set(current_pids))
        self.active_pids = list(set(self.active_pids) - set(finished_pids))
        logging.info("[Watcher] Updated self.active_pids with: " + ' '.join(self.active_pids))

    def main(self):
        self.timestamper    = TimeStamperDaemon(min(self.sys_monitor_interval,self.con_monitor_interval))
        self.sysperf_daemon = SystemPerfmonDaemon(self.sys_monitor_interval,self.output_folder)
        self.syspred_daemon = SystemPredictorDaemon(self.sysperf_daemon)
        self.sysorch_daemon = SystemOrchestratorDaemon(self.syspred_daemon,self.scheduler)
        while not self.terminated:
            # List running containers
            containers = dockerinfo.docker_ps()          

            self.cleanup(containers)
    
            # Find PIDs of tasks inside the container
            for container in containers:
                # Append container name to active containers
                if container not in self.active_containers:
                    self.active_containers.append(container)
                    pid = dockerinfo.pid_ps(container)
                    if (pid not in self.active_pids) and (int(pid) != 0):
                        self.active_pids.append(pid)
                        self.active_daemons.append(ContainerPerfmonDaemon(container,pid,self.con_monitor_interval,self.output_folder))

            time.sleep(2)

def parse_arguments():

    parser = argparse.ArgumentParser(description="Train an LSTM network for given parameters")
    parser.add_argument("-s",    "--system",    type = int,   default=1000,            help="Monitoring interval in msec (Default: 1000)")
    parser.add_argument("-c",    "--container", type = int,   default=1000,            help="Monitoring interval in msec (Default: 1000)")
    parser.add_argument("-d",    "--scheduler", type = str,   default="Random",        help="Sets Orchestrator scheduler logic (Default: Random)")
    parser.add_argument("-o",    "--output",    type = str,   default="./results/log/", help="Path to save the state dictionary of the trained model")
    args=vars(parser.parse_args())
    return args

if __name__== "__main__":
    coloredlogs.install(level='info')
    args=parse_arguments()
    app = Watcher(args["system"],args["container"],args["output"],args["scheduler"])
    app.main()
    logging.info("Everything cleaned up. Thanks for using me :)")
    exit()
