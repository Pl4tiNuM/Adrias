import subprocess
import threading
import numpy as np
import time
import config
import logging
import sysinfo

ts = 0

class TimeStamperDaemon:
    def __init__(self,interval):
        self.interval = interval/1000-0.05
        self.t        = threading.Thread(name='timestampDaemon', target=self.timestamp)
        self.t.daemon = True    # This thread dies when main thread exits
        self.t.start()          # Monitor process in the background
        logging.info("[Timestamper] Initialized")
    
    def timestamp(self):
        global ts
        while True:
            ts=time.time()
            time.sleep(self.interval)

class ContainerPerfmonDaemon:
    def __init__(self,app_name,pid,interval,output):
        self.parent   = app_name  # Name of container the process is executed on
        self.pid      = pid       # PID of running process inside the container
        self.perf     = []        # List to keep track of PID specific perf events
        self.name     = 'containerDaemon'+self.pid
        self.interval = interval
        self.output   = output
        self.t        = threading.Thread(name=self.name, target=self.monitor)
        self.t.daemon = True
        self.t.start()
        logging.info("[%s] Started monitoring PID %s of container %s" % (self.name, self.pid, self.parent))

    '''
    Format of container's perf list is like:
        [record_0, record_1, ..., record_n]
    where
        record_k = [timestamp, branch-instructions, branch-misses, ..., mem-stores]
    where the perf events are defined in config.py
    '''
    def monitor(self):
        sh_cmd = "sudo perf stat -I %d -a -x, -e %s -p %s" % (self.interval, ','.join(config.CONTAINER_PERF_EVENTS), self.pid)
        logging.info("[%s] %s" % (self.name, sh_cmd))
        self.mon = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            shell=True, bufsize=1)
        self.perf = []
        cntr = 1
        lst  = [ts]
        for line in self.mon.stderr:
            tmp = line.decode('utf-8').strip().split(',')[1]
            event = line.decode('utf-8').strip().split(',')[3]
            if (tmp == "<not counted>"):
                # logging.warning("[%s] Event %s not counted" % (self.name, event))
                lst = [ts]
                cntr = 1
                continue
            elif (tmp == "<not supported>"):
                # logging.warning("[%s] Event %s not supported" % (self.name, event))
                lst = [ts]
                cntr = 1
                continue
            else:
                lst.append(tmp)
            
            if cntr == len(config.CONTAINER_PERF_EVENTS):
                self.perf.append(lst)
                # logging.info("[%s] %s" % (self.name, lst))
                lst  = [ts]
                cntr = 1
            else:
                cntr += 1

    def terminate(self):
        logging.info("[%s] Terminating PerfmonDaemon PID %s of container %s" % (self.name, self.pid, self.parent))
        self.mon.kill()
        arr = np.array(self.perf)
        fn  = self.output+'_'.join(["results",str(self.parent),str(self.pid),"perf.out"])
        try:
            np.savetxt(fn,arr,delimiter=',',fmt='%s')
        except:
            logging.warning("[%s] Could not save monitoring results!" % self.name)
            pass
        else:
            logging.info("[%s] Results saved in %s" % (self.name,fn))

class SystemPerfmonDaemon:
    def __init__(self,interval,output):
        self.ncores   = sysinfo.num_cores() 
        self.nsockets = sysinfo.num_sockets()
        logging.info("[SystemPerfmonDaemon] Initializing System Performance monitoring Daemon")
        logging.info("[SystemPerfmonDaemon] Number of cores on the system: %d" % self.ncores)
        logging.info("[SystemPerfmonDaemon] Number of sockets on the system: %d" % self.nsockets)
        self.perf     = []        # List to keep track of system specific perf events
        self.interval = interval
        self.output   = output
        self.t        = threading.Thread(name='sysDaemon', target=self.monitor)
        self.t.daemon = True      # This thread dies when main thread exits
        self.t.start()            # Monitor process in the background
        logging.info("[SystemPerfmonDaemon] Initializing System Performance monitoring Daemon")
        logging.info("[SystemPerfmonDaemon] Number of cores on the system: %d" % self.ncores)
        logging.info("[SystemPerfmonDaemon] Number of sockets on the system: %d" % self.nsockets)
        logging.info("[SystemPerfmonDaemon] Starting monitoring S0 and FPGA")


    '''
    Format of system's perf list is like:
        [record_0, record_1, ..., record_n]
    where
        record_k = [timestamp, FPGA_rx, FPGA_tx, FPGA_lat, S0_branch-instructions, S0_branch-misses, ..., S0_mem-stores]
    where the perf events are defined in config.py
    '''
    def monitor(self):
        sh_cmd = "python3 -u %s/scripts/read_counters.py" % config.THYMESISFLOW_HOME
        logging.info("[SystemPerfmonDaemon] %s" % sh_cmd)
        self.fpgamon = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            shell=True, bufsize=1)
        #FIXME This command only grabs metrics for socket 0. If you want more, you have to modify it accordingly
        sh_cmd = "perf stat -I %d -a -x, -e %s --per-socket | grep S0" % (self.interval, ','.join(config.SYSTEM_PERF_EVENTS))
        logging.info("[SystemPerfmonDaemon] %s" % sh_cmd)
        self.sysmon = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            shell=True, bufsize=1)
        
        self.perf = []
        cntr = 1
        lst  = [ts]
        for ln_fpga in self.fpgamon.stdout:
            fpga_metrics = ln_fpga.decode('utf-8').strip().split(',')
            for ln_sys in self.sysmon.stderr:
                sys_metric  = ln_sys.decode('utf-8').strip().split(',')[3]
                sys_event   = ln_sys.decode('utf-8').strip().split(',')[5]
                if (sys_metric == "<not counted>"):
                    logging.warning("[SystemPerfmonDaemon] Event %s not counted" % sys_event)
                    lst  = [ts]
                    cntr = 1
                    continue
                elif (sys_metric == "<not supported>"):
                    logging.warning("[SystemPerfmonDaemon] Event %s not supported" % sys_event)
                    lst  = [ts]
                    cntr = 1
                    continue
                else:
                    lst.append(sys_metric)
                if cntr == (len(config.SYSTEM_PERF_EVENTS)):
                    lst.extend(fpga_metrics)
                    #logging.info("[SystemPerfmonDaemon] %s" % lst)
                    self.perf.append(lst)
                    lst  = [ts]
                    cntr = 1
                    break
                else:
                    cntr += 1
            
    def get_perf(self,history):
        tmp=np.array(self.perf[-history:])
        # Exclude timestamp in return
        return tmp[:,1:]

    def terminate(self):
        logging.info("[SystemPerfmonDaemon] Terminating FPGA monitoring subprocess")
        self.fpgamon.kill()
        logging.info("[SystemPerfmonDaemon] Terminating Socket's monitoring subprocess")
        self.sysmon.kill()
        arr = np.array(self.perf)
        fn  = self.output+'_'.join(["sysmon","results","perf.out"])
        try:
            np.savetxt(fn,arr,delimiter=',',fmt='%s')
        except:
            logging.error("[SystemPerfmonDaemon] Could not save System's monitoring results!" % fn)
            pass
        else:
            logging.info("[SystemPerfmonDaemon] System's monitoring results saved in %s" % fn)
            