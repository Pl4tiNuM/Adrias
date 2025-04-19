import subprocess

# Find number of cores on the system
def num_cores():
    sh_cmd = "nproc"
    stdout,stderr = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.STDOUT,
                            shell=True).communicate()
    ncores = int(stdout)
    return ncores

# Find socket per core
def num_sockets():
    sh_cmd = "lscpu | grep Socket | awk -F':' '{print $2}'"
    stdout,stderr = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.STDOUT, 
                            shell=True).communicate()
    nsockets = int(stdout)
    return nsockets

def num_numa_nodes():
    sh_cmd = "lscpu | grep NUMA node(s) | awk -F':' '{print $2}'"
    stdout,stderr = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.STDOUT, 
                            shell=True).communicate()
    nnuma = int(stdout)
    return nnuma