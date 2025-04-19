import subprocess
import re
import logging

# Returns the names of current running containers as a list
def docker_ps():
    sh_cmd = "docker ps | awk '{print $NF}' | grep -v NAMES"
    stdout,stderr = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.STDOUT,
                            shell=True).communicate()
    names = stdout.decode().splitlines()
    return names

# Returns the pid of entrypoint.sh script and command of target container as a list
def pid_ps(container_name):
    # grab pid of redis-server/memcached/spark worker
    if "redis" in container_name:
        sh_cmd = "docker top " + container_name + " | grep redis-server | awk '{print $2}'"
    elif "memcached" in container_name:
        sh_cmd = "docker top " + container_name + " | grep memcached | awk '{print $2}'"
    elif "spark" in container_name:
        sh_cmd = "docker top " + container_name + " | grep worker | awk '{print $2}'"
    elif "stream" in container_name:
        sh_cmd = "docker top " + container_name + " | grep entrypoint | awk '{print $2}'"
    else:
        return 0
    stdout,stderr = subprocess.Popen(sh_cmd,
                            stdout= subprocess.PIPE, 
                            stderr=subprocess.STDOUT,
                            shell=True).communicate()
    try:
        pid= stdout.decode().splitlines()[0]
    except:
        logging.error("[pid_ps] Could not decode PID")
        return 0
    else:
        return pid
