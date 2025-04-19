import os
import sys
import itertools
import numpy as np
import more_itertools as mit
import subprocess
from matplotlib.pyplot import hist
import pandas as pd

from pickle import dump
from pandas.io.parsers import read_csv


# mode=sys.argv[1]

spark_benchmarks=[
    'nweight',
    'repartition',
    'sort',
    'terasort',
    'wordcount',
    'als',
    'bayes',
    'gbt',
    'gmm',
    'kmeans',
    'lda',
    'lr',
    'pca',
    'rf',
    'svd',
    'svm',
    'pagerank'
]

sys_columns = [
'Timestamp',
'LLC-load-misses',
'LLC-loads',
'mem-loads',
'mem-stores',
'rx',
'tx',
'lat']

SPARK_PATH="/external/Dropbox/Adrias/results/01-benchmark-profiling/spark/remote/"

l = []

# SPARK
for bench in spark_benchmarks:
    b_perfmon = pd.read_csv(SPARK_PATH+bench+'/sysmon_results_perf.out', names=sys_columns)
    b_perfmon = b_perfmon.drop(['Timestamp'],axis=1)
    b_perfmon = b_perfmon.to_numpy()
    l.append([bench,b_perfmon])

# REDIS
bench='redis'
b_perfmon = pd.read_csv('/external/Dropbox/Adrias/results/01-benchmark-profiling/redis/requests/1635269647/sysmon_results_perf.out', names=sys_columns)
b_perfmon = b_perfmon.drop(['Timestamp'],axis=1)
b_perfmon = b_perfmon.to_numpy()
l.append([bench,b_perfmon])

# MEMCACHED
bench='memcached'
b_perfmon = pd.read_csv('/external/Dropbox/Adrias/results/01-benchmark-profiling/memcached/requests/1635284927/sysmon_results_perf.out', names=sys_columns)
b_perfmon = b_perfmon.drop(['Timestamp'],axis=1)
b_perfmon = b_perfmon.to_numpy()
l.append([bench,b_perfmon])

max_len=max([len(i[1]) for i in l])
for i,x in enumerate(l):
    b=x[0]
    m=x[1]
    shape = np.shape(m)
    padded_array = np.zeros((max_len,m.shape[1]))
    padded_array[:shape[0],:shape[1]] = m
    l[i] = [b,padded_array]

# dump(l, open('Xd_list.pkl', 'wb'))
# exit()

sys_columns = [
'Timestamp',
'LLC-load-misses',
'LLC-loads',
'mem-loads',
'mem-stores',
'rx',
'tx',
'lat']

f = []
g = []
# for RESULTS_PATH in [
#     "/external/Dropbox/Adrias/results/02-simulation/3600/all-local/",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/random/",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/round-robin/",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/adrias",
#     # "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.9",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.6spark/",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.7spark/",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.8spark/",
#     "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.9spark/"]:
    # "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.8spark-0.9lc/",
    # "/external/Dropbox/Adrias/results/02-simulation/3600/adrias-0.8spark-0.8lc/"]:
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-0.7all/",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-0.8all/",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-0.6spark",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-0.7spark",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-0.8spark",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-0.9spark",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-backup",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-backup-2",
    # "/external/Dropbox/Adrias/results/02-simulation/900/adrias-backup-3",
    # "/external/Dropbox/Adrias/results/02-simulation/900/round-robin/",
    # "/external/Dropbox/Adrias/results/02-simulation/900/random/"]:
for RESULTS_PATH in ["/external/Dropbox/Adrias/results/02-simulation/3600/random/","/external/Dropbox/Adrias/results/02-simulation/3600/round-robin/"]:
    try:
        for scenario in next(os.walk(RESULTS_PATH))[1]:
            print(RESULTS_PATH+scenario)
            SCENARIO_PATH = RESULTS_PATH + scenario + '/results/'
            sys_perfmon = pd.read_csv(SCENARIO_PATH + '/sysmon_results_perf.out',names=sys_columns)
            for bench in next(os.walk(SCENARIO_PATH))[1]:
                BENCH_PATH = SCENARIO_PATH + bench
                for t_start in next(os.walk(BENCH_PATH))[1]:
                    TS_PATH = '/'.join([BENCH_PATH,t_start])
                    try:
                        schedule=str(subprocess.check_output(
                                    "tail -n 1 " + TS_PATH + "/" + t_start + ".log | awk -F, '{print $NF}'", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                    except Exception as e:
                        print(e)
                        print("Oops")
                        continue

                    if bench in spark_benchmarks:
                        try:
                            t_end = int(subprocess.check_output(
                                    "tail -n 1 " + TS_PATH + "/" + t_start + ".log | awk '{print $1}' | tr -d [ | tr -d ]", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            hist_index = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start))].index[0]
                            if (hist_index >= 120):
                                hist_perf = sys_perfmon[hist_index-120:hist_index]
                                hist_perf = hist_perf.drop(['Timestamp'],axis=1)
                                hist_perf = hist_perf.to_numpy()
                            else: 
                                continue
                            # exec_perf = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start)) & (sys_perfmon['Timestamp'] <= float(t_end))].mean()
                            exec_perf = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start)) & (sys_perfmon['Timestamp'] <= float(t_start)+120.0)].mean()
                            exec_perf = exec_perf[1:]
                            exec_perf = exec_perf.to_numpy()
                            dflt_perf = [x[1] for x in l if x[0]==bench]
                            try:
                                latency = float(subprocess.check_output(
                                    "tail -n 1 " + TS_PATH + "/report.log | awk '{print $5}' | grep -v Duration", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                            except Exception as e:
                                print(e)
                                continue
                            else:
                                if (schedule=='local'):
                                    f.append([dflt_perf,hist_perf,exec_perf,0,latency,bench])
                                elif (schedule=='remote'):
                                    f.append([dflt_perf,hist_perf,exec_perf,1,latency,bench])
                            # print(l)
                    elif bench in 'redis':
                        try:
                            t_end = int(subprocess.check_output(
                                    "tail -n 1 " + TS_PATH + "/" + t_start + ".log | awk '{print $1}' | tr -d [ | tr -d ]", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            hist_index = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start))].index[0]
                            if (hist_index >= 120):
                                hist_perf = sys_perfmon[hist_index-120:hist_index]
                                hist_perf = hist_perf.drop(['Timestamp'],axis=1)
                                hist_perf = hist_perf.to_numpy()
                            else: 
                                continue
                            # exec_perf = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start)) & (sys_perfmon['Timestamp'] <= float(t_end))].mean()
                            exec_perf = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start)) & (sys_perfmon['Timestamp'] <= float(t_start)+120.0)].mean()
                            exec_perf = exec_perf[1:]
                            exec_perf = exec_perf.to_numpy()
                            dflt_perf = [x[1] for x in l if x[0]==bench]
                            try:
                                latency = float(subprocess.check_output(
                                    "cat " + TS_PATH + "/" + t_start + ".log | grep -oP \"[0-9]*.secs\" | tail -n1 | awk '{print $1}'", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                                p99 = float(subprocess.check_output(
                                    "cat " + TS_PATH + "/" + t_start + ".log | grep -P Totals | tail -n1 | awk '{print $7}'",
                                    shell=True,
                                    universal_newlines=True)[:-1])
                                p999 = float(subprocess.check_output( 
                                    "cat " + TS_PATH + "/" + t_start + ".log | grep -P Totals | tail -n1 | awk '{print $8}'",
                                    shell=True,
                                    universal_newlines=True)[:-1])
                                
                            except Exception as e:
                                print(e)
                                continue
                            else:
                                if (schedule=='local'):
                                    f.append([dflt_perf,hist_perf,exec_perf,0,latency,bench])
                                    g.append([dflt_perf,hist_perf,exec_perf,0,p99,p999,bench])
                                elif (schedule=='remote'):
                                    f.append([dflt_perf,hist_perf,exec_perf,1,latency,bench])
                                    g.append([dflt_perf,hist_perf,exec_perf,1,p99,p999,bench])
                    elif bench in 'memcached':
                        try:
                            t_end = int(subprocess.check_output(
                                    "tail -n 1 " + TS_PATH + "/" + t_start + ".log | awk '{print $1}' | tr -d [ | tr -d ]", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            hist_index = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start))].index[0]
                            if (hist_index >= 120):
                                hist_perf = sys_perfmon[hist_index-120:hist_index]
                                hist_perf = hist_perf.drop(['Timestamp'],axis=1)
                                hist_perf = hist_perf.to_numpy()
                            else: 
                                continue
                            # exec_perf = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start)) & (sys_perfmon['Timestamp'] <= float(t_end))].mean()
                            exec_perf = sys_perfmon.loc[(sys_perfmon['Timestamp'] >= float(t_start)) & (sys_perfmon['Timestamp'] <= float(t_start)+120.0)].mean()
                            exec_perf = exec_perf[1:]
                            exec_perf = exec_perf.to_numpy()
                            dflt_perf = [x[1] for x in l if x[0]==bench]
                            try:
                                latency = float(subprocess.check_output(
                                    "cat " + TS_PATH + "/" + t_start + ".log | grep -oP \"[0-9]*.secs\" | tail -n1 | awk '{print $1}'", 
                                    shell=True,
                                    universal_newlines=True)[:-1])
                                p99 = float(subprocess.check_output(
                                    "cat " + TS_PATH + "/" + t_start + ".log | grep -P Totals | tail -n1 | awk '{print $7}'",
                                    shell=True,
                                    universal_newlines=True)[:-1])
                                p999 = float(subprocess.check_output(
                                    "cat " + TS_PATH + "/" + t_start + ".log | grep -P Totals | tail -n1 | awk '{print $8}'",
                                    shell=True,
                                    universal_newlines=True)[:-1])
                            except Exception as e:
                                print(e)
                                continue
                            else:
                                if (schedule=='local'):
                                    f.append([dflt_perf,hist_perf,exec_perf,0,latency,bench])
                                    g.append([dflt_perf,hist_perf,exec_perf,0,p99,p999,bench])
                                elif (schedule=='remote'):
                                    f.append([dflt_perf,hist_perf,exec_perf,1,latency,bench])
                                    g.append([dflt_perf,hist_perf,exec_perf,1,p99,p999,bench])
                    else:
                        continue
    except Exception as e:
        print(e)

f = np.array(f,dtype=object)
np.save('final_dataset.npy',f)

g = np.array(g,dtype=object)

# g=np.load('final_perc_dataset.npy', allow_pickle=True)
# print(np.percentile(g[:,5],95))
np.save('final_perc_dataset.npy',g)
print(f.shape)
print(g.shape)
