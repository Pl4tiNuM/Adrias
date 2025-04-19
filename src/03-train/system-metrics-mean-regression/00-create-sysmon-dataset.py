#!/usr/bin/python3

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

# RESULTS_DIR="../../../results/02-simulation/random/"

def create_dataset(history, horizon, interval):

    X_all=[]
    y_all=[]
    for RESULTS_DIR in ["../../../results/02-simulation/900/random/"]:
        print(RESULTS_DIR)
        for dir in next(os.walk(RESULTS_DIR))[1]:
            print("Currently processing %s..." % dir)
            fn  = RESULTS_DIR+dir+"/results/sysmon_results_perf.out"
            # Discard timestamps
            arr = np.loadtxt(fn,delimiter=',')
            # plot(arr)
            arr = arr[:,1:]
            print(arr.shape)
            l=arr.shape[0]
            # ptr is our pointer to traverse through the dataset
            ptr=history
            X_tmp=[]
            y_tmp=[]
            while (ptr+horizon <= l):
                # History (input features) is ptr-history:ptr window
                X_tmp=arr[ptr-history:ptr]
                # Horizon (predicted values) is ptr:ptr+horizon window
                y_tmp=arr[ptr:ptr+horizon]
                y_tmp=np.average(y_tmp,axis=0)
                X_all.append(X_tmp)
                y_all.append(y_tmp)
                # Next instance corresponds to "scheduling interval" window, which is ptr+interval
                ptr+=interval
                # exit()

    X_all = np.array(X_all)
        
        # X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)
        # y_all = np.array(y_all)[:,:,-1]
    y_all = np.array(y_all)
    return (X_all,y_all)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create dataset to be used for LSTM training")
    parser.add_argument("-r",    "--history",  type = int,   default=120,          help="How many past timestamps should consider as input")
    parser.add_argument("-z",    "--horizon",  type = int,   default=120,          help="How many future timestamps should predict")
    parser.add_argument("-i",    "--interval", type = int,   default=1,            help="How ofter should we make the prediction")
    args=vars(parser.parse_args())
    
    return args

if __name__== "__main__":

    args=parse_arguments()

    print("--- Create dataset for LSTM training")
    print("History %d" % args["history"])
    print("Horizon %d" % args["horizon"])
    print("Interval %d" % args["interval"])

    x,y = create_dataset(args["history"],args["horizon"],args["interval"])

    print("--- Dataset created")
    print("X shape: ", x.shape)
    print("y shape: ", y.shape)

    with open("X_900.npy", "wb") as f:
        np.save(f,x)
    with open("y_900.npy", "wb") as f:
        np.save(f,y)
