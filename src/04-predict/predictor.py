#!/usr/bin/python3

import sys
sys.path.insert(1, '/nfs_homes/dmasouros/project/src/03-train/regression-average')
# sys.path.insert(1, '/host/03-train/actual')

import numpy as np
import time
import threading
import logging
# from sklearn.preprocessing import MinMaxScaler
# import torch.nn as nn

import torch
from model import LSTMModel
from pickle import load

class SystemPredictorDaemon:
    def __init__(self,sysperf_daemon) -> None:        
        logging.info("[Predictor] Initialized")
        logging.info("[Predictor] Num threads: %d" % torch.get_num_threads())
        torch.set_num_threads(64)
        logging.info("[Predictor] Num threads: %d" % torch.get_num_threads())
        self.sysperf_daemon = sysperf_daemon

        # Load system metrics prediction model
        self.sysperf_model  = torch.load('/nfs_homes/dmasouros/project/src/04-predict/dep/trained-model-sysperf.pt',map_location=torch.device('cpu'))
        self.sysperf_model.eval()
        # Load application latency prediction model
        self.latency_model  = torch.load('/nfs_homes/dmasouros/project/src/04-predict/dep/trained-model-h_pred_train_test.pt',map_location=torch.device('cpu'))
        self.latency_model.eval()
        # Load MinMaxScaler for system metrics
        self.Xs_scaler = load(open('/nfs_homes/dmasouros/project/src/04-predict/dep/Xs_scaler.pkl', 'rb'))
        # Load MinMaxScaler for app's signature
        self.Xd_scaler = load(open('/nfs_homes/dmasouros/project/src/04-predict/dep/Xd_scaler.pkl', 'rb'))
        # Load MinMaxScaler for output
        self.y_scaler = load(open('/nfs_homes/dmasouros/project/src/04-predict/dep/y_scaler.pkl', 'rb'))
        # Load system metrics for isolated execution
        self.sysperf_isol = load(open('/nfs_homes/dmasouros/project/src/04-predict/dep/isolated-sysperf.pkl', 'rb'))
        
        self.history  = 120
        self.t        = threading.Thread(name='predictor', target=self.main)
        self.t.daemon = True      # This thread dies when main thread exits
        self.t.start()            # Monitor process in the background
        logging.info("[SystemPredictorDaemon] Initializing System Predictor Daemon")
    
    def infer(self,bench):
        """
        xs: system metrics (history)
        xa: system metrics (predicted average)
        xd: app's signature
        """
        logging.info("[SystemPredictorDaemon] Got prediction request for %s application" % bench)
        metrics=self.sysperf_daemon.get_perf(self.history)
        xs=metrics.reshape(1,metrics.shape[0]*metrics.shape[1])
        logging.info("[SystemPredictorDaemon] Got metrics from system perfmon daemon ")
        try:
            xs = self.Xs_scaler.transform(xs)
        except: 
            logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            logging.warning("[SystemPredictorDaemon] I do not have enough history yet")
            logging.warning("[SystemPredictorDaemon] Current history: %d seconds" % metrics.shape[0])
            return (-1,-1)
        else:
            # Predict mean system metrics over the next 120 seconds
            try:
                xs=xs.reshape(1,metrics.shape[0],metrics.shape[1])
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 1")
            try:
                xs=torch.from_numpy(xs).float()
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 2")
            try:
                with torch.no_grad():
                    xa = self.sysperf_model(xs)
            except:
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 3")
            # Get app's signature
            try:
                xd = np.array([x[1] for x in self.sysperf_isol if x[0]==bench])
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 4")
            try:
                nsamples, nx, ny = xd.shape
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 5")
            try:
                xd = xd.reshape((nsamples,nx*ny))
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 6")
            try:
                xd = self.Xd_scaler.transform(xd)
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 7")
            try:
                xd = xd.reshape((nsamples,nx,ny))
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 8")
            try:
                xd = torch.from_numpy(xd).float()
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 9")
            # Predict local and remote execution time
            try:
                xm = torch.from_numpy(np.array([0])).float()
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 10")
            try:
                with torch.no_grad():
                    t_local  = self.latency_model(xd,xs,xa,xm)
                t_local  = t_local.detach().numpy()
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 11")
            try:
                t_local  = self.y_scaler.inverse_transform(t_local.reshape(-1, 1))
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 12")
            try:
                t_local  = t_local.reshape(-1)[0]
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 13")
            try:
                xm = torch.from_numpy(np.array([1])).float()
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 14")
            try:
                with torch.no_grad():
                    t_remote = self.latency_model(xd,xs,xa,xm)
                t_remote = t_remote.detach().numpy()
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 15")
            try:
                t_remote  = self.y_scaler.inverse_transform(t_remote.reshape(-1, 1))
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 16")
            try:
                t_remote  = t_remote.reshape(-1)[0]
            except: 
                logging.critical("[SystemPredictorDaemon] %s" % sys.exc_info()[0])
            else:
                logging.info("[SystemPredictorDaemon] passed 17")

            logging.info("[SystemPredictorDaemon] Predicted Local Latency: %f" % t_local)
            logging.info("[SystemPredictorDaemon] Predicted Remote Latency: %f" % t_remote)
            return (t_local,t_remote)

    def terminate(self):
        logging.info("[SystemPredictorDaemon] Terminating ")

    def main(self):
        while True:
            time.sleep(10)
