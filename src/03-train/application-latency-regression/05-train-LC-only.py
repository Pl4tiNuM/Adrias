#!/usr/bin/python3

from os import PRIO_USER
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pickle import dump
from pickle import load
from model import HybridLSTM_LC

gpu_id = 0
horizon_model = torch.load('dep/trained_system_metrics_predictor.pt',map_location=torch.device('cuda'))
horizon_model.eval()
le = LabelEncoder()

def test_LSTM(test_loader, model, p99_tst, p99_prd, p999_tst, p999_prd, model_cfg, b_tst, b_prd):
    print("ENTERING TESTING MODE")
    cnt = 0
    model.eval()
    dt = []

    results = []

    with torch.no_grad():
        for xd,xs,xa,xm,p99,p999,b_batch in test_loader:
            st = time.time()
            xa = horizon_model(xs.cuda(gpu_id))
            out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
            predicted = out
            dt += [time.time() - st]
            b_batch = le.inverse_transform(b_batch)
            for mode,p99_r,p999_r,j,b in zip(xm,p99,p999,predicted.cpu(),b_batch):                
                p99_prd[cnt] = j[0]
                p999_prd[cnt] = j[1]

                results.append([b,mode.item(),p99_r.item(),j[0].item(),p999_r,j[1].item()])

                cnt = cnt + 1

    results = np.asarray(results)
    results = pd.DataFrame(data=results, columns=["Bench","Mode","p99 Real","p99 Predicted","p99.9 Real","p99.9 Predicted"])
    print(results)

    print("99th percentile accuracy : ", r2_score(results["p99 Real"], results["p99 Predicted"]))


def train_model(train_loader, model_cfg):
    # IF GPU IS AVAILABLE MAKE SURE TO SET THE PROPER GPU_ID ON THE TOP OF THIS FILE
    print("ENTERING TRAINING MODE")

    if (model_cfg["cuda"]):
        model = HybridLSTM_LC(model_cfg).cuda(gpu_id)
    else:
        model = HybridLSTM_LC(model_cfg)

    print(model)

    # For binary classification
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*model_cfg["epochs"]), int(.75*model_cfg["epochs"])], gamma=0.1)

    # Train the model
    model.train()
    optimizer.zero_grad()
    total_step = len(train_loader)
    st = time.time()
    for epoch in range(model_cfg["epochs"]):
        for i, (xd,xs,xa,xm,p99,p999,_) in enumerate(train_loader):
            with torch.no_grad():
                    xa = horizon_model(xs.cuda(gpu_id))
            torch.enable_grad()
            if (model_cfg["cuda"]):
                out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
                loss = criterion(out, torch.stack((p99,p999),-1).cuda(gpu_id))
            else:
                out = model(xd,xs,xa)
                loss = criterion(out, torch.stack((p99,p999),-1))
            # print(out,":",y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, model_cfg["epochs"], i+1, total_step, loss.item()))
        scheduler.step()
    st = time.time() - st

    return model


def preprocess(data,batch):
    X_def = np.array([x[0] for x in data[:,0]])
    X_sys = np.array([x for x in data[:,1]])
    X_avg = np.array([x for x in data[:,2]])
    X_mod = np.array([x for x in data[:,3]])
    p99_all  = data[:,4]
    p999_all = data[:,5]
    b_all = data[:,6]

    print("X_avg shape",X_avg.shape)
    print("p99_all shape",p99_all.shape)
    print("p999_all shape",p999_all.shape)

    Xd_trn, Xd_tst, Xs_trn, Xs_tst, Xa_trn, Xa_tst, Xm_trn, Xm_tst, p99_trn, p99_tst, p999_trn, p999_tst, b_trn, b_tst = train_test_split(X_def, X_sys, X_avg, X_mod, p99_all, p999_all, b_all, test_size=0.1)
    Xd_trn, Xs_trn, Xa_trn, Xm_trn, p99_trn, p999_trn, b_trn = X_def, X_sys, X_avg, X_mod, p99_all, p999_all, b_all
    #Xd_tst, Xs_tst, Xa_tst, Xm_tst, p99_tst, p999_tst = X_def, X_sys, X_avg, X_mod, p99_all, p999_all
    p99_prd  = np.zeros(shape=p99_tst.shape)
    p999_prd = np.zeros(shape=p999_tst.shape)
    b_prd = np.zeros(shape=b_tst.shape)
    print("--- Dataset shapes (Batch, History/Horizon, Features)")
    
    print("Xd_trn shape",X_def.shape)
    Xd_scaler = load(open('/host/Adrias/src/04-predict/dep/Xd_scaler.pkl', 'rb'))
    nsamples, nx, ny = Xd_trn.shape
    Xd_trn = Xd_trn.reshape((nsamples,nx*ny))
    Xd_trn = Xd_scaler.transform(Xd_trn)
    Xd_trn = Xd_trn.reshape((nsamples,nx,ny))
    print("Xd_tst shape:", Xd_tst.shape)
    nsamples, nx, ny = Xd_tst.shape
    Xd_tst = Xd_tst.reshape((nsamples,nx*ny))
    Xd_tst = Xd_scaler.transform(Xd_tst)
    Xd_tst = Xd_tst.reshape((nsamples,nx,ny))

    print("Xs_trn shape",Xs_trn.shape)
    Xs_scaler = load(open('/host/Adrias/src/04-predict/dep/Xs_scaler.pkl', 'rb'))
    nsamples, nx, ny = Xs_trn.shape
    Xs_trn = Xs_trn.reshape((nsamples,nx*ny))
    Xs_trn = Xs_scaler.transform(Xs_trn)
    Xs_trn = Xs_trn.reshape((nsamples,nx,ny))
    print("Xs_tst shape:", Xs_tst.shape)
    nsamples, nx, ny = Xs_tst.shape
    Xs_tst = Xs_tst.reshape((nsamples,nx*ny))
    Xs_tst = Xs_scaler.transform(Xs_tst)
    Xs_tst = Xs_tst.reshape((nsamples,nx,ny))

    print("Xa_trn shape",Xa_trn.shape)
    Xa_scaler = load(open('/host/Adrias/src/04-predict/dep/Xa_scaler.pkl', 'rb'))
    Xa_trn = Xa_scaler.transform(Xa_trn)
    print("Xa_tst shape:", Xa_tst.shape)
    Xa_tst = Xa_scaler.transform(Xa_tst)

    print("p99_trn shape:", p99_trn.shape)
    p99_scaler = MinMaxScaler()
    p99_trn = p99_scaler.fit_transform(p99_trn.reshape(-1, 1))
    p99_trn = p99_trn.reshape(-1)
    print("p99_tst shape:", p99_tst.shape)
    p99_tst = p99_scaler.transform(p99_tst.reshape(-1, 1))
    p99_tst = p99_tst.reshape(-1)
    dump(p99_scaler, open('p99_scaler.pkl', 'wb'))

    print("p999_trn shape:", p999_trn.shape)
    p999_scaler = MinMaxScaler()
    p999_trn = p999_scaler.fit_transform(p999_trn.reshape(-1, 1))
    p999_trn = p999_trn.reshape(-1)
    print("p999_tst shape:", p999_tst.shape)
    p999_tst = p999_scaler.transform(p999_tst.reshape(-1, 1))
    p999_tst = p999_tst.reshape(-1)

    print("converting train set to tensor arrays")
    Xd_trn   = torch.from_numpy(Xd_trn).float()
    Xs_trn   = torch.from_numpy(Xs_trn).float()
    Xa_trn   = torch.from_numpy(Xa_trn).float()
    Xm_trn   = torch.from_numpy(Xm_trn).float()
    p99_trn  = torch.from_numpy(p99_trn).float()
    p999_trn = torch.from_numpy(p999_trn).float()

    b_trn = le.fit_transform(b_trn)
    print(b_trn)
    b_trn = torch.from_numpy(b_trn).int()
    trn_dt = torch.utils.data.TensorDataset(Xd_trn,Xs_trn,Xa_trn,Xm_trn,p99_trn,p999_trn,b_trn)
    trn_ld = torch.utils.data.DataLoader(trn_dt, batch_size=batch, shuffle=True, drop_last=True)

    print("converting test set to tensor arrays")
    Xd_tst   = torch.from_numpy(Xd_tst).float()
    Xs_tst   = torch.from_numpy(Xs_tst).float()
    Xa_tst   = torch.from_numpy(Xa_tst).float()
    Xm_tst   = torch.from_numpy(Xm_tst).float()
    p99_tst  = torch.from_numpy(p99_tst).float()
    p999_tst = torch.from_numpy(p999_tst).float()
    b_tst = le.transform(b_tst)
    print(b_tst)
    b_tst = torch.from_numpy(b_tst).int()
    tst_dt = torch.utils.data.TensorDataset(Xd_tst,Xs_tst,Xa_tst,Xm_tst,p99_tst,p999_tst,b_tst)
    tst_ld = torch.utils.data.DataLoader(tst_dt, batch_size=batch, shuffle=False)

    return(trn_ld, tst_ld, p99_tst, p99_prd, p999_tst, p999_prd, b_tst, b_prd)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an LSTM network for given parameters")
    parser.add_argument("-c",    "--cuda",       type = bool,  default=False,       help="GPU available (Default: False)")
    parser.add_argument("-b",    "--batch",      type = int,   default=1024,          help="Batch size for training (Default: 32)")
    parser.add_argument("-l",    "--layers",     type = int,   default=4,           help="Number of LSTM stacked layers (Default: 2)")
    parser.add_argument("-f",    "--features",   type = int,   default=256,         help="Number of features on each LSTM layers (Default: 64)")
    parser.add_argument("-n",    "--rate",       type = float, default=1e-03,       help="Learning rate of the network (Default: 0.001)")
    parser.add_argument("-e",    "--epochs",     type = int,   default=150,         help="Number of training epochs of the network (Default: 100)")
    parser.add_argument("-o",    "--output",     type = str,   default="./trained-model-h_actual.pt",help="Path to save the state dictionary of the trained model")
    args=vars(parser.parse_args())

    return args

if __name__== "__main__":

    args=parse_arguments()

    print("--- Arguments")
    print(args)

    data = np.load("data/LC_120_dataset.npy",allow_pickle=True)
    # print(data)

    trn_ldr, tst_ldr, p99_tst, p99_prd, p999_tst, p999_prd, b_tst, b_prd = preprocess(data,args["batch"])
    print(b_tst)

    model_cfg = {
        "dflt_seq": len(data[0][0][0]),
        "hist_seq": len(data[0][1]),
        "mean_val": len(data[0][1][0]),
        "features": int(args["features"]),
        "layers"  : int(args["layers"]),
        "lr"      :    float(args["rate"]),
        "epochs"  : int(args["epochs"]),
        "cuda"    : bool(args["cuda"])
    }

    trnd_model = train_model(trn_ldr, model_cfg)

    tst_model = test_LSTM(tst_ldr, trnd_model, p99_tst, p99_prd, p999_tst, p999_prd, model_cfg, b_prd, b_tst)

    torch.save(trnd_model, args["output"])