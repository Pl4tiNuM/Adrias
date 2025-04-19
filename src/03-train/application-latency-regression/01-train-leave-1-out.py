#!/usr/bin/python3

from os import PRIO_USER
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mae
from sklearn.model_selection import train_test_split

from pickle import dump
from model import HybridLSTM

gpu_id = 0

def runtime_test():
    
    model.eval()
    pred = []
    real = []
    with torch.no_grad():
        for xd,xs,xa,xm,y in test_loader:
            out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
            pred += out.cpu().tolist()
            real += y.tolist()
    model.train()
    return(r2_score(real,pred))

def whole_retrain():

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*model_cfg["epochs"]), int(.75*model_cfg["epochs"])], gamma=0.1)

	# Train the model
    model.train()

    optimizer.zero_grad()

    r2=runtime_test()
    print(0,0,r2,sep=",")


    for epoch in range(model_cfg["epochs"]):
        for xd,xs,xa,xm,y in retrain_loader:
            if (model_cfg["cuda"]):
                out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
                loss = criterion(out, y.cuda(gpu_id))
            else:
                out = model(xd,xs,xa)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        r2=runtime_test()
        scheduler.step()
        print(epoch,loss.item(),r2,sep=",")

def partial_retrain():

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    model.train()
    for (param,param_name) in zip(model.parameters(),model.named_parameters()):
        if ('LSTM.system' in param_name[0]):
            param.requires_grad=False
        else:
            param.requires_grad=True
    
    optimizer.zero_grad()
    
    r2=runtime_test()
    print(0,0,r2,sep=",")

    for epoch in range(model_cfg["epochs"]):
        for xd,xs,xa,xm,y in retrain_loader:
            if (model_cfg["cuda"]):
                out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
                loss = criterion(out, y.cuda(gpu_id))
            else:
                out = model(xd,xs,xa)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        r2=runtime_test()
        scheduler.step()
        print(epoch,loss.item(),r2,sep=",")

def train_model():
	# IF GPU IS AVAILABLE MAKE SURE TO SET THE PROPER GPU_ID ON THE TOP OF THIS FILE

    global criterion, optimizer, scheduler, model

    print("ENTERING TRAINING MODE")

    if (model_cfg["cuda"]):
        model = HybridLSTM(model_cfg).cuda(gpu_id)
    else:
        model = HybridLSTM(model_cfg)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*model_cfg["epochs"]), int(.75*model_cfg["epochs"])], gamma=0.1)

	# Train the model
    model.train()

    optimizer.zero_grad()
    st = time.time()
    for epoch in range(model_cfg["epochs"]):
        for i, (xd,xs,xa,xm,y) in enumerate(train_loader):
            if (model_cfg["cuda"]):
                out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
                loss = criterion(out, y.cuda(gpu_id))
            else:
                out = model(xd,xs,xa)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if (i+1) % 10 == 0:
            #     print ('\r>> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, model_cfg["epochs"], i+1, total_step, loss.item()),end='')
        r2=runtime_test()
        scheduler.step()
        print(epoch,loss.item(),r2,sep=",")
    st = time.time() - st

    return model

def preprocess_retrain(data,batch,bench):

    global train_loader, test_loader, retrain_loader

    X_def = np.array([x[0] for x in data[:,0]])
    X_sys = np.array([x for x in data[:,1]])
    X_avg = np.array([x for x in data[:,2]])
    X_mod = np.array([x for x in data[:,3]])
    y_all=data[:,4]

    print("X_def shape",X_def.shape)
    print("X_sys shape",X_sys.shape)
    print("X_avg shape",X_avg.shape)
    print("X_mod shape",X_mod.shape)
    print("y_all shape",y_all.shape)

    (
        Xd_trn, Xd_tst,
        Xs_trn, Xs_tst, 
        Xa_trn, Xa_tst, 
        Xm_trn, Xm_tst, 
        y_trn, y_tst,
        inds_trn, inds_tst
    ) = train_test_split(X_def, X_sys, X_avg, X_mod, y_all, np.arange(len(y_all)), test_size=0.4, random_state=42)

    not_bench_inds = [i for i,x in enumerate(data) if ((x[-1] != bench) and (i in inds_trn))]
    not_bench_inds = np.asarray(not_bench_inds)


    bench_inds = [i for i,x in enumerate(data) if x[-1] == bench]

    random.seed(42)
    random.shuffle(bench_inds)

    retrain_instances = 64
    bench_retrain_instances = bench_inds[:retrain_instances]
    bench_retrain_instances = np.asarray(inds_tst.tolist()+bench_retrain_instances)

    bench_test_instances  = np.asarray(bench_inds)


    Xd_trn, Xs_trn, Xa_trn, Xm_trn, y_trn = X_def[not_bench_inds], X_sys[not_bench_inds], X_avg[not_bench_inds], X_mod[not_bench_inds], y_all[not_bench_inds]
    
    Xd_tst, Xs_tst, Xa_tst, Xm_tst, y_tst = X_def[bench_test_instances], X_sys[bench_test_instances], X_avg[bench_test_instances], X_mod[bench_test_instances], y_all[bench_test_instances]

    Xd_retrn, Xs_retrn, Xa_retrn, Xm_retrn, y_retrn = X_def[bench_retrain_instances], X_sys[bench_retrain_instances], X_avg[bench_retrain_instances], X_mod[bench_retrain_instances], y_all[bench_retrain_instances]

    # Xd_retrn, Xs_retrn, Xa_retrn, Xm_retrn, y_retrn = X_def[bench_test_instances], X_sys[bench_test_instances], X_avg[bench_test_instances], X_mod[bench_test_instances], y_all[bench_test_instances]


    y_prd = np.zeros(shape=y_tst.shape)
    print("--- Dataset shapes (Batch, History/Horizon, Features)")

    # Train data
    print("Xd_trn shape",Xd_trn.shape)
    Xd_scaler = MinMaxScaler()
    nsamples, nx, ny = Xd_trn.shape
    Xd_trn = Xd_trn.reshape((nsamples,nx*ny))
    Xd_trn = Xd_scaler.fit_transform(Xd_trn)
    Xd_trn = Xd_trn.reshape((nsamples,nx,ny))
    print("Xs_trn shape",Xs_trn.shape)
    Xs_scaler = MinMaxScaler()
    nsamples, nx, ny = Xs_trn.shape
    Xs_trn = Xs_trn.reshape((nsamples,nx*ny))
    Xs_trn = Xs_scaler.fit_transform(Xs_trn)
    Xs_trn = Xs_trn.reshape((nsamples,nx,ny))
    print("Xa_trn shape",Xa_trn.shape)
    Xa_scaler = MinMaxScaler()
    Xa_trn = Xa_scaler.fit_transform(Xa_trn)
    print("y_trn shape:", y_trn.shape)
    y_scaler = MinMaxScaler()
    y_trn = y_scaler.fit_transform(y_trn.reshape(-1, 1))
    y_trn = y_trn.reshape(-1)
    print("converting train set to tensor arrays")
    Xd_trn = torch.from_numpy(Xd_trn).float()
    Xs_trn = torch.from_numpy(Xs_trn).float()
    Xa_trn = torch.from_numpy(Xa_trn).float()
    Xm_trn = torch.from_numpy(Xm_trn).float()
    y_trn = torch.from_numpy(y_trn).float()
    trn_dt = torch.utils.data.TensorDataset(Xd_trn,Xs_trn,Xa_trn,Xm_trn,y_trn)
    train_loader = torch.utils.data.DataLoader(trn_dt, batch_size=batch, shuffle=True, drop_last=True)

    # Test data
    print("Xd_tst shape:", Xd_tst.shape)
    nsamples, nx, ny = Xd_tst.shape
    Xd_tst = Xd_tst.reshape((nsamples,nx*ny))
    Xd_tst = Xd_scaler.transform(Xd_tst)
    Xd_tst = Xd_tst.reshape((nsamples,nx,ny))
    print("Xs_tst shape:", Xs_tst.shape)
    nsamples, nx, ny = Xs_tst.shape
    Xs_tst = Xs_tst.reshape((nsamples,nx*ny))
    Xs_tst = Xs_scaler.transform(Xs_tst)
    Xs_tst = Xs_tst.reshape((nsamples,nx,ny))
    print("Xa_tst shape:", Xa_tst.shape)
    Xa_tst = Xa_scaler.transform(Xa_tst)
    print("y_tst shape:", y_tst.shape)
    y_tst = y_scaler.transform(y_tst.reshape(-1, 1))
    y_tst = y_tst.reshape(-1)
    print("converting test set to tensor arrays")
    Xd_tst = torch.from_numpy(Xd_tst).float()
    Xs_tst = torch.from_numpy(Xs_tst).float()
    Xa_tst = torch.from_numpy(Xa_tst).float()
    Xm_tst = torch.from_numpy(Xm_tst).float()
    y_tst = torch.from_numpy(y_tst).float()
    tst_dt = torch.utils.data.TensorDataset(Xd_tst,Xs_tst,Xa_tst,Xm_tst,y_tst)
    test_loader = torch.utils.data.DataLoader(tst_dt, batch_size=batch, shuffle=False)

    # Retrain data
    print("Xd_retrn shape",Xd_retrn.shape)
    nsamples, nx, ny = Xd_retrn.shape
    Xd_retrn = Xd_retrn.reshape((nsamples,nx*ny))
    Xd_retrn = Xd_scaler.transform(Xd_retrn)
    Xd_retrn = Xd_retrn.reshape((nsamples,nx,ny))
    print("Xs_retrn shape",Xs_retrn.shape)
    nsamples, nx, ny = Xs_retrn.shape
    Xs_retrn = Xs_retrn.reshape((nsamples,nx*ny))
    Xs_retrn = Xs_scaler.transform(Xs_retrn)
    Xs_retrn = Xs_retrn.reshape((nsamples,nx,ny))
    print("Xa_retrn shape",Xa_retrn.shape)
    Xa_retrn = Xa_scaler.transform(Xa_retrn)
    print("y_retrn shape:", y_retrn.shape)
    y_retrn = y_scaler.transform(y_retrn.reshape(-1, 1))
    y_retrn = y_retrn.reshape(-1)
    print("converting retrain set to tensor arrays")
    Xd_retrn = torch.from_numpy(Xd_retrn).float()
    Xs_retrn = torch.from_numpy(Xs_retrn).float()
    Xa_retrn = torch.from_numpy(Xa_retrn).float()
    Xm_retrn = torch.from_numpy(Xm_retrn).float()
    y_retrn = torch.from_numpy(y_retrn).float()
    retrn_dt = torch.utils.data.TensorDataset(Xd_retrn,Xs_retrn,Xa_retrn,Xm_retrn,y_retrn)
    retrain_loader = torch.utils.data.DataLoader(retrn_dt, batch_size=batch, shuffle=True, drop_last=True)

    return


def preprocess_test_specific_benchmark(data,batch,bench):

    global train_loader, test_loader

    X_def = np.array([x[0] for x in data[:,0]])
    X_sys = np.array([x for x in data[:,1]])
    X_avg = np.array([x for x in data[:,2]])
    X_mod = np.array([x for x in data[:,3]])
    y_all=data[:,4]

    print("X_def shape",X_def.shape)
    print("X_sys shape",X_sys.shape)
    print("X_avg shape",X_avg.shape)
    print("X_mod shape",X_mod.shape)
    print("y_all shape",y_all.shape)

    # Xd_trn, Xd_tst, Xs_trn, Xs_tst, Xa_trn, Xa_tst, Xm_trn, Xm_tst, y_trn, y_tst = train_test_split(X_def, X_sys, X_avg, X_mod, y_all, test_size=0.05)


    not_bench_inds = [i for i,x in enumerate(data) if data[i,-1] != bench]
    bench_inds = [i for i,x in enumerate(data) if data[i,-1] == bench]

    random.seed(42)
    random.shuffle(bench_inds)

    train_instances = 512
    test_instances  = 128
    bench_train_instances = bench_inds[:train_instances]
    bench_test_instances  = bench_inds[-test_instances:]

    not_bench_inds = np.asarray(not_bench_inds+bench_train_instances)
    bench_test_instances = np.asarray(bench_test_instances)

    Xd_trn, Xs_trn, Xa_trn, Xm_trn, y_trn = X_def[not_bench_inds], X_sys[not_bench_inds], X_avg[not_bench_inds], X_mod[not_bench_inds], y_all[not_bench_inds]
    
    Xd_tst, Xs_tst, Xa_tst, Xm_tst, y_tst = X_def[bench_test_instances], X_sys[bench_test_instances], X_avg[bench_test_instances], X_mod[bench_test_instances], y_all[bench_test_instances]




    y_prd = np.zeros(shape=y_tst.shape)
    print("--- Dataset shapes (Batch, History/Horizon, Features)")

    print("Xd_trn shape",Xd_trn.shape)
    Xd_scaler = MinMaxScaler()
    nsamples, nx, ny = Xd_trn.shape
    Xd_trn = Xd_trn.reshape((nsamples,nx*ny))
    Xd_trn = Xd_scaler.fit_transform(Xd_trn)
    Xd_trn = Xd_trn.reshape((nsamples,nx,ny))
    print("Xd_tst shape:", Xd_tst.shape)
    nsamples, nx, ny = Xd_tst.shape
    Xd_tst = Xd_tst.reshape((nsamples,nx*ny))
    Xd_tst = Xd_scaler.transform(Xd_tst)
    Xd_tst = Xd_tst.reshape((nsamples,nx,ny))

    print("Xs_trn shape",Xs_trn.shape)
    Xs_scaler = MinMaxScaler()
    nsamples, nx, ny = Xs_trn.shape
    Xs_trn = Xs_trn.reshape((nsamples,nx*ny))
    Xs_trn = Xs_scaler.fit_transform(Xs_trn)
    Xs_trn = Xs_trn.reshape((nsamples,nx,ny))
    print("Xs_tst shape:", Xs_tst.shape)
    nsamples, nx, ny = Xs_tst.shape
    Xs_tst = Xs_tst.reshape((nsamples,nx*ny))
    Xs_tst = Xs_scaler.transform(Xs_tst)
    Xs_tst = Xs_tst.reshape((nsamples,nx,ny))

    print("Xa_trn shape",Xa_trn.shape)
    Xa_scaler = MinMaxScaler()
    Xa_trn = Xa_scaler.fit_transform(Xa_trn)
    print("Xa_tst shape:", Xa_tst.shape)
    Xa_tst = Xa_scaler.transform(Xa_tst)

    print("y_trn shape:", y_trn.shape)
    y_scaler = MinMaxScaler()
    y_trn = y_scaler.fit_transform(y_trn.reshape(-1, 1))
    y_trn = y_trn.reshape(-1)
    print("y_tst shape:", y_tst.shape)
    y_tst = y_scaler.transform(y_tst.reshape(-1, 1))
    y_tst = y_tst.reshape(-1)

    print("converting train set to tensor arrays")
    Xd_trn = torch.from_numpy(Xd_trn).float()
    Xs_trn = torch.from_numpy(Xs_trn).float()
    Xa_trn = torch.from_numpy(Xa_trn).float()
    Xm_trn = torch.from_numpy(Xm_trn).float()
    y_trn = torch.from_numpy(y_trn).float()
    trn_dt = torch.utils.data.TensorDataset(Xd_trn,Xs_trn,Xa_trn,Xm_trn,y_trn)
    train_loader = torch.utils.data.DataLoader(trn_dt, batch_size=batch, shuffle=True, drop_last=True)

    print("converting test set to tensor arrays")
    Xd_tst = torch.from_numpy(Xd_tst).float()
    Xs_tst = torch.from_numpy(Xs_tst).float()
    Xa_tst = torch.from_numpy(Xa_tst).float()
    Xm_tst = torch.from_numpy(Xm_tst).float()
    y_tst = torch.from_numpy(y_tst).float()
    tst_dt = torch.utils.data.TensorDataset(Xd_tst,Xs_tst,Xa_tst,Xm_tst,y_tst)
    test_loader = torch.utils.data.DataLoader(tst_dt, batch_size=batch, shuffle=True)

    return


def preprocess(data,batch,bench):

    global train_loader, test_loader

    X_def = np.array([x[0] for x in data[:,0]])
    X_sys = np.array([x for x in data[:,1]])
    X_avg = np.array([x for x in data[:,2]])
    X_mod = np.array([x for x in data[:,3]])
    y_all=data[:,4]

    print("X_def shape",X_def.shape)
    print("X_sys shape",X_sys.shape)
    print("X_avg shape",X_avg.shape)
    print("X_mod shape",X_mod.shape)
    print("y_all shape",y_all.shape)

    (
        Xd_trn, Xd_tst,
        Xs_trn, Xs_tst, 
        Xa_trn, Xa_tst, 
        Xm_trn, Xm_tst, 
        y_trn, y_tst,
        inds_trn, inds_tst
    ) = train_test_split(X_def, X_sys, X_avg, X_mod, y_all, np.arange(len(y_all)), test_size=0.4, random_state=42)


    not_bench_inds = [i for i,x in enumerate(data) if ((data[i,-1] != bench) and (i not in inds_tst))]
    Xd_trn, Xs_trn, Xa_trn, Xm_trn, y_trn = X_def[not_bench_inds], X_sys[not_bench_inds], X_avg[not_bench_inds], X_mod[not_bench_inds], y_all[not_bench_inds]
    
    bench_inds = [i for i,x in enumerate(data) if data[i,-1] == bench]
    Xd_tst, Xs_tst, Xa_tst, Xm_tst, y_tst = X_def[bench_inds], X_sys[bench_inds], X_avg[bench_inds], X_mod[bench_inds], y_all[bench_inds]

    print("--- Dataset shapes (Batch, History/Horizon, Features)")

    print("Xd_trn shape",Xd_trn.shape)
    Xd_scaler = MinMaxScaler()
    nsamples, nx, ny = Xd_trn.shape
    Xd_trn = Xd_trn.reshape((nsamples,nx*ny))
    Xd_trn = Xd_scaler.fit_transform(Xd_trn)
    Xd_trn = Xd_trn.reshape((nsamples,nx,ny))
    print("Xs_trn shape",Xs_trn.shape)
    Xs_scaler = MinMaxScaler()
    nsamples, nx, ny = Xs_trn.shape
    Xs_trn = Xs_trn.reshape((nsamples,nx*ny))
    Xs_trn = Xs_scaler.fit_transform(Xs_trn)
    Xs_trn = Xs_trn.reshape((nsamples,nx,ny))
    print("Xa_trn shape",Xa_trn.shape)
    Xa_scaler = MinMaxScaler()
    Xa_trn = Xa_scaler.fit_transform(Xa_trn)
    print("y_trn shape:", y_trn.shape)
    y_scaler = MinMaxScaler()
    y_trn = y_scaler.fit_transform(y_trn.reshape(-1, 1))
    y_trn = y_trn.reshape(-1)
    print("converting train set to tensor arrays")
    Xd_trn = torch.from_numpy(Xd_trn).float()
    Xs_trn = torch.from_numpy(Xs_trn).float()
    Xa_trn = torch.from_numpy(Xa_trn).float()
    Xm_trn = torch.from_numpy(Xm_trn).float()
    y_trn = torch.from_numpy(y_trn).float()
    trn_dt = torch.utils.data.TensorDataset(Xd_trn,Xs_trn,Xa_trn,Xm_trn,y_trn)
    train_loader = torch.utils.data.DataLoader(trn_dt, batch_size=batch, shuffle=True, drop_last=True)

    print("Xd_tst shape:", Xd_tst.shape)
    nsamples, nx, ny = Xd_tst.shape
    Xd_tst = Xd_tst.reshape((nsamples,nx*ny))
    Xd_tst = Xd_scaler.transform(Xd_tst)
    Xd_tst = Xd_tst.reshape((nsamples,nx,ny))
    print("Xs_tst shape:", Xs_tst.shape)
    nsamples, nx, ny = Xs_tst.shape
    Xs_tst = Xs_tst.reshape((nsamples,nx*ny))
    Xs_tst = Xs_scaler.transform(Xs_tst)
    Xs_tst = Xs_tst.reshape((nsamples,nx,ny))
    print("Xa_tst shape:", Xa_tst.shape)
    Xa_tst = Xa_scaler.transform(Xa_tst)
    print("y_tst shape:", y_tst.shape)
    y_tst = y_scaler.transform(y_tst.reshape(-1, 1))
    y_tst = y_tst.reshape(-1)
    print("converting test set to tensor arrays")
    Xd_tst = torch.from_numpy(Xd_tst).float()
    Xs_tst = torch.from_numpy(Xs_tst).float()
    Xa_tst = torch.from_numpy(Xa_tst).float()
    Xm_tst = torch.from_numpy(Xm_tst).float()
    y_tst = torch.from_numpy(y_tst).float()
    tst_dt = torch.utils.data.TensorDataset(Xd_tst,Xs_tst,Xa_tst,Xm_tst,y_tst)
    test_loader = torch.utils.data.DataLoader(tst_dt, batch_size=batch, shuffle=False)

    return


def parse_arguments():
	parser = argparse.ArgumentParser(description="Train an LSTM network for given parameters")
	parser.add_argument("-c",    "--cuda",       type = bool,  default=False,       help="GPU available (Default: False)")
	parser.add_argument("-b",    "--batch",      type = int,   default=1024,        help="Batch size for training (Default: 32)")
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

    global model_cfg, model

    data = np.load("final_dataset.npy",allow_pickle=True)
    
    for bench in ["gbt"]:
        print("======= ",bench," =======")
        preprocess_retrain(data,args["batch"],bench)
        # preprocess(data,args["batch"],bench)

        model_cfg = {
            "dflt_seq": len(data[0][0][0]),
            "hist_seq": len(data[0][1]),
            "mean_val": len(data[0][1][0]),
            "features": int(args["features"]),
            "layers"  : int(args["layers"]),
            "lr"      :	float(args["rate"]),
            "epochs"  : int(args["epochs"]),
            "cuda"    : bool(args["cuda"])
            }

        # train_model()

        # torch.save(model, "trained_model_l1o_"+bench+".pt")
        #continue
        # train_model()
        model = torch.load("trained_model_l1o_"+bench+".pt")
        whole_retrain()
        # partial_retrain(model)
        # tst_model = test_LSTM()

        # torch.save(model, args["output"]+bench+".pt")