#!/usr/bin/python3

from os import PRIO_USER
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pickle import dump
from model import HybridLSTMnoHorizon

gpu_id = 0

def test_LSTM(test_loader, model, y_test, y_pred, model_cfg):
	print("ENTERING TESTING MODE")
	cnt = 0
	model.eval()
	dt = []

	y_local_real  = []
	y_local_pred  = []
	y_remote_real = []
	y_remote_pred = []
	with torch.no_grad():
		for xd,xs,xa,xm,y in test_loader:
			st = time.time()
			out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
			predicted = out
			dt += [time.time() - st]
			for mode,i,j in zip(xm,y,predicted.cpu()):
				#print (i, " : ", j)
				
				y_pred[cnt] = j
				if (mode == 0):
					y_local_real.append(i)
					y_local_pred.append(j)
				else:
					y_remote_real.append(i)
					y_remote_pred.append(j)

				cnt = cnt + 1
		y_local_real = np.asarray(y_local_real)
		y_local_pred = np.asarray(y_local_pred)
		y_remote_real = np.asarray(y_remote_real)
		y_remote_pred = np.asarray(y_remote_pred)
		print(y_local_real)
		print(y_local_pred)
		print(y_remote_real.shape)
		print(y_remote_pred.shape)

		print("All accuracy : ", r2_score(y_test, y_pred))
		print("Local accuracy : ", r2_score(y_local_real, y_local_pred))
		print("Remote accuracy : ", r2_score(y_remote_real, y_remote_pred))
		print("Time per batch: %.4f" % np.mean(np.asarray(dt)))
		dump(y_test, open('y_test.pkl', 'wb'))
		dump(y_pred, open('y_pred.pkl', 'wb'))


def train_model(train_loader, model_cfg):
	# IF GPU IS AVAILABLE MAKE SURE TO SET THE PROPER GPU_ID ON THE TOP OF THIS FILE
	print("ENTERING TRAINING MODE")

	if (model_cfg["cuda"]):
		model = HybridLSTMnoHorizon(model_cfg).cuda(gpu_id)
	else:
		model = HybridLSTMnoHorizon(model_cfg)

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
		for i, (xd,xs,xa,xm,y) in enumerate(train_loader):
			if (model_cfg["cuda"]):
				out = model(xd.cuda(gpu_id),xs.cuda(gpu_id),xa.cuda(gpu_id),xm.cuda(gpu_id))
				loss = criterion(out, y.cuda(gpu_id))
			else:
				out = model(xd,xs,xa)
				loss = criterion(out, y)
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
	y_all=data[:,4]

	print("X_avg shape",X_avg.shape)
	print("y_all shape",y_all.shape)

	Xd_trn, Xd_tst, Xs_trn, Xs_tst, Xa_trn, Xa_tst, Xm_trn, Xm_tst, y_trn, y_tst = train_test_split(X_def, X_sys, X_avg, X_mod, y_all, test_size=0.4)
	y_prd = np.zeros(shape=y_tst.shape)
	print("--- Dataset shapes (Batch, History/Horizon, Features)")
	
	print("Xd_trn shape",X_def.shape)
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
	trn_ld = torch.utils.data.DataLoader(trn_dt, batch_size=batch, shuffle=True, drop_last=True)

	print("converting test set to tensor arrays")
	Xd_tst = torch.from_numpy(Xd_tst).float()
	Xs_tst = torch.from_numpy(Xs_tst).float()
	Xa_tst = torch.from_numpy(Xa_tst).float()
	Xm_tst = torch.from_numpy(Xm_tst).float()
	y_tst = torch.from_numpy(y_tst).float()
	tst_dt = torch.utils.data.TensorDataset(Xd_tst,Xs_tst,Xa_tst,Xm_tst,y_tst)
	tst_ld = torch.utils.data.DataLoader(tst_dt, batch_size=batch, shuffle=False)

	return(trn_ld, tst_ld, y_tst, y_prd)


def parse_arguments():
	parser = argparse.ArgumentParser(description="Train an LSTM network for given parameters")
	parser.add_argument("-c",    "--cuda",       type = bool,  default=False,       help="GPU available (Default: False)")
	parser.add_argument("-b",    "--batch",      type = int,   default=1024,          help="Batch size for training (Default: 32)")
	parser.add_argument("-l",    "--layers",     type = int,   default=4,           help="Number of LSTM stacked layers (Default: 2)")
	parser.add_argument("-f",    "--features",   type = int,   default=256,         help="Number of features on each LSTM layers (Default: 64)")
	parser.add_argument("-n",    "--rate",       type = float, default=1e-03,       help="Learning rate of the network (Default: 0.001)")
	parser.add_argument("-e",    "--epochs",     type = int,   default=150,         help="Number of training epochs of the network (Default: 100)")
	parser.add_argument("-o",    "--output",     type = str,   default="./trained-model-h_none.pt",help="Path to save the state dictionary of the trained model")
	args=vars(parser.parse_args())

	return args

if __name__== "__main__":

	args=parse_arguments()

	print("--- Arguments")
	print(args)

	data = np.load("../final_dataset_actual_horizon.npy",allow_pickle=True)

	trn_ldr, tst_ldr, y_tst, y_prd = preprocess(data,args["batch"])

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

	trnd_model = train_model(trn_ldr, model_cfg)

	tst_model = test_LSTM(tst_ldr, trnd_model, y_tst, y_prd, model_cfg)

	torch.save(trnd_model, args["output"])
	
	# print("Trained model saved in " + args["output"])

	# sys.exit(0)
