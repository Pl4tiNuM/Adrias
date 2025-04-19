#!/usr/bin/python3

import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pickle import dump
from model import LSTMModel

gpu_id = 0

def test_LSTM(test_loader, model, y_test, y_pred, model_cfg):
	print("ENTERING TESTING MODE")
	cnt = 0
	model.eval()
	dt = []
	with torch.no_grad():
		for X, y in test_loader:
			st = time.time()
			out = model(X.cuda(gpu_id))
			predicted = out
			dt += [time.time() - st]
			for i,j in zip(y,predicted.cpu()):
				#print (i, " : ", j)
				y_pred[cnt] = j
				cnt = cnt + 1
		# print(y_test)
		# for i in y_pred:
		# 	print(i)
		print("accuracy: ", r2_score(y_test, y_pred))
		# print("Time per batch: %.4f" % np.mean(np.asarray(dt)))
		dump(y_test, open('y_test.pkl', 'wb'))
		dump(y_pred, open('y_pred.pkl', 'wb'))

def train_LSTM(train_loader, model_cfg):
	# IF GPU IS AVAILABLE MAKE SURE TO SET THE PROPER GPU_ID ON THE TOP OF THIS FILE
	print("ENTERING TRAINING MODE")
	if (model_cfg["cuda"]):
		model = LSTMModel(model_cfg).cuda(gpu_id)
	else:
		model = LSTMModel(model_cfg)

	print(model)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*model_cfg["epochs"]), int(.75*model_cfg["epochs"])], gamma=0.1)


	model.train()
	# Train the model
	total_step = len(train_loader)
	st = time.time()
	for epoch in range(model_cfg["epochs"]):
		for i, (seq,y) in enumerate(train_loader):
			if (model_cfg["cuda"]):
				out = model(seq.cuda(gpu_id))
				loss = criterion(out, y.cuda(gpu_id))
			else:
				out = model(seq)
				loss = criterion(out, y)
			# print(out,":",y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (i+1) % 10 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, model_cfg["epochs"], i+1, total_step, loss.item()))
		scheduler.step()
	st = time.time() - st

	return model


def preprocess(X,y,batch):
	X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2)
	y_prd = np.zeros(shape=y_tst.shape)
	print("--- Dataset shapes (Batch, History/Horizon, Features)")
	
	print("X_trn shape: ", X_trn.shape)
	X_scaler = MinMaxScaler()
	nsamples, nx, ny = X_trn.shape
	X_trn = X_trn.reshape((nsamples,nx*ny))
	X_trn = X_scaler.fit_transform(X_trn)
	X_trn = X_trn.reshape((nsamples,nx,ny))

	y_scaler = MinMaxScaler()
	y_trn = y_scaler.fit_transform(y_trn)

	print("X_tst shape:", X_tst.shape)
	nsamples, nx, ny = X_tst.shape
	X_tst = X_tst.reshape((nsamples,nx*ny))
	X_tst = X_scaler.transform(X_tst)
	X_tst = X_tst.reshape((nsamples,nx,ny))
	y_tst = y_scaler.transform(y_tst)

	print("converting train set to tensor arrays")
	X_trn = torch.from_numpy(X_trn).float()
	y_trn = torch.from_numpy(y_trn).float()
	trn_dt = torch.utils.data.TensorDataset(X_trn,y_trn)
	trn_ld = torch.utils.data.DataLoader(trn_dt, batch_size=batch, shuffle=False, drop_last=True)
	print("X_trn shape: ", X_trn.shape)
	print("y_trn shape: ", y_trn.shape)

	print("converting test set to tensor arrays")
	X_tst = torch.from_numpy(X_tst).float()
	y_tst = torch.from_numpy(y_tst).float()
	tst_dt = torch.utils.data.TensorDataset(X_tst,y_tst)
	tst_ld = torch.utils.data.DataLoader(tst_dt, batch_size=batch, shuffle=False)
	print("X_tst shape:", X_tst.shape)
	print("y_tst shape:", y_tst.shape)
	print("y_prd shape:", y_prd.shape)

	dump(X_scaler, open('x_scaler.pkl', 'wb'))
	# print("X_scaler saved in " + "../../.model/x_scaler.pkl")
	dump(y_scaler, open('y_scaler.pkl', 'wb'))
	# print("y_scaler saved in " + "../../.model/y_scaler.pkl")

	return(trn_ld, tst_ld, y_tst, y_prd)


def parse_arguments():
	parser = argparse.ArgumentParser(description="Train an LSTM network for given parameters")
	parser.add_argument("-c",    "--cuda",       type = bool,  default=False,       help="GPU available (Default: False)")
	parser.add_argument("-x",    "--X_dataset",  type = str,   default=None,        help="Dataset file containing input features")
	parser.add_argument("-y",    "--y_dataset",  type = str,   default=None,        help="Dataset file containing output predictions")
	parser.add_argument("-b",    "--batch",      type = int,   default=512,          help="Batch size for training (Default: 32)")
	parser.add_argument("-l",    "--layers",     type = int,   default=4,           help="Number of LSTM stacked layers (Default: 2)")
	parser.add_argument("-f",    "--features",   type = int,   default=256,         help="Number of features on each LSTM layers (Default: 64)")
	parser.add_argument("-n",    "--rate",       type = float, default=1e-03,       help="Learning rate of the network (Default: 0.001)")
	parser.add_argument("-e",    "--epochs",     type = int,   default=150,         help="Number of training epochs of the network (Default: 100)")
	parser.add_argument("-o",    "--output",     type = str,   default="./model.pt",help="Path to save the state dictionary of the trained model")
	args=vars(parser.parse_args())

	return args

if __name__== "__main__":

	args=parse_arguments()

	print("--- Arguments")
	print(args)

	X_all = np.load(args["X_dataset"])
	y_all = np.load(args["y_dataset"])
	trn_ldr, tst_ldr, y_tst, y_prd = preprocess(X_all,y_all,args["batch"])

	model_cfg = {
		"seq_in"  : X_all.shape[2],
		"out_len" : y_all.shape[1],
		"features": int(args["features"]),
		"layers"  : int(args["layers"]),
		"lr"      :	float(args["rate"]),
		"epochs"  : int(args["epochs"]),
		"cuda"    : bool(args["cuda"])
		}

	trnd_model = train_LSTM(trn_ldr, model_cfg)

	tst_model = test_LSTM(tst_ldr, trnd_model, y_tst, y_prd, model_cfg)

	dump(trnd_model, open('model.pkl', 'wb'))

	# torch.save(trnd_model.state_dict(), "../../.model/model.pt")
	
	# print("Trained model saved in " + "../../.model/model.pt")

	# sys.exit(0)