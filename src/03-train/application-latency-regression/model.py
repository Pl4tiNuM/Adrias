import torch
import torch.nn as nn
import torch.utils.data

class HybridLSTM(nn.Module):
    def __init__(self, model_cfg):
        super(HybridLSTM, self).__init__()

        self.dflt_len    = model_cfg["dflt_seq"]
        self.hist_len    = model_cfg["hist_seq"]
        self.mean_num    = model_cfg["mean_val"]
        self.hidden_size = model_cfg["features"]
        self.num_layers  = model_cfg["layers"]

        self.LSTM_default = nn.LSTM(input_size    = self.mean_num,
                            hidden_size   = 64,
                            num_layers    = 1,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.LSTM_system = nn.LSTM(input_size    = self.mean_num,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.NN = nn.Sequential(
                                nn.BatchNorm1d(64+self.hidden_size+self.mean_num+1),
                                nn.ReLU(True),
                                nn.Linear(in_features = 64+self.hidden_size+self.mean_num+1, out_features = self.hidden_size//2),
                                nn.BatchNorm1d(self.hidden_size//2),
                                nn.ReLU(True),
                                nn.Dropout(.1),
                                nn.Linear(in_features = self.hidden_size//2, out_features = 1))

                                

    def forward(self, xd,xs,xa,xm):
        xd_out, _ = self.LSTM_default(xd)
        xd_out = xd_out[:, -1, :]

        xs_out, _ = self.LSTM_system(xs)
        xs_out = xs_out[:, -1, :]

        tmp=torch.cat((xd_out, xs_out, xa, xm.reshape(-1,1)), 1)
        out = self.NN(tmp)
        return out.reshape(-1)

class HybridLSTM_LC(nn.Module):
    def __init__(self, model_cfg):
        super(HybridLSTM_LC, self).__init__()

        self.dflt_len    = model_cfg["dflt_seq"]
        self.hist_len    = model_cfg["hist_seq"]
        self.mean_num    = model_cfg["mean_val"]
        self.hidden_size = model_cfg["features"]
        self.num_layers  = model_cfg["layers"]

        self.LSTM_default = nn.LSTM(input_size    = self.mean_num,
                            hidden_size   = 64,
                            num_layers    = 1,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.LSTM_system = nn.LSTM(input_size    = self.mean_num,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.NN = nn.Sequential(
                                nn.BatchNorm1d(64+self.hidden_size+self.mean_num+1),
                                nn.ReLU(True),
                                nn.Linear(in_features = 64+self.hidden_size+self.mean_num+1, out_features = self.hidden_size//2),
                                nn.BatchNorm1d(self.hidden_size//2),
                                nn.ReLU(True),
                                nn.Dropout(.1),
                                nn.Linear(in_features = self.hidden_size//2, out_features = 2))

                                

    def forward(self, xd,xs,xa,xm):
        xd_out, _ = self.LSTM_default(xd)
        xd_out = xd_out[:, -1, :]

        xs_out, _ = self.LSTM_system(xs)
        xs_out = xs_out[:, -1, :]

        tmp=torch.cat((xd_out, xs_out, xa, xm.reshape(-1,1)), 1)
        out = self.NN(tmp)
        return out

class HybridLSTMnoHorizon(nn.Module):
    def __init__(self, model_cfg):
        super(HybridLSTMnoHorizon, self).__init__()

        self.dflt_len    = model_cfg["dflt_seq"]
        self.hist_len    = model_cfg["hist_seq"]
        self.mean_num    = model_cfg["mean_val"]
        self.hidden_size = model_cfg["features"]
        self.num_layers  = model_cfg["layers"]

        self.LSTM_default = nn.LSTM(input_size    = self.mean_num,
                            hidden_size   = 64,
                            num_layers    = 1,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.LSTM_system = nn.LSTM(input_size    = self.mean_num,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.NN = nn.Sequential(
                                nn.BatchNorm1d(64+self.hidden_size+1),
                                nn.ReLU(True),
                                nn.Linear(in_features = 64+self.hidden_size+1, out_features = self.hidden_size//2),
                                nn.BatchNorm1d(self.hidden_size//2),
                                nn.ReLU(True),
                                nn.Dropout(.1),
                                nn.Linear(in_features = self.hidden_size//2, out_features = 1))

                                

    def forward(self, xd,xs,xa,xm):
        xd_out, _ = self.LSTM_default(xd)
        xd_out = xd_out[:, -1, :]

        xs_out, _ = self.LSTM_system(xs)
        xs_out = xs_out[:, -1, :]

        tmp=torch.cat((xd_out, xs_out, xm.reshape(-1,1)), 1)
        out = self.NN(tmp)
        return out.reshape(-1)

class LSTMModel(nn.Module):
    def __init__(self, model_cfg):
        super(LSTMModel, self).__init__()

        self.input_len   = model_cfg["seq_in"]
        self.output_len  = model_cfg["out_len"]
        self.hidden_size = model_cfg["features"]
        self.num_layers  = model_cfg["layers"]
        

        self.LSTM = nn.LSTM(input_size    = self.input_len,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.NN = nn.Sequential(
                                nn.BatchNorm1d(self.hidden_size),
                                nn.ReLU(True),
                                nn.Linear(in_features = self.hidden_size, out_features = int(self.hidden_size//1.5)),
                                nn.BatchNorm1d(int(self.hidden_size//1.5)),
                                nn.ReLU(True),
                                nn.Linear(in_features = int(self.hidden_size//1.5), out_features = int(self.hidden_size//3)),
                                nn.BatchNorm1d(int(self.hidden_size//3)),
                                nn.ReLU(True),
                                nn.Dropout(.1),
                                nn.Linear(in_features = int(self.hidden_size//3), out_features = self.output_len))


    def forward(self, inp):
        nsamples, _, _ = inp.shape
        out, _ = self.LSTM(inp)
        out = out[:, -1, :]
        out = self.NN(out)
        return out.sigmoid()