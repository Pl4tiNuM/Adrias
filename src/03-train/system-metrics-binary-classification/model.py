import torch
import torch.nn as nn
import torch.utils.data

class LSTMModel(nn.Module):
    def __init__(self, model_cfg):
        super(LSTMModel, self).__init__()

        self.input_len   = model_cfg["seq_in"]
        if (model_cfg["classes"] == 2):
            self.nclasses    = 1
        else:
            self.nclasses    = model_cfg["classes"]
        self.hidden_size = model_cfg["features"]
        self.num_layers  = model_cfg["layers"]

        self.LSTM = nn.GRU(input_size    = self.input_len,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = 0,
                            bias          = True,
                            batch_first   = True,
                            bidirectional = False)

        self.NN = nn.Sequential(
                                nn.BatchNorm1d(self.hidden_size),
                                nn.ReLU(True),
                                nn.Linear(in_features = self.hidden_size, out_features = self.hidden_size//2),
                                nn.BatchNorm1d(self.hidden_size//2),
                                nn.ReLU(True),
                                nn.Dropout(.1),
                                nn.Linear(in_features = self.hidden_size//2, out_features = self.nclasses))

    def forward(self, inp):
        out, _ = self.LSTM(inp)
        out = out[:, -1, :]
        out = self.NN(out)
        return out.reshape(inp.shape[0])