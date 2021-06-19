from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LoDDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        labels = self.labels[idx].reshape(-1, 1)

        if self.transform is not None:
            data, labels = self.transform(data, labels)
        return data.astype(float), labels.astype(int)


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class Classifier(nn.Module):
    def __init__(self, inch=8, kernel_size=3):
        super().__init__()
        self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)

        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc = nn.Linear(128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.act = nn.ReLU(inplace=True)

        self.inp_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x = self.bn4(x)
        x = x.permute(0, 2, 1)

        x = self.fc(x)
        return x


class SnapshotEns:
    """In memory snapshots class."""

    def __init__(self, device: torch.device, k: int = 1, early_stopping: bool = True,
                 patience: int = 3, swa: bool = False):
        self.best_loss = np.array([np.inf] * k)
        self.k = k
        self.device = device
        self.models = [nn.Module()] * k
        self.early_stopping = early_stopping
        self.patience = patience
        self.swa = swa
        self.counter = 0
        self.early_stop = False

    def update(self, model: nn.Module, loss: float):

        if np.any(self.best_loss > loss):
            self._sort()
            pos = np.where(self.best_loss > loss)[0][-1]

            self.best_loss[pos] = loss
            self.models[pos] = deepcopy(model.eval()).cpu()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True and self.early_stopping

    def _sort(self):

        ids = np.argsort(self.best_loss)
        self.best_loss = self.best_loss[ids]
        self.models = [self.models[z] for z in ids]

    def set_weights(self, model: nn.Module, best: bool = False):

        n = 1 if best else min(self.k, sum(self.best_loss != np.inf))
        state_dict = {}
        for pos, score in enumerate(self.best_loss):

            if pos == n:
                break

            w = 1 / n

            new_state = self.models[pos].state_dict()
            # upd new state with weights
            for i in new_state.keys():
                new_state[i] = new_state[i].double() * w

            if pos == 0:
                state_dict = new_state
            else:
                # upd state
                for i in state_dict.keys():
                    state_dict[i] += new_state[i]

        model.load_state_dict(state_dict)

    def set_best_params(self, model: nn.Module):

        self._sort()
        self.set_weights(model, best=False if self.swa else True)

        # TODO: think about dropping all models if use SWA. Change state_dict and load_state_dict
        # drop empty slots
        min_k = min(self.k, sum(self.best_loss != np.inf))
        self.models = self.models[:min_k]
        self.best_loss = self.best_loss[:min_k]
