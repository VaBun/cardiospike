import numpy as np
from sklearn.metrics import f1_score
import torch
import random

def prepare_data(df, cols):
    y = np.array(list(df.groupby('id').apply(lambda x: x['y'].values)), dtype='object')
    df = np.array(list(df.groupby('id').apply(lambda x: x[cols].values)), dtype='object')
    return df, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f1(y_true, y_pred, tr=0.5):
    return f1_score(y_true, sigmoid(y_pred) > tr, labels=[0, 1], zero_division=1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 2)
    random.seed(seed + 4)


def optimize_grid(y_true, y_pred, metric):
    best = 0
    best_tr = 0
    for tr in np.linspace(0, 1, 10):
        _m = metric(y_true, y_pred, tr)
        if _m > best:
            best = _m
            best_tr = tr
    return best_tr
