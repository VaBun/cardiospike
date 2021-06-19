from __future__ import print_function

import argparse
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils.utils import prepare_data, f1, set_seed, optimize_grid
from utils.model import LoDDataset, Classifier, SnapshotEns
from utils.augs import (AugCompose, AugOneOf,
                        NormalizeExtractXRollingMedian,
                        PadIfNeeded, FillNAorINF,
                        RandomSample, RandomSampleWithPositive)

DEVICE = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='Train WaveNet')
    parser.add_argument('--data-path', default='train.csv',
                        help='path to train sample')

    args = parser.parse_args()
    data_path = args.data_path

    N_EPOCHS = 50
    valid_batch_size = 1  # compute full sequence
    train_batch_size = 8
    train_seq_size = 100

    data = pd.read_csv(data_path)

    train_augs = AugCompose([PadIfNeeded(size=train_seq_size),
                             AugOneOf([RandomSample(size=train_seq_size),
                                       RandomSampleWithPositive(size=train_seq_size)], p=[0.3, 0.7]),
                             PadIfNeeded(size=train_seq_size),
                             NormalizeExtractXRollingMedian(tr=200),
                             FillNAorINF()
                             ])

    valid_augs = AugCompose([NormalizeExtractXRollingMedian(), FillNAorINF()])

    # create validation splits
    ids = np.unique(data.id)
    folds = list(KFold(n_splits=5, random_state=0, shuffle=True).split(ids))

    set_seed(0)
    statistics = {}
    cols = ['x', 'x_diff']
    for n, (tr, te) in enumerate(folds):
        # prepare data

        # remove 132 id due to hight S2N
        ids_train = sorted(list(set(ids[tr]) - {132}))
        ids_val = sorted(list(set(ids[te]) - {132}))

        _train = data[data.id.isin(ids_train)].copy()
        _train['x_diff'] = _train.groupby('id')['x'].diff().fillna(0)
        _mean = _train['x_diff'].mean()
        _std = _train['x_diff'].std()
        _train['x_diff'] = (_train['x_diff'] - _mean) / _std

        _valid = data[data.id.isin(ids_val)].copy()
        _valid['x_diff'] = _valid.groupby('id')['x'].diff().fillna(0)
        _valid['x_diff'] = (_valid['x_diff'] - _mean) / _std
        statistics[n] = {'mean': _mean, 'std': _std}
        xtrain, ytrain = prepare_data(_train, cols)
        xvalid, yvalid = prepare_data(_valid, cols)

        net = Classifier(inch=len(cols))
        net.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters())

        sn = SnapshotEns(device=DEVICE, k=5, early_stopping=False, patience=50, swa=True)

        train_dataset = LoDDataset(xtrain, ytrain, transform=train_augs)
        valid_dataset = LoDDataset(xvalid, yvalid, transform=valid_augs)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

        print('Train shape:', len(train_dataset))
        print('Valid shape:', len(valid_dataset))
        f1_log = []
        train_pred = torch.Tensor([]).to(DEVICE)
        train_y = torch.Tensor([]).to(DEVICE)
        for epoch in range(N_EPOCHS):
            train_loss = []

            net.train()
            for xs, ys in tqdm(train_loader):
                # get the inputs
                xs, ys = xs.to(DEVICE, dtype=torch.float), ys.to(DEVICE, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(xs)
                outputs = outputs.view(-1)
                _y = ys.view(-1)

                loss = criterion(outputs, _y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)

                optimizer.step()
                train_loss.append(loss.item())
                train_pred = torch.cat([train_pred, outputs.data], 0)
                train_y = torch.cat([train_y, _y.data], 0)

            # scheduler.step()
            # evaluate
            net.eval()
            valid_pred = torch.Tensor([]).to(DEVICE)
            valid_y = torch.Tensor([]).to(DEVICE)
            tr_tr = optimize_grid(train_y.cpu().numpy().flatten(), train_pred.cpu().numpy(), f1)

            valid_loss = []
            with torch.no_grad():
                for xs, ys in valid_loader:
                    xs, ys = xs.to(DEVICE, dtype=torch.float), ys.to(DEVICE, dtype=torch.float)

                    ypred = net(xs)
                    ypred = ypred.view(-1)
                    _y = ys.view(-1)

                    valid_pred = torch.cat([valid_pred, ypred], 0)
                    valid_y = torch.cat([valid_y, _y], 0)

                    valid_loss.append(criterion(ypred, _y).item())

            vl_tr = optimize_grid(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy(), f1)
            f1_epoch = f1(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy())
            f1_tr = f1(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy(), tr_tr)
            f1_vl = f1(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy(), vl_tr)

            f1_log.append(f1_epoch)
            sn.update(net, -f1_epoch)
            print('Epoch {} train loss = {:.5f}; valid loss = {:.5f}; valid F1 = {:.5f}'.format(epoch,
                                                                                                np.mean(train_loss),
                                                                                                np.mean(valid_loss),
                                                                                                f1_epoch))
            print('Epoch {} train best tr F1 = {:.5f}, val best tr F1 = {:.5f}'.format(epoch, f1_tr, f1_vl))
        print('Fold: {}'.format(n))
        print('Max F1: {:.5f}'.format(np.max(f1_log)))
        print('Epoch #' + str(np.argmax(f1_log)))
        sn.set_best_params(net)
        net.eval()
        valid_pred = torch.Tensor([]).to(DEVICE)
        valid_y = torch.Tensor([]).to(DEVICE)
        tr_tr = optimize_grid(train_y.cpu().numpy().flatten(), train_pred.cpu().numpy(), f1)

        valid_loss = []
        with torch.no_grad():
            for xs, ys in valid_loader:
                xs, ys = xs.to(DEVICE, dtype=torch.float), ys.to(DEVICE, dtype=torch.float)

                ypred = net(xs)
                ypred = ypred.view(-1)
                _y = ys.view(-1)

                valid_pred = torch.cat([valid_pred, ypred], 0)
                valid_y = torch.cat([valid_y, _y], 0)

                valid_loss.append(criterion(ypred, _y).item())

        vl_tr = optimize_grid(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy(), f1)
        f1_epoch = f1(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy())
        f1_vl = f1(valid_y.cpu().numpy().flatten(), valid_pred.cpu().numpy(), vl_tr)

        f1_log.append(f1_epoch)
        print('F1 = {:.5f}, val best tr F1 = {:.5f}'.format(f1_epoch, f1_vl))
        torch.save(net.state_dict(), f'./models/fold_{n}')
        print('==========')

    # save statistics
    with open('./models/statistics_fold.pkl', 'wb') as f:
        pickle.dump(statistics, f)

    print('Done.')


if __name__ == '__main__':
    main()
