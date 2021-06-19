from __future__ import print_function

import argparse
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import torch

from utils.utils import prepare_data
from utils.model import LoDDataset, Classifier
from utils.augs import AugCompose, NormalizeExtractXRollingMedian, FillNAorINF

DEVICE = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='Inference WaveNet')
    parser.add_argument('--data-path', default='test.csv',
                        help='path to test sample')
    parser.add_argument('--ss-path', default='sample_submission.csv',
                        help='path to sample submission (default: sample_submission.csv)')

    args = parser.parse_args()
    data_path = args.data_path
    ss_path = args.ss_path

    # load train folds statistics
    with open('./models/statistics_fold.pkl', 'rb') as f:
        statistics = pickle.load(f)

    test_augs = AugCompose([NormalizeExtractXRollingMedian(), FillNAorINF()])
    ss = pd.read_csv(ss_path)
    ss['y'] = 0

    # iterate over kfold trained models
    for n in range(5):

        # prepare data
        _mean = statistics[n]['mean']
        _std = statistics[n]['std']

        test = pd.read_csv(data_path)
        test['y'] = test['id'].values

        _test = test.copy()
        _test['x_diff'] = _test.groupby('id')['x'].diff().fillna(0)
        _test['x_diff'] = (_test['x_diff'] - _mean) / _std

        xtest, ytest = prepare_data(_test, ['x', 'x_diff'])

        # load model
        model = Classifier(inch=2)
        model.load_state_dict(torch.load(f'./models/fold_{n}'))
        model.to(DEVICE)
        model.eval()

        test_dataset = LoDDataset(xtest, ytest, transform=test_augs)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for xs, _id in test_loader:
                xs = xs.to(DEVICE, dtype=torch.float)

                ypred = model(xs)
                ypred = ypred.view(-1)
                _ids = _id.data.cpu().numpy()[0, 0, 0]
                pred = (ypred.sigmoid().data.cpu().numpy() > 0.5).astype(int)
                ss.loc[ss.id == _ids, 'y'] += pred

    # apply voting
    ss.loc[:, 'y'] = ((ss.loc[:, 'y']) > 3).astype(int)

    # post-processing: remove all ones chains less than 6
    for_remove = []
    curr = []
    curr_count = 0
    for n, val in enumerate(ss.y.values):
        if val == 1:
            curr.append(n)
            curr_count += 1
        else:
            if curr_count < 6:
                for_remove.extend(curr)
            curr = []
            curr_count = 0
    for i in reversed(for_remove):
        ss.iloc[i, 2] = 0

    ss[['id', 'time', 'y']].to_csv('nn_final.csv', index=False)

    print('Done.')


if __name__ == '__main__':
    main()
