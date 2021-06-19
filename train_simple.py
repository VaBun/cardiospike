import argparse
import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GridSearchCV 



def normalize(X):
    norm = np.linalg.norm(X, axis=1)
    norm[norm == 0] = 1
    return X / norm[:,np.newaxis]


class SpikeDetectorClassifier(BaseEstimator):
    
    def __init__(self, kernels=None, thresholds=None, tolerance=20, max_difference=300, expansion=(4, 4)):
        self.tolerance = tolerance
        self.max_difference = max_difference
        self.kernels = kernels
        self.thresholds = thresholds
        self.expansion = expansion
    
    def difference(self, x, kernel_name, pad=False):
        x = x[1:] - x[:-1]
        x[np.abs(x) < self.tolerance] = 0
        x = np.clip(x, -self.max_difference, self.max_difference)
        x = np.concatenate((x, np.array([0])))
        if pad:
            radius = len(self.kernels[kernel_name]) // 2
            x = np.concatenate((np.zeros(radius, dtype=int), x, np.zeros(radius, dtype=int)))
        return x
    
    def kernel_product(self, x, kernel_name):
        window_size = len(self.kernels[kernel_name])
        x_pad = self.difference(x, kernel_name, pad=True)
        windows = np.zeros((len(x_pad), window_size))
        for i in range(len(x)-1):
            windows[i] = x_pad[i:i+window_size]
        return normalize(windows) * normalize(np.array([self.kernels[kernel_name]]))
    
    def spike_coordinates(self, x):
        spikes = set()
        for kernel_name in self.kernels:
            product = self.kernel_product(x, kernel_name)
            spikes.update(set(np.where(product.sum(axis=1) > self.thresholds[kernel_name])[0]))
        return np.array(list(spikes))
    
    def predict_one(self, x):
        spikes = self.spike_coordinates(x)
        y_pred = np.zeros_like(x, dtype=int)
        for spike_i in spikes:
            extended_i = np.arange(max(0, spike_i-self.expansion[0]), min(len(x)-1, spike_i+self.expansion[1]+1))
            y_pred[extended_i] = 1
        return y_pred
    
    def fit(self, X, y):
        pass
    
    def predict(self, X_list):
        preds = []
        for x in X_list:
            preds.append(self.predict_one(x))
        return np.concatenate(preds)
    
    def score(self, X_list, y_list):
        y_true = np.concatenate(y_list)
        y_pred = self.predict(X_list)
        return f1_score(y_true, y_pred)




def main():
    parser = argparse.ArgumentParser(description='Find best parameters for simple model')
    parser.add_argument('--data-path', default='train.csv',
                        help='path to train sample')

    args = parser.parse_args()
    data_path = args.data_path

    data = pd.read_csv(data_path)

    X = []
    Y = []
    ids_list = []
    for ids, group in data.groupby('id'):
        if ids == 132:
            continue
        ids_list.append(ids)
        X.append(group['x'].values)
        Y.append(group['y'].values)

    sdc = SpikeDetectorClassifier()

    kernels = [{'M': [0, 1, -2.2, 1, 0]}, \
               {'M': [0, 1, -2.4, 1, 0]}, \
               {'M': [0, 1, -2.6, 1, 0]}]
    thresholds = [{'M': i} for i in np.linspace(0.97, 0.98, 15)]
    tolerance = [15, 20]
    expansion = [(4, 4)]
    parameters = {'kernels': kernels, 
                  'thresholds': thresholds,
                  'tolerance': tolerance,
                  'expansion': expansion}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(sdc, parameters, cv=kf)
    gs.fit(X, Y)
    print('Best params:')
    print(gs.best_params_)
    print('Best score:')
    print(gs.best_score_)

    with open('simple_model_best_params.json', 'w') as f:
        f.write(json.dumps(gs.best_params_))



if __name__ == '__main__':
    main()
