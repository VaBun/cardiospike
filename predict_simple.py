import argparse
import json
import numpy as np
import pandas as pd



def normalize(X):
    norm = np.linalg.norm(X, axis=1)
    norm[norm == 0] = 1
    return X / norm[:,np.newaxis]


# Version without sklearn:
class SimpleSpikeDetector:
    
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
    
    def predict(self, x):
        spikes = self.spike_coordinates(x)
        y_pred = np.zeros_like(x, dtype=int)
        for spike_i in spikes:
            extended_i = np.arange(max(0, spike_i-self.expansion[0]), min(len(x)-1, spike_i+self.expansion[1]+1))
            y_pred[extended_i] = 1
        return y_pred



def main():

    parser = argparse.ArgumentParser(description='Inference WaveNet')
    parser.add_argument('--data-path', default='test.csv',
                        help='path to test sample')
    parser.add_argument('--ss-path', default='sample_submission.csv',
                        help='path to sample submission (default: sample_submission.csv)')

    args = parser.parse_args()
    data_path = args.data_path
    ss_path = args.ss_path

    test = pd.read_csv(data_path)
    ss = pd.read_csv(ss_path)
    ss['y'] = 0

    # load params from train
    with open('simple_model_best_params.json', 'r') as f:
        model_params = json.load(f)
    ssd = SimpleSpikeDetector(**model_params)

    preds_list = []
    for ids in ss['id'].drop_duplicates().values:
        group = test[test['id'].isin([ids])]
        x = group['x'].values
        preds_list.append(ssd.predict(x))

    ss['y'] = np.concatenate(preds_list)
    ss.to_csv('sub.csv', index=False)



if __name__ == '__main__':
    main()
