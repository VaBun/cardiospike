import random
import numpy as np
import pandas as pd
from copy import deepcopy


class BasicAug:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, data, y):
        data = deepcopy(data)
        y = deepcopy(y)
        if random.random() <= self.p:
            data, y = self.transform(data, y)

        return data, y

    def transform(self, data, y):
        raise NotImplementedError()


class AugCompose(BasicAug):
    def __init__(self, transforms):
        super(AugCompose, self).__init__(1)
        self.transforms = transforms

    def __call__(self, data, y):
        for idx, t in enumerate(self.transforms):
            data, y = t(data, y)
        return data, y


class AugOneOf(BasicAug):
    def __init__(self, transforms, p):
        super(AugOneOf, self).__init__(1)
        self.p = p
        self.transforms = transforms

    def __call__(self, data, y):
        t = random.choices(self.transforms, self.p)[0]
        data, y = t(data, y)
        return data, y


def pad_both(x, val, size_left, size_right):
    if not isinstance(val, (list, tuple)):
        val = [val]
    append_left = np.repeat(np.array(val), size_left).reshape(len(val), size_left).T
    append_right = np.repeat(np.array(val), size_right).reshape(len(val), size_right).T

    return np.concatenate([append_left, x, append_right])


class PadIfNeeded(BasicAug):

    def __init__(self, size, kind='both', r='random', p=1, x_val=None, y_val=0):
        super(PadIfNeeded, self).__init__(p)
        self.size = size
        self.kind = kind
        self.r = r
        self.x_val = x_val
        self.y_val = y_val

    def transform(self, data, y):

        if len(y) < self.size:
            pad_size = self.size - len(y)
            if self.kind == 'right':
                pad_left = 0
                pad_right = pad_size

            elif self.kind == 'left':
                pad_left = pad_size
                pad_right = 0
            elif self.kind == 'both':
                if self.r == 'random':
                    split = random.choice(range(pad_size)) + 1
                else:
                    split = pad_size // 2
                pad_left = pad_size - split
                pad_right = split

            else:
                if random.random() <= 0.5:
                    pad_left = pad_size
                    pad_right = 0
                else:
                    pad_left = 0
                    pad_right = pad_size
            if self.x_val is None:
                x_val = list(np.median(data, axis=0))
            else:
                x_val = self.x_val
            data = pad_both(data, x_val, pad_left, pad_right)
            y = pad_both(y, self.y_val, pad_left, pad_right)

        return data, y


class RandomSample(BasicAug):

    def __init__(self, size, p=1):
        super(RandomSample, self).__init__(p)
        self.size = size

    def transform(self, data, y):
        # sample left border from non positive points
        center = random.choice(np.argwhere(y[:, 0] == 0).flatten())
        # find left and right borders
        pad_size = self.size
        split = random.choice(range(pad_size)) + 1
        left = np.maximum(0, center - (pad_size - split))
        right = np.minimum(len(y) - 1, center + split)

        diffs = np.diff(y[:, 0])

        # check if right not in ones interval:

        if (y[right, 0] == 1) and (right != len(y) - 1):
            right = np.argwhere(diffs[:right] == 1).flatten()[-1]

        # check if left not in ones interval:
        if y[left, 0] == 1:
            left = np.argwhere(diffs[left:] == -1).flatten()[0] + left + 1

        left = np.maximum(0, left)
        right = np.minimum(len(y) - 1, right)

        return data[left:right], y[left:right]


class RandomSampleWithPositive(BasicAug):

    def __init__(self, size, p=1):
        super(RandomSampleWithPositive, self).__init__(p)
        self.size = size

    def transform(self, data, y):
        # sample some point from positive class
        positive_point = random.choice(np.argwhere(y[:, 0] == 1).flatten())
        # find it borders
        diffs = np.diff(y[:, 0])
        _d = diffs[:positive_point] == 1
        left_chosen_positive = np.argwhere(_d).flatten()[-1] + 1 if _d.sum() > 0 else 0
        _d = diffs[positive_point:] == -1
        right_chosen_positive = np.argwhere(_d).flatten()[0] + positive_point if _d.sum() > 0 else len(y) - 1
        chosen_len = right_chosen_positive - left_chosen_positive

        # find left and right borders
        pad_size = self.size - chosen_len
        split = random.choice(range(pad_size)) + 1
        left = np.maximum(0, left_chosen_positive - (pad_size - split))
        right = np.minimum(len(y) - 1, right_chosen_positive + split)
        # check if right not in ones interval:
        if (y[right, 0] == 1) and (right != len(y) - 1):
            right = np.argwhere(diffs[:right] == 1).flatten()[-1]

        # check if left not in ones interval:
        if (y[left, 0] == 1) and (left != left_chosen_positive):
            left = np.argwhere(diffs[left:] == -1).flatten()[0] + left + 1

        left = np.maximum(0, left)
        right = np.minimum(len(y) - 1, right)

        return data[left:right], y[left:right]


class NormalizeExtractXMean(BasicAug):

    def __init__(self, p=1):
        super(NormalizeExtractXMean, self).__init__(p)

    def transform(self, data, y):
        data[:, 0] = data[:, 0] - np.mean(data[:, 0])
        return data, y


class NormalizeExtractXMedian(BasicAug):

    def __init__(self, p=1):
        super(NormalizeExtractXMedian, self).__init__(p)

    def transform(self, data, y):
        data[:, 0] = data[:, 0] - np.median(data[:, 0])
        return data, y


class NormalizeExtractXRollingMedian(BasicAug):

    def __init__(self, window=30, tr=200, p=1):
        super(NormalizeExtractXRollingMedian, self).__init__(p)
        self.window = window
        self.tr = tr

    def transform(self, data, y):
        _data = pd.Series(data[:, 0]).fillna(600)
        if len(_data) < self.tr:
            median = np.array([np.median(_data.values)] * len(_data))
        else:
            median = _data.rolling(self.window).apply(np.median)
            median = median.fillna(median[~median.isna()].values[0]).values
        data[:, 0] = data[:, 0] - median

        return data, y


class NormalizeExtractXRollingMean(BasicAug):

    def __init__(self, window=30, p=1):
        super(NormalizeExtractXRollingMean, self).__init__(p)
        self.window = window

    def transform(self, data, y):
        _data = pd.Series(data[:, 0]).fillna(600)
        if len(_data) < 200:
            mean = np.array([np.mean(_data.values)] * len(_data))
        else:
            mean = _data.rolling(self.window).apply(np.mean)
            mean = mean.fillna(mean[~mean.isna()].values[0]).values
        data[:, 0] = data[:, 0] - mean

        return data, y


class NormalizeExtractAllRollingMedian(BasicAug):

    def __init__(self, window=30, p=1):
        super(NormalizeExtractAllRollingMedian, self).__init__(p)
        self.window = window

    def transform(self, data, y):
        for _i in range(data.shape[1]):
            _data = pd.Series(data[:, _i]).fillna(600)
            if len(_data) < 200:
                median = np.array([np.median(_data.values)] * len(_data))
            else:
                median = _data.rolling(self.window).apply(np.median)
                median = median.fillna(median[~median.isna()].values[0]).values
            data[:, _i] = data[:, _i] - median

        return data, y


class FillNAorINF(BasicAug):

    def __init__(self, p=1):
        super(FillNAorINF, self).__init__(p)

    def transform(self, data, y):
        data[data == np.inf] = 0
        data[data == -np.inf] = 0
        data[np.isnan(data)] = 0
        y[y == np.inf] = 0
        y[y == -np.inf] = 0
        y[np.isnan(y)] = 0
        return data, y
