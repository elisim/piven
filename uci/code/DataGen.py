"""
Data creation:
Load the data, normalize it, and split into train and test.
"""
import numpy as np

DATA_PATH = "../UCI_Datasets"


class DataGenerator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # used for metrics calculation
        self.scale_c = None  # std
        self.shift_c = None  # mean

    def create_data(self, seed_in=5, train_prop=0.9):
        """
        @param seed_in: seed for numpy random seed
        @param train_prop: train proportion
        """
        np.random.seed(seed_in)

        # load UCI data
        dataset = self.dataset_name
        dataset_path = f"{DATA_PATH}/{dataset}.txt"

        if dataset == 'YearPredictionMSD':
            data = np.loadtxt(dataset_path, delimiter=',')
        elif dataset == 'naval':
            data = np.loadtxt(dataset_path)
            data = data[:, :-1]  # have 2 y as GT, ignore last
        else:
            data = np.loadtxt(dataset_path)

        # save normalization constants (used for calculating results)
        if dataset == 'YearPredictionMSD':
            scale_c = np.std(data[:, 0])  # in YearPredictionMSD, label's index = 0
            shift_c = np.mean(data[:, 0])
        else:
            scale_c = np.std(data[:, -1])
            shift_c = np.mean(data[:, -1])

        # normalize data
        for i in range(data.shape[1]):
            sdev_norm = np.std(data[:, i])
            sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm  # avoid zero variance features
            data[:, i] = (data[:, i] - np.mean(data[:, i])) / sdev_norm

        # split train test
        if dataset == 'YearPredictionMSD':
            # train: first 463,715 examples
            # test: last 51,630 examples
            train = data[:463715, :]
            test = data[-51630:, :]

        else:
            # split into train/test in random
            perm = np.random.permutation(data.shape[0])
            train_size = int(round(train_prop * data.shape[0]))
            train = data[perm[:train_size], :]
            test = data[perm[train_size:], :]

        # split to target and data
        if dataset == 'YearPredictionMSD':
            y_train = train[:, 0].reshape(-1, 1)
            X_train = train[:, 1:]
            y_val = test[:, 0].reshape(-1, 1)
            X_val = test[:, 1:]

        else:
            y_train = train[:, -1].reshape(-1, 1)
            X_train = train[:, :-1]
            y_val = test[:, -1].reshape(-1, 1)
            X_val = test[:, :-1]

        self.scale_c = scale_c
        self.shift_c = shift_c

        return X_train, y_train, X_val, y_val
