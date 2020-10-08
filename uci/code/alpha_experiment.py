"""
Run alpha experiment comparing QD and PIVEN
Run as:
    python alpha_experiment.py --dataset <ds>
    where dataset name should be in UCI_Datasets folder
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import argparse

from sklearn.metrics import mean_squared_error
from pathlib import Path
from DeepNetPI import TfNetwork
from DataGen import DataGenerator
from utils import pi_to_gauss

RESULTS_PATH = '../results-alpha'


def pickle_dump(dataset, qd_metrics_per_alpha, piven_metrics_per_alpha):
    with open(f"{RESULTS_PATH}/{dataset}/qd_metrics_per_alpha.pkl", 'wb') as f_qd:
        pickle.dump(qd_metrics_per_alpha, f_qd)

    with open(f"{RESULTS_PATH}/{dataset}/piven_metrics_per_alpha.pkl", 'wb') as f_piven:
        pickle.dump(piven_metrics_per_alpha, f_piven)


def plot(metric_name, qd_metric_lst, piven_metric_lst, alpha_values, dataset):
    plt.figure(figsize=(10, 5))
    plt.plot(alpha_values, qd_metric_lst, '-o', label="QD", markersize=10)
    plt.plot(alpha_values, piven_metric_lst, '-s', label="PIVEN", markersize=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("alpha", fontsize=20)
    plt.ylabel(metric_name, fontsize=20)
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig(f"{RESULTS_PATH}/{dataset}/{metric_name}.png", bbox_inches="tight")


def save_plots(qd_metrics_per_alpha, piven_metrics_per_alpha, alpha_values, dataset):
    qd_metrics = []
    qd_std = []
    for alpha_idx in range(len(qd_metrics_per_alpha)):
        qd_metrics.append(np.mean(qd_metrics_per_alpha[alpha_idx], axis=0))  # mean
        qd_std.append(np.std(qd_metrics_per_alpha[alpha_idx], axis=0))  # std

    piven_metrics = []
    piven_std = []
    for alpha_idx in range(len(piven_metrics_per_alpha)):
        piven_metrics.append(np.mean(piven_metrics_per_alpha[alpha_idx], axis=0))  # mean
        piven_std.append(np.std(piven_metrics_per_alpha[alpha_idx], axis=0))  # std

    qd_picp_lst = [res[0] for res in qd_metrics]
    piven_picp_lst = [res[0] for res in piven_metrics]

    qd_mpiw_lst = [res[1] for res in qd_metrics]
    piven_mpiw_lst = [res[1] for res in piven_metrics]

    qd_mse_lst = [res[2] for res in qd_metrics]
    piven_mse_lst = [res[2] for res in piven_metrics]

    plot('PICP', qd_picp_lst, piven_picp_lst, alpha_values, dataset)
    plot('MPIW', qd_mpiw_lst, piven_mpiw_lst, alpha_values, dataset)
    plot('RMSE', qd_mse_lst, piven_mse_lst, alpha_values, dataset)


def picp_mpiw(y_l_pred, y_u_pred, y_val_0):
    K_u = y_u_pred > y_val_0
    K_l = y_l_pred < y_val_0
    picp_ = np.mean(K_u * K_l)
    mpiw_ = round(np.mean(y_u_pred - y_l_pred), 3)
    return picp_, mpiw_


def ensemble_to_prediction(y_pred_all, alpha, method):
    qd_y_pred_all = np.array(y_pred_all)
    _, _, y_u_pred, y_l_pred, y_v_pred = pi_to_gauss(qd_y_pred_all, method=method, alpha=alpha)

    if method == 'piven':
        y_pred = y_v_pred * y_u_pred + (1 - y_v_pred) * y_l_pred

    else:  # qd
        y_pred = 0.5 * y_u_pred + 0.5 * y_l_pred

    return y_u_pred, y_l_pred, y_pred


def predict_model(model, x, y, sess, method):
    _, y_pred = model.predict(sess, X_test=x, y_test=y)
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]

    if method == 'piven':
        y_v_pred = y_pred[:, 2]
        y_pred = y_v_pred * y_u_pred + (1 - y_v_pred) * y_l_pred

    else:  # qd
        y_pred = 0.5 * y_u_pred + 0.5 * y_l_pred

    return y_u_pred, y_l_pred, y_pred


def train(dataset, alpha_values, params):
    runs = params['n_runs']  # number of runs
    h_size = params['h_size']  # number of hidden units
    n_epoch = params['epochs']  # number epochs to train for
    l_rate = params['lr']  # learning rate of optimizer
    decay_rate = params['decay_rate']  # learning rate decay
    lambda_in = params['lambda_in']  # lambda_in param in the loss
    sigma_in = params['sigma_in']  #  initialize std dev of NN weights
    soften = params['soften']  # soften param in the loss
    patience = params['patience']  # patience
    is_early_stop = patience != -1
    n_ensemble = 5

    if dataset == 'YearPredictionMSD':
        n_batch = 1000  # batch size
        out_biases = [5., -5.]
    else:
        n_batch = 100  # batch size
        out_biases = [3., -3.]

    qd_metrics_per_alpha = []
    piven_metrics_per_alpha = []

    print(f"Dataset = {dataset}, runs = {runs}, epochs = {n_epoch}\n")
    print("Started...")

    for alpha in alpha_values:
        qd_metrics_per_run = []
        piven_metrics_per_run = []

        for run in range(1, runs + 1):
            # fix seed
            seed = run
            np.random.seed(seed)
            tf.random.set_random_seed(seed)

            # ensemble results
            qd_y_pred_all = []
            piven_y_pred_all = []

            # create data
            gen_data = DataGenerator(dataset_name=dataset)
            X_train, y_train, X_val, y_val = gen_data.create_data(seed_in=seed)

            for i in range(n_ensemble):
                tf.reset_default_graph()
                with tf.Session() as qd_sess:
                    # create qd network
                    qd_model = TfNetwork(x_size=X_train.shape[1],
                                         y_size=2,
                                         h_size=h_size,
                                         alpha=alpha,
                                         soften=soften,
                                         lambda_in=lambda_in,
                                         sigma_in=sigma_in,
                                         out_biases=out_biases,
                                         method='qd',
                                         patience=patience,
                                         dataset=dataset)

                    # train qd
                    print(f"\nalpha = {alpha}, run = {run}, ensemble = {i + 1}, QD training...")
                    qd_model.train(qd_sess, X_train, y_train, X_val, y_val,
                                   n_epoch=n_epoch,
                                   l_rate=l_rate,
                                   decay_rate=decay_rate,
                                   is_early_stop=is_early_stop,
                                   n_batch=n_batch)

                    # predict on X_val
                    _, y_pred_qd = qd_model.predict(qd_sess, X_test=X_val, y_test=y_val)
                    qd_y_pred_all.append(y_pred_qd)

            y_u_qd, y_l_qd, y_qd = ensemble_to_prediction(qd_y_pred_all, alpha, method='qd')

            # calc metrics
            y_val_0 = y_val[:, 0]
            qd_picp, qd_mpiw = picp_mpiw(y_l_qd, y_u_qd, y_val_0)
            mse_qd = round(mean_squared_error(gen_data.scale_c * y_val_0, gen_data.scale_c * y_qd), 3)
            qd_metrics_per_run.append([qd_picp, qd_mpiw, mse_qd ** 0.5])

            for i in range(n_ensemble):
                tf.reset_default_graph()
                with tf.Session() as piven_sess:
                    # create piven network
                    piven_model = TfNetwork(x_size=X_train.shape[1],
                                            y_size=2,
                                            h_size=h_size,
                                            alpha=alpha,
                                            soften=soften,
                                            lambda_in=lambda_in,
                                            sigma_in=sigma_in,
                                            out_biases=out_biases,
                                            method='piven',
                                            patience=patience,
                                            dataset=dataset)

                    # train piven
                    print(f"\nalpha = {alpha}, run = {run}, ensemble = {i + 1}, PIVEN training...")
                    piven_model.train(piven_sess, X_train, y_train, X_val, y_val,
                                      n_epoch=n_epoch,
                                      l_rate=l_rate,
                                      decay_rate=decay_rate,
                                      is_early_stop=is_early_stop,
                                      n_batch=n_batch)

                    # predict on X_val
                    _, y_pred_piven = piven_model.predict(piven_sess, X_test=X_val, y_test=y_val)
                    piven_y_pred_all.append(y_pred_piven)

            y_u_piven, y_l_piven, y_piven = ensemble_to_prediction(piven_y_pred_all, alpha, method='piven')

            # calc metrics
            y_val_0 = y_val[:, 0]
            piven_picp, piven_mpiw = picp_mpiw(y_l_piven, y_u_piven, y_val_0)
            mse_piven = round(mean_squared_error(gen_data.scale_c * y_val_0, gen_data.scale_c * y_piven), 3)
            piven_metrics_per_run.append([piven_picp, piven_mpiw, mse_piven ** 0.5])

        # save metrics
        qd_metrics_per_alpha.append(qd_metrics_per_run)
        piven_metrics_per_alpha.append(piven_metrics_per_run)

    return qd_metrics_per_alpha, piven_metrics_per_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, metavar='', help='dataset name, from UCI_Datasets folder', required=True)
    args = parser.parse_args()
    dataset = args.dataset
    alpha_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    with open('params.json') as params_json:
        all_params = json.load(params_json)
        try:
            params = next(el for el in all_params if el['dataset'] == dataset)
        except StopIteration:
            raise ValueError(f"Invalid dataset name: {dataset}")

    Path(f"{RESULTS_PATH}/{dataset}").mkdir(parents=True, exist_ok=True)
    qd_metrics_per_alpha, piven_metrics_per_alpha = train(dataset, alpha_values, params)
    print(f"\nDone {dataset}. Saving results...")
    pickle_dump(dataset, qd_metrics_per_alpha, piven_metrics_per_alpha)
    save_plots(qd_metrics_per_alpha, piven_metrics_per_alpha, alpha_values, dataset)


if __name__ == "__main__":
    main()
