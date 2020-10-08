import numpy as np
import pandas as pd
from scipy.stats import norm


def gauss_neg_log_like(y_true, y_pred_gauss_mid, y_pred_gauss_dev, scale_c):
    """
    return negative gaussian log likelihood
    """
    n = y_true.shape[0]
    y_true = y_true.reshape(-1) * scale_c
    y_pred_gauss_mid = y_pred_gauss_mid * scale_c
    y_pred_gauss_dev = y_pred_gauss_dev * scale_c
    neg_log_like = -np.sum(norm.logpdf(y_true.reshape(-1), loc=y_pred_gauss_mid, scale=y_pred_gauss_dev))
    neg_log_like = neg_log_like / n

    return neg_log_like


def gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all, alpha=0.05):
    """
    input is individual NN estimates of mean and std dev
    1. combine into ensemble estimates of mean and std dev
    2. convert to prediction intervals
    """
    z_score_ = z_score(alpha)

    # 1. merge to one estimate (described in paper, mixture of gaussians)
    y_pred_gauss_mid = np.mean(y_pred_gauss_mid_all, axis=0)
    y_pred_gauss_dev = np.sqrt(np.mean(np.square(y_pred_gauss_dev_all) \
                                       + np.square(y_pred_gauss_mid_all), axis=0) - np.square(y_pred_gauss_mid))

    # 2. create pi's
    y_pred_U = y_pred_gauss_mid + z_score_ * y_pred_gauss_dev
    y_pred_L = y_pred_gauss_mid - z_score_ * y_pred_gauss_dev

    return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L


def pi_to_gauss(y_pred_all, method, alpha=0.05):
    """
    input is individual NN estimates of upper and lower bounds
    1. combine into ensemble estimates of upper and lower bounds
    2. convert to mean and std dev of gaussian

    @param y_pred_all: predictions. shape [no. ensemble, no. predictions, 2]
                       or [no. ensemble, no. predictions, 3] in case method is piven or only-rmse
    @param method: method name
    """
    in_ddof = 1 if y_pred_all.shape[0] > 1 else 0
    z_score_ = z_score(alpha)

    y_upper_mean, y_upper_std = np.mean(y_pred_all[:, :, 0], axis=0), np.std(y_pred_all[:, :, 0], axis=0, ddof=in_ddof)
    y_lower_mean, y_lower_std = np.mean(y_pred_all[:, :, 1], axis=0), np.std(y_pred_all[:, :, 1], axis=0, ddof=in_ddof)

    y_pred_U = y_upper_mean + z_score_ * y_upper_std / np.sqrt(y_pred_all.shape[0])
    y_pred_L = y_lower_mean - z_score_ * y_lower_std / np.sqrt(y_pred_all.shape[0])

    if method == 'qd' or method == 'mid' or method == 'deep-ens':
        v = None
    elif method == 'piven' or method == 'only-rmse':
        v = np.mean(y_pred_all[:, :, 2], axis=0)
    else:
        raise ValueError(f"Unknown method {method}")

    # need to do this before calc mid and std dev
    y_pred_U_temp = np.maximum(y_pred_U, y_pred_L)
    y_pred_L = np.minimum(y_pred_U, y_pred_L)
    y_pred_U = y_pred_U_temp

    y_pred_gauss_mid = np.mean((y_pred_U, y_pred_L), axis=0)
    y_pred_gauss_dev = (y_pred_U - y_pred_gauss_mid) / z_score_

    return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L, v


def results_to_csv(path, results, params, n_runs, n_ensemble, in_ddof):
    avg = np.mean(results, axis=0)
    std_dev = np.std(results, axis=0, ddof=in_ddof)
    std_err = std_dev / np.sqrt(n_runs)
    # fix bug: std_err was to std_dev.
    data = {'avg': np.round(avg, 3), 'std_err': np.round(std_err, 3), 'std_dev': np.round(std_dev, 3)}
    data_df = pd.DataFrame(data, index=['PICP', 'MPIW', 'CWC', 'RMSE', 'RMSE_ELI', 'NLL', 'shap_W', 'shap_p'])

    params['n_ensemble'] = n_ensemble
    params_df = pd.DataFrame.from_dict(params, orient='index', columns=['avg'])
    concat = pd.concat([data_df, params_df], axis=0, ignore_index=False, sort=True)
    concat.to_csv(path)


def np_QD_loss(y_true, y_pred_L, y_pred_U, alpha, lambda_in=8.):
    """
    manually (with np) calc the QD_hard loss
    """
    n = y_true.shape[0]
    y_U_cap = y_pred_U > y_true.reshape(-1)
    y_L_cap = y_pred_L < y_true.reshape(-1)
    k_hard = y_U_cap * y_L_cap
    PICP = np.sum(k_hard) / n
    # in case didn't capture any need small no.
    MPIW_cap = np.sum(k_hard * (y_pred_U - y_pred_L)) / (np.sum(k_hard) + 0.001)
    loss = MPIW_cap + lambda_in * np.sqrt(n) * (max(0, (1 - alpha) - PICP) ** 2)

    return loss


def z_score(alpha):
    score = norm.ppf(1 - alpha / 2)
    return round(score, 3)
