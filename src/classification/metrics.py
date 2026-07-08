# Weighted MAE/MSE/R2 used to track validation performance of the mean/variance networks.
import numpy as np
import torch
from sklearn.metrics import r2_score

def compute_regression_metrics(weights, pred, labels):
    weights = np.squeeze(weights)
    dif = np.squeeze(np.asarray(np.abs(pred - labels)))
    weighted_dif = np.multiply(weights, dif)
    mae = np.sum(weighted_dif) / np.sum(weights)

    sq_dif = np.power(dif, 2)
    weighted_sq_dif = np.multiply(weights, sq_dif)
    mse = np.sum(weighted_sq_dif) / np.sum(weights)

    norm_weights = weights / np.sum(weights)
    a = np.sum(np.multiply(np.power(dif, 2), norm_weights))
    b1 = np.squeeze(labels)
    # b2 = np.sum(np.multiply(np.squeeze(labels), norm_weights))
    b2 = np.dot(np.squeeze(labels), norm_weights)
    b = np.sum(np.multiply(np.power(np.subtract(b1, b2), 2), norm_weights))
    # print('a {} b {} b1 {} b2 {}'.format(a, b, b1, b2))
    r2 = 1 - (a / b)
    # print('Labels {} Pred {}'.format(labels.shape, pred.shape))
    r2 = r2_score(labels, pred)
    # print('R2 a {} - b {} - b1 {} - b2 {} - w {} - a/b {}'.format(a, b, b1, b2, np.dot(np.squeeze(labels), norm_weights), a/b))
    metrics = {'mae': mae,
               'mse': mse,
               'r2': r2}

    return metrics


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
