import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight


class CustomDataset(Dataset):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        xt = torch.from_numpy(self.x[idx, :])
        yt = torch.from_numpy(self.y[idx])
        wt = torch.from_numpy(self.w[idx])
        return xt, yt, wt


def num_to_onehot(sample):
    # Get unique values
    if len(sample.shape) > 1:
        sample = np.squeeze(sample)
    categories = np.sort(np.unique(sample))
    new_array = np.zeros((sample.shape[0], len(categories)))
    for s in range(len(categories)):
        index = (sample == categories[s])
        new_array[index, s] = 1
    return new_array


def normalize_column(column):
    min_c = column.min()
    max_c = column.max()
    eps = np.finfo(float).eps
    feature_vector = (column.values - min_c) / (max_c - min_c + eps)
    return feature_vector


def load_data(file, variables, output_var):
    # Load csv
    df = pd.read_csv(file, index_col=0)

    L = []
    colnames = []
    int_variables = ['INQ020_7', 'INDFMMPI_7', 'WHD050_30', 'alcoholfrecuencia', 'RIDAGEYR.x', 'BMXHT', 'BMXWT',
                     'BMXBMI', 'BMXWAIST', 'LBXTR_64', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'BPXDI1', 'LBDSCHSI_43',
                     'LBXSTR_43', 'LBXSGL_43', 'LBXGH_39']

    cat_variables = ['BPQ020_40', 'HIQ011_1', 'HOQ065_13', 'KIQ026_19', 'MCQ160K_35', 'MCQ160N_35', 'MCQ220_35',
                     'MCQ365A_35', 'MCQ365D_35', 'MCQ365B_35', 'SLQ050_9', 'MCQ220', 'RIAGENDR', 'RIDRETH3']

    all_variables = int_variables + cat_variables

    for i, c in enumerate(variables):
        assert c in all_variables, "Variable {} not included".format(c)
        # L.extend([np.expand_dims(df[c], 1)])
        if c in int_variables:
            # Normalize to 0-1 range
            feature_vector = normalize_column(df[c])
            L.extend([np.expand_dims(feature_vector, 1)])
            colnames.extend([c])

        elif c in cat_variables:
            # One-hot encoding for categorical variables
            feature_vector = num_to_onehot(df[c].to_numpy())
            L.extend([feature_vector])
            for j in range(feature_vector.shape[1]):
                colnames.extend([c + "_" + str(j)])

    input = np.hstack(L).astype('float32')
    target = df[output_var].values.astype('float32').reshape(-1, 1)
    seqn = df['SEQN'].to_numpy().astype('float32')
    weights = np.array(df['wtmec4yr_adj_norm']) / np.sum(df['wtmec4yr_adj_norm'])
    weights = weights.astype('float32').reshape(-1, 1)

    data = {
        'x': input,
        'y': target,
        'w': weights,
        'colnames': colnames,
        'id': seqn,
        'class_0': class_0,
        'class_1': class_1
    }

    return data

