import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight


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
                     'LBXSTR_43', 'LBXSGL_43', 'LBXGH_39','BMI']

    cat_variables = ['BPQ020_40', 'HIQ011_1', 'HOQ065_13', 'KIQ026_19', 'MCQ160K_35', 'MCQ160N_35', 'MCQ220_35',
                     'MCQ365A_35', 'MCQ365D_35', 'MCQ365B_35', 'SLQ050_9', 'MCQ220', 'RIAGENDR', 'RIDRETH3']

    all_variables = int_variables + cat_variables

    for i, c in enumerate(variables):
        assert c in all_variables, "Variable {} not included".format(c)

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

    seqn = df['SEQN'].to_numpy()
    x = np.hstack(L).astype('float32')
    y = df[output_var].values.astype('float32').reshape(-1, 1)  # nn.BCELoss()
    w = 1/df['wtmec4yr_adj_norm'].values.astype('float32').reshape(-1, 1)

    # z = np.random.uniform(0, 1, len(y)).reshape(-1, 1)
    z = np.random.standard_normal(len(y)).reshape(-1, 1)
    sigma = 0.01

    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    data = {
        'seqn': seqn,
        'x': x,
        'y': y,
        'w': w,
        'z': z,
        'sigma': sigma,
        'colnames': colnames,
    }

    return data

