#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

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
    if len(sample.shape) > 1:
        sample = np.squeeze(sample)
    categories = np.sort(np.unique(sample))
    new_array = np.zeros((sample.shape[0], len(categories)))
    for s in range(len(categories)):
        index = (sample == categories[s])
        new_array[index, s] = 1
    return new_array

def normalize_column(column):
    scaler = StandardScaler()
    c = np.array(column.values).reshape(-1, 1)
    feature_vector = scaler.fit_transform(c)
    return feature_vector.squeeze()

def create_dict(df, index_0, index_1, output_var, variables, all_variables, int_variables, cat_variables):
    L, colnames = [], []

    

    for i, c in enumerate(variables):
        assert c in all_variables, "Variable {} not included".format(c)
        if c in int_variables:
            #feature_vector = df[c]
            feature_vector = normalize_column(df[c])

            L.extend([np.expand_dims(feature_vector, 1)])
            colnames.extend([c])
        elif c in cat_variables:
            feature_vector = num_to_onehot(df[c].to_numpy())
            L.extend([feature_vector])
            for j in range(feature_vector.shape[1]):
                colnames.extend([c + "_" + str(j)])
    input1 = np.hstack(L).astype('float32')
    
    target = df[output_var].values.astype('float32').reshape(-1, 1)
    seqn = df['SEQN'].to_numpy().astype('float32') if 'SEQN' in df else np.arange(len(df))
    if 'weights' in df.columns:
        weights = df['weights'].astype('float32').values
    else:
        weights = np.ones(df.shape[0])  # Default weights if 'weights' column is not present
    weights = weights.astype('float32').reshape(-1, 1)

    dict = {
        'x': input1,
        'y': target,
        'w': weights,
        'colnames': colnames,
        'id': seqn
    }

    dict_0 = {
        'x': input1[index_0],
        'y': target[index_0],
        'w': weights[index_0],
        'colnames': colnames,
        'id': seqn[index_0]
    }

    dict_1 = {
        'x': input1[index_1],
        'y': target[index_1],
        'w': weights[index_1],
        'colnames': colnames,
        'id': seqn[index_1]
    }
    return dict, dict_0, dict_1

def load_data(file, variables, output_var,var_to_group):
    df = pd.read_csv(file)
    # Split based on 'mostrar' variable
    index_0 = np.where(df[var_to_group] == 0) # Adjust based on how you define groups
    index_1 = np.where(df[var_to_group] == 1) # Adjust based on how you define groups

    int_variables = variables  # Adjust as needed
    cat_variables = []  # Assuming no categorical variables; add if needed

    all_variables = int_variables + cat_variables
    return create_dict(df, index_0, index_1, output_var, variables, all_variables, int_variables, cat_variables)


## REMOVE : Only for testing
def create_dict_2(df, df2, variables):
    L, colnames = [], []
    for i, c in enumerate(variables):

        mean = np.mean(df2[c])
        std = np.std(df2[c])
        feature_vector = (df[c] - mean) / std
        L.extend([np.expand_dims(feature_vector, 1)])
        colnames.extend([c])

    input0 = np.hstack(L).astype('float32')
    target = np.zeros((200, 1))
    weights = np.zeros((200, 1))

    dict = {
        'x': input0,
        'y': target,
        'w': weights,
        'colnames': colnames
    }

    return dict


#%%

