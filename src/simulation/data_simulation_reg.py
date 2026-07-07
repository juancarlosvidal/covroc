import pandas as pd
import numpy as np
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
    scaler = StandardScaler()
    c = np.array(column.values).reshape(-1, 1)
    feature_vector = scaler.fit_transform(c)
    return feature_vector.squeeze()


def create_dict(df, index_0, index_1, output_var, variables, all_variables, int_variables, cat_variables):
    L, colnames = [], []
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

    input = np.hstack(L).astype('float32')
    target = df[output_var].values.astype('float32').reshape(-1, 1)
    # target = np.log(df[output_var].values.astype('float32').reshape(-1, 1))
    # target = -np.log(df[output_var].values.astype('float32').reshape(-1, 1))
    # target = -df[output_var].values.astype('float32').reshape(-1, 1)
    seqn = df['SEQN'].to_numpy().astype('float32') if 'SEQN' in df else np.arange(len(df))
    if 'weights' in df.columns:
        weights = df['weights'].astype('float32').values
    else:
        weights = np.ones(df.shape[0])  # Default weights if 'weights' column is not present
    weights = weights.astype('float32').reshape(-1, 1)


    dict = {
        'x': input,
        'y': target,
        'w': weights,
        'colnames': colnames,
        'id': seqn
    }

    dict_0 = {
        'x': input[index_0],
        'y': target[index_0],
        'w': weights[index_0],
        'colnames': colnames,
        'id': seqn[index_0]
    }

    dict_1 = {
        'x': input[index_1],
        'y': target[index_1],
        'w': weights[index_1],
        'colnames': colnames,
        'id': seqn[index_1]
    }

    return dict, dict_0, dict_1


def create_dict_2(df, df2, variables):
    L, colnames = [], []
    for i, c in enumerate(variables):

        # L.extend([np.expand_dims(df[c], 1)])
        # Normalize to 0-1 range
        mean = np.mean(df2[c])
        std = np.std(df2[c])
        feature_vector = (df[c] - mean) / std
        L.extend([np.expand_dims(feature_vector, 1)])
        # feature_vector = df[c]
        # L.extend([np.expand_dims(feature_vector, 1)])
        colnames.extend([c])

    input = np.hstack(L).astype('float32')
    target = np.zeros((200, 1))
    weights = np.zeros((200, 1))

    dict = {
        'x': input,
        'y': target,
        'w': weights,
        'colnames': colnames
    }

    return dict

def load_data(file, variables, output_var,var_to_group):
    # Load csv
    df1 = pd.read_csv(file)
    index_0 = np.where((df1[var_to_group] == 0)  )
    index_1 = np.where((df1[var_to_group] == 1) )
    # index_0 = np.where(df['mortstat'] == 0)
    # index_1 = np.where(df['mortstat'] == 1)

    
    int_variables = variables  # Assuming 'x' and 'mostrar' are numerical
    cat_variables = []  # Assuming no categorical variables; add if needed

    all_variables = int_variables + cat_variables

    return create_dict(df1, index_0, index_1, output_var, variables, all_variables, int_variables, cat_variables)



