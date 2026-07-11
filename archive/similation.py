import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.DataFrame()

    age = np.linspace(20, 80, num=61)
    lgh43 = np.linspace(80, 150, num=141)
    [age_grid, glh43_grid] = np.meshgrid(age, lgh43)
    data = np.array([age_grid, glh43_grid]).reshape(2, -1).T

    age = np.arange(20, 80)
    age_col = np.repeat(age, 200)
    str43_col = np.repeat(1, 60*200)
    # lgh39 = np.linspace(4.5, 7, num=200)
    # lgh39_col = np.repeat(lgh39, 60)
    lgh43 = np.linspace(80, 150, num=200)
    lgh43_col = np.repeat(lgh43, 60)

    df['RIDAGEYR.x'] = age_col
    # df['LBXGH_39'] = lgh39_col
    # df.to_csv('./input_sim/sim_39_all.csv')
    df['LBXSGL_43'] = lgh43_col
    df.to_csv('./input_sim/sim_43_all.csv')