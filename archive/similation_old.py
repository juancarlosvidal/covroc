import numpy as np
import pandas as pd

if __name__ == '__main__':
    pacientes = np.zeros((200, 3))
    for i in range(100):
        pacientes[i, 0] = 80
        pacientes[i, 1] = 120
    pacientes[:, 2] = np.linspace(4.5, 7, num=200)
    # pacientes[:, 2] = np.linspace(80, 150, num=200)

    df = pd.DataFrame(pacientes)
    df.columns = ['RIDAGEYR.x', 'LBXSTR_43', 'LBXGH_39']
    # df.columns = ['RIDAGEYR.x', 'LBXSTR_43', 'LBXSGL_43']
    df.to_csv('./input_sim/sim_39_80.csv')