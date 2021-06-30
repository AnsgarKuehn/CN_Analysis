import pandas as pd
import numpy as np
#path = '../Data/raw/VF_008_analysis_ref_centroid/tomo_gray_SUB_008_002_mink_val/'
#xyz_path = '../Data/raw/VF_008_analysis_ref_centroid/tomo_gray_SUB_008_002.xyz'
#save_path = ''


def read_mf(df, path):
    data = pd.read_table(path + 'w000_w100_w200_w300', delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                 names = ['value', 'type', 'closed', 'reference'])

    for tp in ['w000', 'w100', 'w200', 'w300']:
        #tp stands for type
        df[tp] = data.loc[data.type == tp, 'value']
    return df


def read_mv(df, path):
    data = pd.read_table(path + 'w010_w110_w210_w310', delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                 names = ['value_x', 'value_y', 'value_z', 'type', 'closed', 'reference'])

    for tp in ['w010', 'w110', 'w210', 'w310']:
        df[[tp+'_x', tp+'_y', tp+'_z']] = data.loc[data.type == tp, ['value_x', 'value_y', 'value_z']]
    return df



def read_mt(df, path):
    for tensor in ['w020_eigsys', 'w120_eigsys', 'w220_eigsys', 'w320_eigsys', 'w102_eigsys', 'w202_eigsys']:
        data = pd.read_table(path + tensor, delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                     names = [tensor+'Ev1', 'v1_x', 'v1_y', 'v1_z', 
                              tensor+'Ev2', 'v2_x', 'v2_y', 'v2_z',
                              tensor+'Ev3', 'v3_x', 'v3_y', 'v3_z', 'type', 'closed', 'reference'])
        df[[tensor+'Ev1', tensor+'Ev2', tensor+'Ev3']] = data[[tensor+'Ev1', tensor+'Ev2', tensor+'Ev3']]
    return df


def read_qw(df, path):
    qw = ['q0', 'w0', 'q1', 'w1', 'q2', 'w2', 'q3', 'w3', 'q4', 'w4', 'q5', 'w5', 'q6', 'w6', 'q7', 'w7', 'q8', 'w8']
    data = pd.read_table(path + 'msm_ql', delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                 names = qw + ['type', 'closed', 'reference'])
    df[qw] = data[qw]
    return df


def read_xyz(df, xyz_path):
    xyz = pd.read_table(xyz_path, delimiter = r'\s+', names = ['P', 'x', 'y', 'z'], header=1)
    xyz.index += 1
    df[['x', 'y', 'z']] = xyz.loc[df.index, ['x', 'y', 'z']]
    return df

def anisotropy_and_centroid(df):
    for j in ['0', '1', '2', '3']:
        #curvature centroids
        for i in ['x', 'y', 'z']:
            numerator = 'w'+j+'10_'+i
            denominator = 'w'+j+'00'
            df['cc_'+j+'_'+i] = df[numerator]/df[denominator] - df[i]
        
        #center of mass
        df['com_'+j] = np.sqrt(np.sum(np.square(df[['cc_'+j+'_x', 'cc_'+j+'_y', 'cc_'+j+'_z']]), axis = 1))
        #local anisotropy (covariant)
        df['cov_ani_'+j] = df['w'+j+'20_eigsysEv1']/df['w'+j+'20_eigsysEv3']
        #local anisotropy (invariant)
        if j in ['1', '2']:
            df['inv_ani_'+j] = df['w'+j+'02_eigsysEv1']/df['w'+j+'02_eigsysEv3']

    return df



def process_measurement(path, xyz_path):
    
    df = pd.DataFrame()
    df =read_mf(df, path)
    df = read_mv(df, path)
    df = read_mt(df, path)
    df = read_qw(df, path)
    df = read_xyz(df, xyz_path)
    df = anisotropy_and_centroid(df)

    return df

def cut_df(df, r_th = 30, z_th = 50):
    x_mean = df.x.mean()
    y_mean = df.y.mean()
    df['r'] = np.sqrt(np.square(df.x-x_mean)+np.square(df.y - y_mean))

    mask1 = df.r < df.r.max() - r_th
    mask2 = df.r > df.r.min() + r_th

    mask3 = df.z < df.z.max() - z_th
    mask4 = df.z > df.z.min() + z_th

    mask = np.all([mask1,mask2,mask3,mask4], axis = 0)
    return df[mask]