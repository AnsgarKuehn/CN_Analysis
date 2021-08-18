import pandas as pd
import numpy as np
import os
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cn_tools.cns_fit import approximate_diameter

def prepare_df(df, index):
    '''
    Computes the radius w.r.t. the zylinder as well as the local package fraction.
    Also removes all datapoints too close to a boarder or with unphysical volume fraction (see cut_df)

            Parameters:
                    df (pd.DataFrame): DataFrame containing preprocessed minkowski data.
                    index (str) XXX_YYY Determining the measurement and indicating the resolution
                    
            Returns:
                    DataFrame
    '''
    df['lpf'] = (4/3*np.pi*(df.diameter/2)**3)/df.w000
    
    x_mean = df.x.mean()
    y_mean = df.y.mean()
    df['r'] = np.sqrt(np.square(df.x-x_mean)+np.square(df.y - y_mean))

    return cut_df(df, index)


    
def cut_df(df, index):
    '''
    Removes all datapoints too close to a boarder, the z-axis or with unphysical volume fraction (0.35<phi<0.8)

            Parameters:
                    df (pd.DataFrame): DataFrame containing preprocessed minkowski data.
                    index (str) XXX_YYY Determining the measurement and indicating the resolution
                    
            Returns:
                    DataFrame
    '''
    res = approximate_diameter(index)
    #r_min, r_max, z_min, z_max
    bounds = {20:[30, 200, 100, 400], 
              30:[45, 300, 135, 600]}
    r_min, r_max, z_min, z_max = bounds[res]

    mask1 = df.r < r_max
    mask2 = df.r > r_min

    mask3 = df.z < z_max
    mask4 = df.z > z_min
    
    mask5 = df.lpf < 0.8
    mask6 = df.lpf > 0.35

    mask = np.all([mask1, mask2, mask3, mask4, mask5, mask6], axis = 0)
    return df[mask]




features = ['w000', 'w100', 'w200', 'w300', 'w020_eigsysEv1', 'w020_eigsysEv2', 'w020_eigsysEv3', 'w120_eigsysEv1', 'w120_eigsysEv2', 'w120_eigsysEv3', 'w220_eigsysEv1', 'w220_eigsysEv2', 'w220_eigsysEv3', 'w320_eigsysEv1', 'w320_eigsysEv2', 'w320_eigsysEv3', 'w102_eigsysEv1', 'w102_eigsysEv2', 'w102_eigsysEv3', 'w202_eigsysEv1', 'w202_eigsysEv2', 'w202_eigsysEv3', 'q2', 'w2', 'q3', 'w3', 'q4', 'w4', 'q5', 'w5', 'q6', 'w6', 'q7', 'w7', 'q8', 'w8', 'com_0', 'com_1', 'com_2', 'com_3', 'cc_0_x', 'cc_0_y', 'cc_0_z', 'cc_1_x', 'cc_1_y', 'cc_1_z', 'cc_2_x', 'cc_2_y', 'cc_2_z', 'cc_3_x', 'cc_3_y', 'cc_3_z', 'nn', 'lpf', 'cov_ani_0', 'cov_ani_1', 'cov_ani_2', 'cov_ani_3', 'inv_ani_1', 'inv_ani_2']

feature_lists = {'min_fun':['w000', 'w100', 'w200'], 
                 'min_ev':['w020_eigsysEv1', 'w020_eigsysEv2', 'w020_eigsysEv3', 'w120_eigsysEv1', 'w120_eigsysEv2', 'w120_eigsysEv3', 'w220_eigsysEv1', 'w220_eigsysEv2', 'w220_eigsysEv3', 'w320_eigsysEv1','w320_eigsysEv2', 'w320_eigsysEv3', 'w102_eigsysEv1', 'w102_eigsysEv2', 'w102_eigsysEv3', 'w202_eigsysEv1', 'w202_eigsysEv2', 'w202_eigsysEv3'],
                'curv_cen':['cc_0_x', 'cc_0_y', 'cc_0_z', 'cc_1_x', 'cc_1_y', 'cc_1_z', 'cc_2_x', 'cc_2_y', 'cc_2_z', 'cc_3_x', 'cc_3_y', 'cc_3_z'],
                 'imt':['q2', 'w2', 'q3', 'w3', 'q4', 'w4', 'q5', 'w5', 'q6', 'w6', 'q7', 'w7', 'q8', 'w8'],
                 'com': ['com_0', 'com_1', 'com_2', 'com_3'],
                 'custom':['nn', 'lpf'],
                 'aniso':['cov_ani_0', 'cov_ani_1', 'cov_ani_2', 'cov_ani_3', 'inv_ani_1', 'inv_ani_2']}


def merge_measurements(directory):
    '''
    Concatenates all measurement CSVs within a directory and reindexes them such that each datapoint is linked to its measurement.
            Parameters:
                    directoy (str)
                    
            Returns:
                    DataFrame (all merges measurements)
    '''
    #list of all file paths in the directory
    file_paths = [directory + file_name for file_name in os.listdir(directory)]
    #all temporary dataframes are appended to a list that is concatenated later
    measurement_list = []
    for file_index in trange(len(file_paths)):
        file_path = file_paths[file_index]
        meas_index = file_path[-11:-4:] + '_'
        df_temp = pd.read_csv(file_path, index_col = 0)
        #reindexing
        df_temp.index = [meas_index + str(ind) for ind in df_temp.index]
        #appending
        measurement_list.append(df_temp)
    #merge all DFs in measurement_list
    return pd.concat(measurement_list)

def read_mf(df, path):
    '''
    Reads in the Minkowski Functionals from the Karambola Output
            Parameters:
                    df (pd.DataFrame): The dataframe where the data is collected.
                    path (str): path to the Directory with Karambola Output
            Returns:
                    DataFrame
    '''
    data = pd.read_table(path + 'w000_w100_w200_w300', delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                 names = ['value', 'type', 'closed', 'reference'])

    for tp in ['w000', 'w100', 'w200', 'w300']:
        #tp stands for type
        df[tp] = data.loc[data.type == tp, 'value']
    return df


def read_mv(df, path):
    '''
    Reads in the Minkowski Vectors from the Karambola Output
            Parameters:
                    df (pd.DataFrame): The dataframe where the data is collected.
                    path (str): path to the Directory with Karambola Output
            Returns:
                    DataFrame
    '''
    data = pd.read_table(path + 'w010_w110_w210_w310', delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                 names = ['value_x', 'value_y', 'value_z', 'type', 'closed', 'reference'])

    for tp in ['w010', 'w110', 'w210', 'w310']:
        df[[tp+'_x', tp+'_y', tp+'_z']] = data.loc[data.type == tp, ['value_x', 'value_y', 'value_z']]
    return df



def read_mt(df, path):
    '''
    Reads in the rank 2 Minkowski Tensors from the Karambola Output
            Parameters:
                    df (pd.DataFrame): The dataframe where the data is collected.
                    path (str): path to the Directory with Karambola Output
    '''
    for tensor in ['w020_eigsys', 'w120_eigsys', 'w220_eigsys', 'w320_eigsys', 'w102_eigsys', 'w202_eigsys']:
        data = pd.read_table(path + tensor, delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                     names = [tensor+'Ev1', 'v1_x', 'v1_y', 'v1_z', 
                              tensor+'Ev2', 'v2_x', 'v2_y', 'v2_z',
                              tensor+'Ev3', 'v3_x', 'v3_y', 'v3_z', 'type', 'closed', 'reference'])
        df[[tensor+'Ev1', tensor+'Ev2', tensor+'Ev3']] = data[[tensor+'Ev1', tensor+'Ev2', tensor+'Ev3']]
    return df


def read_qw(df, path):
    '''
    Reads in the IMTs from the Karambola Output
            Parameters:
                    df (pd.DataFrame): The dataframe where the data is collected.
                    path (str): path to the Directory with Karambola Output
            Returns:
                    DataFrame
    '''
    qw = ['q0', 'w0', 'q1', 'w1', 'q2', 'w2', 'q3', 'w3', 'q4', 'w4', 'q5', 'w5', 'q6', 'w6', 'q7', 'w7', 'q8', 'w8']
    data = pd.read_table(path + 'msm_ql', delimiter = '\s+', comment = '#', engine = 'python', index_col = 0,
                 names = qw + ['type', 'closed', 'reference'])
    df[qw] = data[qw]
    return df


def read_xyz(df, xyz_path):
    '''
    Reads in the coordinates from the Karambola Output
            Parameters:
                    df (pd.DataFrame): The dataframe where the data is collected.
                    path (str): path to the Directory with Karambola Output
            Returns:
                    DataFrame
    '''
    xyz = pd.read_table(xyz_path, delimiter = r'\s+', names = ['P', 'x', 'y', 'z'], header=1)
    
    #index starts at 1 but is read in starting at 0. Thus, the correction
    xyz.index += 1
    df[['x', 'y', 'z']] = xyz.loc[df.index, ['x', 'y', 'z']]
    return df

def anisotropy_and_centroid(df):
    '''
    Computes the generalized anisotropies and curvature centroids
            Parameters:
                    df (pd.DataFrame): The dataframe where the data is collected.
            Returns:
                    DataFrame
    '''
    for j in ['0', '1', '2', '3']:
        #curvature centroids
        for i in ['x', 'y', 'z']:
            numerator = 'w'+j+'10_'+i
            denominator = 'w'+j+'00'
            df['cc_'+j+'_'+i] = df[numerator]/df[denominator]# - df[i]
        
        #center of mass
        df['com_'+j] = np.sqrt(np.sum(np.square(df[['cc_'+j+'_x', 'cc_'+j+'_y', 'cc_'+j+'_z']]), axis = 1))
        #local anisotropy (covariant)
        df['cov_ani_'+j] = df['w'+j+'20_eigsysEv1']/df['w'+j+'20_eigsysEv3']
        #local anisotropy (invariant)
        if j in ['1', '2']:
            df['inv_ani_'+j] = df['w'+j+'02_eigsysEv1']/df['w'+j+'02_eigsysEv3']

    return df



def process_measurement(path, xyz_path):
    '''
    Reads all relevant data from a directory containing Karambola Output and computes the generalized ansiotropies and curvature centroids.
            Parameters:
                    path (str): path to folder with Karambola output
                    xyz_path (str): path to coordinate file
            Returns:
                    DataFrame
    '''
    df = pd.DataFrame()
    df = read_mf(df, path)
    df = read_mv(df, path)
    df = read_mt(df, path)
    df = read_qw(df, path)
    df = read_xyz(df, xyz_path)
    df = anisotropy_and_centroid(df)

    return df

