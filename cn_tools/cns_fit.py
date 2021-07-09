import pandas as pd
import numpy as np
import scipy.special as sp
from scipy.spatial import KDTree
from lmfit.models import GaussianModel, StepModel, Model
from collections import Counter
import re

def compute_nearest_neighbours(file_name, poly_file):
    
    df = pd.read_csv(file_name, index_col = 0)
    pattern = re.compile(r'c\(\d+, \d+, \d+, \d+\)')
    ind_list = []
    for line in open(poly_file):
        if match := re.search(pattern, line):
            ind_list += match.group()[2:-1].split(', ')
    nearest_neighbours = Counter(ind_list)
    df = pd.read_csv(file_name, index_col = 0)
    for i in df.index:
        df.at[i, 'nn'] = nearest_neighbours[str(i)]
    df.to_csv(file_name)
    

def compute_local_contacts(file_name, Z):
    
    precise_interval, precise_cns, precise_pc = cns_pc_from_file(file_name, dr = 0.001, lower_range=0, upper_range=0.5)
    precise_index = get_index(precise_cns, Z)
    precise_diamter = precise_interval[precise_index]

    df = pd.read_csv(file_name, index_col = 0)
    xyz = df[['x','y','z']].to_numpy()
    tree = KDTree(xyz)
    df['contact_number'] = tree.query_ball_point(xyz, precise_diamter, return_length=True)-1
    df.to_csv(file_name)
    
    
def Z_from_cns(interval, cns, mu, sigma, boundary = None):

    off_model = Model(off)
    off_model.set_param_hint('a', min = 0)
    off_model.set_param_hint('mu', value = mu, vary = False)

    step_model = StepModel(form='erf')
    step_model.set_param_hint('center', value = mu, vary = False)
    step_model.set_param_hint('sigma', value = sigma, vary = False)

    cns_function = off_model + step_model
    init = cns_function.make_params(a = 1, amplitude = 5, sigma = 0.1)
    
    if isinstance(boundary, (float, int)):
        boundary_index = get_index(interval, boundary)
        out = cns_function.fit(data = cns[:boundary_index], params = init, x = interval[:boundary_index])
    else:
        out = cns_function.fit(data = cns, params = init, x = interval)
    return out.best_values['amplitude']


def sigma_from_cns(interval, cns, mu):
    
    off_model = Model(off)
    off_model.set_param_hint('a', min = 0)
    off_model.set_param_hint('mu', value = mu, vary = False)

    step_model = StepModel(form='erf')
    step_model.set_param_hint('center', value = mu, vary = False)
    
    cns_function = off_model + step_model
    init = cns_function.make_params(a = 1, amplitude = 5, sigma = 0.1)
    out = cns_function.fit(data = cns, params = init, x = interval)
    
    return out.best_values['sigma']

def mu_from_gauss(interval, pc):
    model = GaussianModel()
    init = model.guess(data = pc, x = interval)
    out = model.fit(data = pc, params = init, x = interval)
    return out.best_values['center']

def cns_pc_from_file(file_name, dr=0.005, lower_range = 1, upper_range = 2):
    df = pd.read_csv(file_name, index_col = 0)
    rho = df.shape[0]/df.w000.sum()
    xyz = df[['x','y','z']].to_numpy()
    d_appr = approximate_diameter(file_name)
    interval = np.arange(d_appr-lower_range,d_appr+upper_range,dr)
    cns, pc = cns_pc(interval, dr, rho, xyz)
    return interval, cns, pc


def cns_pc(interval, dr, rho, xyz):
    
    tree = KDTree(xyz)
    dists = tree.count_neighbors(tree, interval)
    cns_av = dists/xyz.shape[0]-1
    pc_av = np.diff(dists, prepend=dists[0])/(4*np.pi*interval**2*dr)
    
    return cns_av, pc_av


def erf(x,Z,sigma,d):
    return Z/2*(sp.erf(sigma*(x-d))+1)


def off(x,a,mu):
    y = a*(x-mu)
    y[np.where(x < mu)] = 0
    return y


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def get_index(array, value):
    return np.argmin(np.abs(array-value))


def approximate_diameter(path):
    '''Initial value of diameter in units of voxel (either 20 or 30) is dependent of measurement resolution.
    This functions return the correct initial given the path to a file'''
    resolution_dict = {'005':20, '006':20, '007':20, '008':30, '012':30}
    return resolution_dict[path.split('_')[-2]]


def compute_mean_sigma(sigma_file_path = '../Data/preprocessed/sigma.csv', resolution = 20):

    df_sigma = pd.read_csv(sigma_file_path, index_col = 0)
    if resolution == 20:
        index = [i for i in df_sigma.index if i[:3] in ['005', '006', '007']]
    elif resolution == 30:
        index = [i for i in df_sigma.index if i[:3] in ['012', '008']]
    else:
        print('resolution has to be 20 or 30')
    s = df_sigma.loc[index, 'sigma']
    d_s = df_sigma.loc[index, 'd_sigma']
    sigma_mean = (s/d_s**2).sum()/(1/d_s**2).sum()
    return sigma_mean
