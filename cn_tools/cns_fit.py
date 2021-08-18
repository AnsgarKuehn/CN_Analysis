import pandas as pd
import numpy as np
import scipy.special as sp
from scipy.spatial import KDTree
from lmfit.models import GaussianModel, StepModel, Model
from collections import Counter
import re

#def compute_nearest_neighbours()
def compute_nearest_neighbours(df, poly_file):
    '''
    Computes the nearest neighbours from a poly file, writes them into a csv and saves it.

            Parameters:
                    file_name (str): Full path to the csv file the data shall be saved in.
                    poly_file (str): Full path to the poly file containing the info about the nearest
                            neighbours.

            Returns:
                    Nothing is returned. The data is written and saved in the csv under file_name.
    '''
    
    #compute nearest neighbours from poly file
    pattern = re.compile(r'c\(\d+, \d+, \d+, \d+\)')
    ind_list = []
    for line in open(poly_file):
        if match := re.search(pattern, line):
            ind_list += match.group()[2:-1].split(', ')
    nearest_neighbours = Counter(ind_list)
    #write nearest neighbours into data frame and save
    for i in df.index:
        df.at[i, 'nn'] = nearest_neighbours[str(i)]
        
    return df
    
def compute_local_contacts(xyz, xyz_cut, Z, interval, cns, namespace = ''):
    '''
    Computes the local contacts of a configuration given the global contact number Z.

            Parameters:
                    file_name (str): Full path to the csv file the data shall be saved in.
                    Z (float): The .

            Returns:
                    Nothing is returned. The data is written and saved in the csv under file_name.
    '''
    #compute cns data with finer resolution

    index = get_index(cns, Z)
    diamter = interval[index]
    
    #compute local contacts
    tree = KDTree(xyz)
    
    return tree.query_ball_point(xyz_cut, diamter, return_length = True) -1 
    
    
def Z_from_cns(interval, cns, mu, sigma, boundary = None):
    '''
    Computes the global contact number of a configuration by fitting the cns function using lmfit.

            Parameters:
                    interval (np.array): Array of distances serving as the x axis for the cns fit.
                    cns (np.array): Array of the contact number serving as a function of the distance.
                    mu (float): The mean distance of all particles in a configuration
                    sigma (float): The width of the errorfunction calculated as a mean over all configurations
                        with the same resolution.
                    boundary (float or int) (optional): When provided, boundary serves as the upper limit
                        for the range in which the cns fit is conducted.

            Returns:
                    Z (float) The global contact number.
    '''
    #define lmfit model for the linear offset
    off_model = Model(off)
    off_model.set_param_hint('a', min = 0)
    off_model.set_param_hint('mu', value = mu, vary = False)
    
    #define lmfit model for the errorfunction
    step_model = StepModel(form='erf')
    step_model.set_param_hint('center', value = mu, vary = False)
    step_model.set_param_hint('sigma', value = sigma, vary = False)
    
    #define composite model
    cns_function = off_model + step_model
    init = cns_function.make_params(a = 1, amplitude = 5, sigma = 0.1)
    
    #perform fit within boundary if provided
    if isinstance(boundary, (float, int)):
        boundary_index = get_index(interval, boundary)
        out = cns_function.fit(data = cns[:boundary_index], params = init, x = interval[:boundary_index])
    else:
        out = cns_function.fit(data = cns, params = init, x = interval)
    return out.best_values['amplitude']


def sigma_from_cns(interval, cns, mu):
    '''
    Computes the sigma of a configuration by fitting the cns function using lmfit.

            Parameters:
                    interval (np.array): Array of distances serving as the x axis for the cns fit.
                    cns (np.array): Array of the contact number serving as a function of the distance.
                    mu (float): The mean distance of all particles in a configuration
                    
            Returns:
                    sigma (float) width of the errorfunction being part of the cns function.
    '''
    #define lmfit model for the linear offset
    off_model = Model(off)
    off_model.set_param_hint('a', min = 0)
    off_model.set_param_hint('mu', value = mu, vary = False)
    
    #define lmfit model for the errorfunction
    step_model = StepModel(form='erf')
    step_model.set_param_hint('center', value = mu, vary = False)
    
    #define composite model
    cns_function = off_model + step_model
    
    #perform fit
    init = cns_function.make_params(a = 1, amplitude = 5, sigma = 0.1)
    out = cns_function.fit(data = cns, params = init, x = interval)
    
    return out.best_values['sigma']

def mu_from_gauss(interval, pc):
    '''
    Computes the mu of a configuration by fitting the cns function using lmfit.

            Parameters:
                    interval (np.array): Array of distances serving as the x axis for the cns fit.
                    pc (np.array): Array of the pair correlation as a function of the distance.
                    
            Returns:
                    mu (float) center of the gaussian which has been fit to the pair correlation.
    '''
    #define lmfit model
    model = GaussianModel()
    
    #perform fit
    init = model.guess(data = pc, x = interval)
    out = model.fit(data = pc, params = init, x = interval)
    
    return out.best_values['center']

def cns_pc_from_file(file_name, dr=0.005, lower_range = 1, upper_range = 2):
    '''
    Computes the contact number scaling (cns) and the pair correlation of a configuration
    using the binary tree from scipy.spatial given the path to a configuration csv.

            Parameters:
                    file_name (str): Full path to the csv containing the data of a configuration.
                    dr (float): The resolution of the distance array on which cns and pc a calculated.
                    lower_range (float): Determines the lower bound of the distance array in units of voxel.
                            Computed as: approximate_diameter - lower range
                    upper_range (float): Determines the upper bound of the distance array in units of voxel.
                            Computed as: approximate_diameter + lower range
                    
            Returns:
                    touple of np.arrays (interval, cns, pc) 
    '''
    #read in relevant data
    df = pd.read_csv(file_name, index_col = 0)
    rho = df.shape[0]/df.w000.sum()
    xyz = df[['x','y','z']].to_numpy()
    
    #compute approximate diameter (depends on the resolution of the measurement)
    d_appr = approximate_diameter(file_name)
    
    #compute data
    interval = np.arange(d_appr-lower_range, d_appr+upper_range, dr)
    
    tree = KDTree(xyz)
    neighbors = tree.count_neighbors(tree, interval)
    cns = neighbors/xyz.shape[0]-1
    pc = np.diff(neighbors, prepend=neighbors[0])/(4*np.pi*interval**2*dr*rho)
    
    return interval, cns, pc

def cns_cut(xyz, xyz_cut, mu, dr = 0.005, lower_range = 1, upper_range = 2):
    '''
    Computes the contact number scaling (cns) and the pair correlation of a configuration
        using the binary tree from scipy.spatial given a distance array (interval).

            Parameters:
                    interval (np.array):  Array of distances serving as the x axis for the cns fit.
                    dr (float): The resolution of the distance array on which cns and pc a calculated.
                    xyz (N*3 np.array): Array containing all particle coordinates
            Returns:
                    touple of np.arrays (cns, pc) 
    '''
    interval = np.arange(mu-lower_range, mu+upper_range, dr)
    
    tree = KDTree(xyz)
    tree_cut = KDTree(xyz_cut)
    
    cns_cut = tree.count_neighbors(tree_cut, interval)/xyz_cut.shape[0]-1
    
    return interval, cns_cut



def off(x,a,mu):
    '''
    Linear offset of the for \theta(x-mu)*a(x-mu)

            Parameters:
                    x (np.array)
                    a (float): Slope of the linear funciton.
                    mu (float): Root of linear function and mean particle distance
            Returns:
                    y (np.array)
    '''
    y = a*(x-mu)
    y[np.where(x < mu)] = 0
    return y


def get_index(array, value):
    '''
    Returns the index of an array with the entry being closest to value.
    
            Parameters:
                    array (np.array)
                    value (float)
            Returns:
                    int
    '''
    return np.argmin(np.abs(array-value))


def approximate_diameter(path):
    '''
    Returns the initial value of diameter in units of voxel (either 20 or 30) is dependent of measurement resolution.
    This functions return the correct initial given the path to a directory conatining measurement files.
            Parameters:
                    path (str):  path to a directory conatining measurement files
            Returns:
                    approximate diameter (20 or 30) (int)
    '''
    resolution_dict = {'005':20, '006':20, '007':20, '008':30, '012':30}
    return resolution_dict[path.split('_')[-2]]


def compute_mean_sigma(sigma_file_path = '../Data/preprocessed/sigma.csv', resolution = 20):
    '''
    Computes the average of all sigmas with the same resolution.
            Parameters:
                    sigma_file_path (str):  path to file containing all measured sigmas
            Returns:
                    approximate sigma (float)
    '''
    df_sigma = pd.read_csv(sigma_file_path, index_col = 0)
    if resolution == 20:
        index = [i for i in df_sigma.index if i[:3] in ['005', '006', '007']]
    elif resolution == 30:
        index = [i for i in df_sigma.index if i[:3] in ['012', '008']]
    else:
        print('resolution has to be 20 or 30')

    sigma_mean = df_sigma.loc[index, 'sigma'].mean()
    return sigma_mean
