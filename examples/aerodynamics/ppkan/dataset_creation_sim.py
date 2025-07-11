import numpy as np
import time
from tqdm import tqdm
import os
import json

from typing import Any

import paddle
from paddle.io import Dataset, DataLoader

import airfrans as af

from geomdl import fitting
from geomdl import BSpline

def preprocess_data_airfrans(root_dir="./Dataset/", task="scarce", train=True, download_data=False, sample_points=10000, batch_size=1024, **kwargs):
    """
    Preprocess the data from the AirFrans dataset.
    Args:
    root_dir (str): The path to the root directory of the dataset. Default is "./Dataset".
    task (str): The name of the task to be performed on the dataset. Default is "scarce".
    train (bool): Whether to load the training or testing data. Default is True.
    download_data (bool): Whether to download the dataset. Default is False.
    sample_points (int): Number of points to sample from each simulation. Default is 10000.

    Returns:
    tuple: A tuple containing two lists - X_list and y_list.
    X_list (list): [simulation_num * sample_points, Foil_coord(60) + inlet_velocity(2) + sdf (1) + data_point_coord(2) ]
    y_list (list): [simulation_num * sample_points, pressure(1) + velocity(2) + turbulence_viscosity(1)]
    """
    X_list = []
    Y_list = []
    with open(root_dir + 'manifest.json', 'r') as f:
        manifest = json.load(f)
    
    if train:
        manifest_train = manifest[task + '_train']
        sim_num = len(manifest_train)
    else:
        manifest_train = manifest['full' + '_test']
        sim_num = len(manifest_train)#int(0.1 * len(manifest_train))
    nacas = []
    for sim_name in manifest_train[:sim_num]:
        simulation_name_splitted = sim_name.split("_")
        invel = float(simulation_name_splitted[2])
        aoa = float(simulation_name_splitted[3]) * np.pi / 180
        invel = np.array([invel * np.cos(aoa), invel * np.sin(aoa)])

        params = [float(j) for j in simulation_name_splitted[4:]]
        foil_pts = af.naca_generator.naca_generator(params, 100)
        naca = fit_airfoil_nurbs(foil_pts, 30, 3)
        naca = np.array(naca)
        naca = naca[:,:].flatten()
        naca = np.concatenate((naca, invel))
        nacas.append(naca)

    nacas = np.asarray(nacas)
    invel_mean = np.mean(nacas[:,60:], axis=0)
    invel_std = np.std(nacas[:,60:], axis=0)
    nacas[:,60:] = (nacas[:,60:] - invel_mean) / (invel_std + 1e-8)

    for index, sim_name in enumerate(manifest_train[:sim_num]): 
        simulation = af.Simulation(root=root_dir, name=sim_name, T=298.15)
        seed = np.random.randint(0, 10000)
        
        sample = simulation.sampling_volume(seed, sample_points, density='mesh_density')
        X = np.hstack((np.tile(nacas[index], (len(sample), 1)), sample[:, 2:3], sample[:, 0:2]))
        X = X.astype(np.float32)
        Y = sample[:, 3:].astype(np.float32)
        
        X_list.append(X)
        Y_list.append(Y)

    X_list = np.concatenate(X_list, axis=0)
    print(X_list.shape)
    Y_list = np.concatenate(Y_list, axis=0)
    Y_list, Y_mean, Y_std = z_score_normalizer(Y_list)
    print(np.mean(Y_list, axis=0), np.std(Y_list, axis=0))

    data_set = FoilDataset(X_list, Y_list)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    import pickle
    data_dict = {
        'X_list': X_list,
        'invelsdf_mean': invel_mean,
        'invelsdf_std': invel_std,
        'y_list': Y_list,
        'y_mean': Y_mean,
        'y_std': Y_std
      }
    if train:
      filename = 'train_data.pkl'
    else:
      filename = 'test_data.pkl'
    with open(filename, 'wb') as f:
      pickle.dump(data_dict, f)
      print(f'Data saved in {filename}.')

    return data_set, data_loader, invel_mean, invel_std, Y_mean, Y_std, X_list[:sample_points], Y_list[:sample_points]
    

def complete_field_prediction(invel_mean, invel_std, Y_mean, Y_std,
                               model,
                               root_dir="./Dataset/", task="scarce", train=True, download_data=False, sample_points=10000, batch_size=512, 
                               ):
  with open(root_dir + 'manifest.json', 'r') as f:
      manifest = json.load(f)
  

  manifest_train = manifest['full' + '_test']
  sim_num = len(manifest_train)

  sim_index = np.random.randint(0, sim_num)
  sim_name = manifest_train[sim_index]
  simulation_name_splitted = sim_name.split("_")
  invel = float(simulation_name_splitted[2])
  aoa = float(simulation_name_splitted[3]) * np.pi / 180
  invel = np.array([invel * np.cos(aoa), invel * np.sin(aoa)])
  print(invel)
  invel = (invel - invel_mean) / (invel_std + 1e-8)
  print(invel)
  params = [float(j) for j in simulation_name_splitted[4:]]
  foil_pts = af.naca_generator.naca_generator(params, 100)
  naca = fit_airfoil_nurbs(foil_pts, 30, 3)
  naca = np.array(naca)
  naca = naca[:,:].flatten()
  naca = np.concatenate((naca, invel))


  simulation = af.Simulation(root=root_dir, name=sim_name, T=298.15)
  sdf = simulation.sdf
  coord = simulation.position
  Y = np.hstack((simulation.velocity, simulation.pressure, simulation.nu_t))
  Y = (Y - Y_mean) / (Y_std + 1e-8)
  
  X = np.hstack((np.tile(naca, (len(sdf), 1)), sdf[:, 0:1], coord[:, 0:2]))
  X = X.astype(np.float32)
  Y = Y.astype(np.float32)
  
  print(X.shape)
  print(Y.shape)

  data_set = FoilDataset(X, Y)
  data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=0)

  predictions = []
  for x_batch, _ in data_loader:
    pred = model({
      'branch1': x_batch[:, :63],
      'trunk': x_batch[:, 63:],
    })
    predictions.append(pred.numpy())
    del x_batch, pred
  predictions = np.concatenate(predictions, axis=0)
  predictions = (predictions * Y_std) + Y_mean
  print(predictions.shape)

  pred_sim = simulation
  pred_sim.pressure = predictions[:, 0:1]
  pred_sim.velocity = predictions[:, 1:3]
  pred_sim.nu_t = predictions[:, 3:4]
  pred_coefficients = pred_sim.force_coefficient(compressible=False, reference=False)
  ref_coefficients = simulation.force_coefficient(compressible=False, reference=True)
  print(f"Predicted coefficients: {pred_coefficients}, Reference coefficients: {ref_coefficients}")

  pred_sim.save(root=f"/shared/KAN_paddle/AFDataset/outputs/predictions/{sim_name}")







def fit_airfoil_nurbs(points_arrray: np.ndarray, num_ctrlpts: int = 30, degree: int = 3):
  """
  Fit AirFoil Curve using NURBS curve fitting algorithm
  Args:
    points_arrray (np.ndarray): Array of airfoil surface points to fit
    num_ctrlpts (int): number of control points to return
    degree (int): degree of the B-Spline curve
  Returns:
    fitted_control_points (np.ndarray): Array of fitted control points
  """
  points_list = points_arrray.tolist()
  curve = fitting.approximate_curve(points_list, degree, ctrlpts_size=num_ctrlpts)

  return curve.ctrlpts

def z_score_normalizer(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  norm_data = (data - mean) / (std + 1e-8)
  return norm_data, mean, std

class FoilDataset(Dataset):
  def __init__(self, x_data, y_data):
    super(FoilDataset, self).__init__()
    self.x_data = x_data
    self.y_data = y_data
  
  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return len(self.x_data)


def main():

  train_set, train_loader, invel_mean, invel_std, y_mean, y_std, _, _ = preprocess_data_airfrans(root_dir="./Dataset/", task="scarce", train=False, download_data=False, sample_points=10000, batch_size=1024)
  print(invel_mean, invel_std, y_mean, y_std)
  
if __name__ == '__main__':
  main()