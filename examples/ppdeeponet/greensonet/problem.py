import os
import paddle
import numpy as np
from scipy.interpolate import griddata
import meshio

class Stokes(object):

    def __init__(self, data_folder, velocity_component_name):
        self.data_folder = data_folder
        self.velocity_component_name = velocity_component_name

    def u_exact(self, x, case_index):
        u_exact_file = os.path.join(self.data_folder,
            f'res{case_index}.vtu')
        mesh = meshio.read(u_exact_file)
        points = mesh.points
        if self.velocity_component_name == 'x':
            var_name = "u"
        elif self.velocity_component_name == 'y':
            var_name = "v"
        elif self.velocity_component_name == 'z':
            var_name = "w"
        else:
            raise ValueError('Invalid velocity component name.')
        u_exact = mesh.point_data.get(var_name, None)
        interpolated_values = griddata(points=points, values=u_exact, xi=x,
            method='nearest')
        return interpolated_values[..., np.newaxis]

    def f(self, x, case_index):
        """right-hand term
        Params:
        -------
        x: ndarrays of float with shape (n, 3)
        """
        x = x.detach().cpu().numpy()
        f_file = os.path.join(self.data_folder, f'res{case_index}.vtu')
        mesh = meshio.read(f_file)
        points = mesh.points
        fx = mesh.point_data.get('fx', None)
        interpolated_values = griddata(points=points, values=fx, xi=x, method='nearest')
        res = paddle.to_tensor(data=interpolated_values[..., np.newaxis]).astype(dtype='float32')
        return res

    def a(self, x):
        x = x.detach().cpu().numpy()
        res = np.ones_like(x[:, [0]]) / 100
        res = paddle.to_tensor(data=res).astype(dtype='float32')
        return res

    def g(self, x, boundary_type):
        """ Dirichlet boundary condition
        Params:
        -------
        x: ndarrays of float with shape (n, 3)
        boundary_type: ndarrays of int with shape (n, 1)
        """
        x = x.detach().cpu().numpy()
        boundary_type = boundary_type.detach().cpu().numpy()
        if tuple(x.shape)[0] != tuple(boundary_type.shape)[0]:
            raise ValueError(
                f'The number of points in x : {tuple(x.shape)[0]} and boundary_type : {tuple(boundary_type.shape)[0]} must be the same.'
                )
        result = np.zeros_like(x[:, [0]])
        mask = boundary_type == 3
        result[mask.flatten()] = 1
        result = paddle.to_tensor(data=result).astype(dtype='float32')
        return result
