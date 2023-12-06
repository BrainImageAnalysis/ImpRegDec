import torch

import torch.nn.functional as F
import numpy as np

from collections import defaultdict

def make_coords_tensor(dims=(256, 256, 64), is_vector=True):

    '''
    modification, from https://proceedings.mlr.press/v172/wolterink22a.html
    '''        
    
    n_dims = len(dims)
    
    coords = [torch.linspace(-1, 1, dims[i]) for i in range(n_dims)]
    coords = torch.meshgrid(*coords)
    coords = torch.stack(coords, dim=n_dims)
    
    if is_vector==True:
    
        coords = coords.view([np.prod(dims), n_dims])
        
    return coords


def fast_bilinear_color_interpolation(input_array, x_indices, y_indices):
    
    '''
    modification of the method from https://proceedings.mlr.press/v172/wolterink22a.html
    '''        
    
    n_dim = input_array.shape[0]
    
    x_indices = (x_indices + 1) * (input_array.shape[1] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    
    output = torch.zeros(input_array.shape[1]*input_array.shape[2], n_dim).to(input_array.device)

    for i in range(n_dim):
        
        temp = (
              input_array[i, x0, y0] * (1 - x) * (1 - y) 
            + input_array[i, x1, y0] * x * (1 - y) 
            + input_array[i, x0, y1] * (1 - x) * y
            + input_array[i, x1, y1] * x * y
             )
        
        output[:, i] = temp
    
    return output

def fast_bilinear_interpolation(input_array, x_indices, y_indices):
    
    '''
    modification of the method from https://proceedings.mlr.press/v172/wolterink22a.html
    '''        
    
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    x = x_indices - x0
    y = y_indices - y0

    output = (
              input_array[x0, y0] * (1 - x) * (1 - y) 
            + input_array[x1, y0] * x * (1 - y) 
            + input_array[x0, y1] * (1 - x) * y
            + input_array[x1, y1] * x * y
             ) 
    
    return output

class MetricMonitor:
    
    '''
    
    
    '''        
    
    def __init__(self, float_precision=4):
        
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        
        self.metrics = defaultdict(lambda: {'value': 0, 'count': 0, 'average': 0})

    def update(self, metric_name, value):
        
        metric = self.metrics[metric_name]

        metric['value'] += value
        metric['count'] += 1
        metric['average'] = metric['value'] / metric['count']

    def __str__(self):
        
        return ' | '.join(
            ['{metric_name}: {temp:.{float_precision}f}'.format(metric_name=metric_name, temp=metric['average'], float_precision=self.float_precision) for (metric_name, metric) in self.metrics.items()]
        )    


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    