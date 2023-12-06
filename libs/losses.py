import math

import torch
import torch.nn.functional as F

import numpy as np


class MSELoss(torch.nn.Module):    
    
    '''
    mean squared error loss function
    '''
    
    def __init__(self, 
                 is_tensor=False):
        
        super(MSELoss, self).__init__()
        
        self.is_tensor = is_tensor
        
    def forward(self, y_true, y_pred):

        if self.is_tensor==True:

            return (y_true - y_pred)**2

        else:

            return torch.mean((y_true - y_pred)**2)        
            
class LNCCLoss(torch.nn.Module):
    
    '''
    local normalized cross-correlation loss function
    '''

    def __init__(self, 
                 win=(9, 9, 9),
                 n_channels=1,
                 is_tensor=False):
        
        super(LNCCLoss, self).__init__()
        
        self.is_tensor = is_tensor
        self.win = win
        self.n_channels = n_channels
        self.win_size = np.prod(win)*self.n_channels
        self.ndims = len(win)
        
        self.conv = getattr(torch.nn, 'Conv%dd' % self.ndims)(n_channels, 1, self.win, stride=1, padding='same', padding_mode='replicate', bias=False)
        
        with torch.no_grad():
            
            torch.nn.init.ones_(self.conv.weight)
                    
        for param in self.conv.parameters():
            
            param.requires_grad = False
            
    def forward(self, y_true, y_pred):
        
        true_sum = self.conv(y_true) / self.win_size
        pred_sum = self.conv(y_pred) / self.win_size    
                
        true_cent = y_true - true_sum
        pred_cent = y_pred - pred_sum
        
        nominator = self.conv(true_cent * pred_cent)
        nominator = nominator * nominator
                
        denominator = self.conv(true_cent * true_cent) * self.conv(pred_cent * pred_cent)
        
        cc = (nominator + 1e-6) / (denominator + 1e-6)  
        cc = torch.clamp(cc, 0, 1)  
        
        if self.is_tensor==True:
        
            return -cc
            
        else:

            return -torch.mean(cc)    
        
class NCCLoss(torch.nn.Module):
    
    '''
    normalized cross-correlation loss function
    '''    
    
    def __init__(self, is_tensor=False):
        
        super(NCCLoss, self).__init__()

        self.is_tensor = is_tensor
        
    def forward(self, y_true, y_pred):
        
        nominator = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
        
        denominator = y_true.std() * y_pred.std()
        
        cc = (nominator + 1e-6) / (denominator + 1e-6)  
        cc = torch.clamp(cc, 0, 1)  
        
        if self.is_tensor==True:
        
            return -cc
            
        else:

            return -torch.mean(cc)    
        

class JacobianLossCoords(torch.nn.Module):
    
    '''
    
    '''    
    
    def __init__(self, 
                 add_identity=True,
                 is_tensor=False):
        
        super(JacobianLossCoords, self).__init__()
            
        self.add_identity = add_identity
        self.is_tensor = is_tensor
            
    def forward(self, input_coords, output):
        
        jac = self.compute_jacobian_matrix(input_coords, output, add_identity=self.add_identity)
        
        loss = 1 - torch.det(jac)
        
        if self.is_tensor==True:
        
            return torch.abs(loss)
            
        else:

            return torch.mean(torch.abs(loss))
        

    def compute_jacobian_matrix(self, input_coords, output, add_identity=True):

        dim = input_coords.shape[1]

        jacobian_matrix = torch.zeros(input_coords.shape[0], dim, dim)

        for i in range(dim):

            jacobian_matrix[:, i, :] = self.gradient(input_coords, output[:, i])

            if add_identity:

                jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])

        return jacobian_matrix        
    
        
    def gradient(self, input_coords, output, grad_outputs=None):

        grad_outputs = torch.ones_like(output)

        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        return grad        
            
        
class GradLossCoords(torch.nn.Module):
    
    '''
    
    '''

    def __init__(self, 
                 penalty='l2'):
        
        super(GradLossCoords, self).__init__()
        
        self.penalty = penalty #for later

    def forward(self, input_coords, output, grad_outputs=None):

        grad_outputs = torch.ones_like(output)

        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        return torch.mean(grad*grad)
        
        
class DiceLoss(torch.nn.Module):
    
    """
    Dice loss function with optional weighting, [batch, n_objects, *dims] 
    """

    def __init__(self, 
                 weighting=False):
        
        super(DiceLoss, self).__init__()
        
        self.weighting = weighting
    
    def forward(self, y_true, y_pred):
        
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        
        dice_axes = top / bottom
        
        if self.weighting==True:
            
            w_axes = [0] + list(range(2, ndims+2))
            weights = torch.mean(y_true, w_axes)
            weights = 1.0 / weights
            weights = weights / torch.sum(weights)
        
            dice = torch.mean(weights*dice_axes)
            
        else:
            
            dice = torch.mean(dice_axes)
        
        return 1 - dice        
        
        
class DiceScore(torch.nn.Module):
    
    """
    Dice loss function with optional weighting, [batch, n_objects, *dims] 
    """

    def __init__(self, 
                 weighting=False):
        
        super(DiceScore, self).__init__()
        
        self.weighting = weighting
    
    def forward(self, y_true, y_pred):
        
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        
        dice_axes = top / bottom
        
        if self.weighting==True:
            
            w_axes = [0] + list(range(2, ndims+2))
            weights = torch.mean(y_true, w_axes)
            weights = 1.0 / weights
            weights = weights / torch.sum(weights)    
            
            dice_axes = weights*dice_axes
        
        return dice_axes      
        
def DiceScore_np(y_true, y_pred):
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (np.sum(np.clip(y_true_f + y_pred_f, a_min=1e-5, a_max=2)))            
        
        
class JacobianDet(torch.nn.Module):
    
    '''
    
    '''    
    
    def __init__(self, 
                 add_identity=True,
                 is_tensor=False):
        
        super(JacobianDet, self).__init__()
            
        self.add_identity = add_identity
        self.is_tensor = is_tensor
            
    def forward(self, input_coords, output):
        
        jac = self.compute_jacobian_matrix(input_coords, output, add_identity=self.add_identity)
        
        jac = torch.det(jac)
        
        return jac
        

    def compute_jacobian_matrix(self, input_coords, output, add_identity=True):

        dim = input_coords.shape[1]

        jacobian_matrix = torch.zeros(input_coords.shape[0], dim, dim)

        for i in range(dim):

            jacobian_matrix[:, i, :] = self.gradient(input_coords, output[:, i])

            if add_identity:

                jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])

        return jacobian_matrix        
    
        
    def gradient(self, input_coords, output, grad_outputs=None):

        grad_outputs = torch.ones_like(output)

        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        return grad           
    
class ExclusionLossCoords(torch.nn.Module):
    
    '''
    
    '''    
    
    def __init__(self, 
                 add_identity=True,
                 is_tensor=False,
                 alpha_1=1,
                 alpha_2=1):
        
        super().__init__()
            
        self.add_identity = add_identity
        self.is_tensor = is_tensor
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.tanh = torch.nn.Tanh()
        
    def forward(self, input_coords, output_1, output_2):
        
        jac_1 = self.compute_jacobian_matrix(input_coords, output_1, add_identity=self.add_identity)
        jac_2 = self.compute_jacobian_matrix(input_coords, output_2, add_identity=self.add_identity)
        
        loss = self.tanh(self.alpha_1*jac_1) * self.tanh(self.alpha_2*jac_2)
        
        return torch.mean(torch.abs(loss))

    def compute_jacobian_matrix(self, input_coords, output, add_identity=True):

        dim = input_coords.shape[1]

        jacobian_matrix = torch.zeros(input_coords.shape[0], dim, dim).to(input_coords.device)

        for i in range(dim):

            jacobian_matrix[:, i, :] = self.gradient(input_coords, output[:, i])

            if add_identity:

                jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])

        return jacobian_matrix        
    
        
    def gradient(self, input_coords, output, grad_outputs=None):

        grad_outputs = torch.ones_like(output)

        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        return grad         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    