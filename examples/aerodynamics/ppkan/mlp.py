import paddle
import paddle.nn as nn
import numpy as np
"""
2 branch input dims = batch_size-sdf 2-in_vel 
1 trunk input dims = 2-x,y_coord
"""

# Network Define
class BranchNet(nn.Layer):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)  

class TrunkNet(nn.Layer):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim

        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Layer):
    def __init__(self, branch1_in, branch1_hidden, trunk_in, trunk_hidden, **kargs):
        super().__init__()
        self.branch_net1 = BranchNet(input_dim=branch1_in, hidden_dims=branch1_hidden) # invel x and y
        #self.branch_net2 = BranchNet(input_dim=branch2_in, hidden_dims=branch2_hidden) # foil coordinates
        #self.branch_net3 = BranchNet(input_dim=branch3_in, hidden_dims=branch3_hidden)
        self.trunk_net = TrunkNet(input_dim=trunk_in, hidden_dims=trunk_hidden)
        self.hidden_out = trunk_hidden[-1]
        
    def forward(self, inputs):
        b1_out = self.branch_net1(inputs['branch1'])
        #b2_out = self.branch_net2(inputs['branch2'])
       
        t_out = self.trunk_net(inputs['trunk'])    # [Ni, 64]
        # y = paddle.mean(b1_out * t_out, axis=1, keepdim=True)
        # return y
        step = int(self.hidden_out/4)
        y1 = paddle.sum(b1_out[:,0:step] * t_out[:,0:step], axis=1, keepdim=True) # [Ni, 1])
        y2 = paddle.sum(b1_out[:,step:2*step] * t_out[:,step:2*step], axis=1, keepdim=True) # [Ni, 1])
        y3 = paddle.sum(b1_out[:,2*step:3*step] * t_out[:,2*step:3*step], axis=1, keepdim=True) # [Ni, 1]
        y4 = paddle.sum(b1_out[:,3*step:] * t_out[:,3*step:], axis=1, keepdim=True) # [Ni, 1]
        return paddle.concat([y1, y2, y3, y4], axis=1)  # [Ni, 4]
        # step = int(self.hidden_out/4)
        # y1 = paddle.sum(b1_out[:,0:step] * b2_out[:, 0:step] * t_out[:,0:step], axis=1, keepdim=True) # [Ni, 1])
        # y2 = paddle.sum(b1_out[:,step:2*step] * b2_out[:, step:2*step] * t_out[:,step:2*step], axis=1, keepdim=True) # [Ni, 1])
        # y3 = paddle.sum(b1_out[:,2*step:3*step] * b2_out[:, 2*step:3*step] * t_out[:,2*step:3*step], axis=1, keepdim=True) # [Ni, 1]
        # y4 = paddle.sum(b1_out[:,3*step:] * b2_out[:, 3*step:] * t_out[:,3*step:], axis=1, keepdim=True) # [Ni, 1]
        # return paddle.concat([y1, y2, y3, y4], axis=1)  # [Ni, 4]
