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
        y = paddle.einsum('bi,ni->bn', b1_out, t_out)  # 或 paddle.matmul(y_branch, y_trunk.T)
    
        return y  