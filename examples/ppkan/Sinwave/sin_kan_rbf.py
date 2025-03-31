import paddle
import numpy as np
import os
import sys
sys.path.append('/shared/KAN_paddle/DOKAN_pp/')
from utils import *
from kan_efficiency import *
from kan_rbf import *
import time
import pandas as pd
from tqdm import tqdm
import random
from statistics import mean, median
import matplotlib.pyplot as plt
import GPUtil
import csv


path = os.path.dirname(os.path.abspath(__file__))

available_gpus = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)
if len(available_gpus) == 0:
    print("No GPU available, using CPU")
    paddle.set_device('cpu')
else:
    selected_gpu = available_gpus[0]
    print(f"Using GPU {selected_gpu}")
    paddle.set_device(f'gpu:{selected_gpu}')

# Load Data
x = np.load(f'{path}/data/x.npy')
c_train = np.load(f'{path}/data/c_train.npy')
c_test = np.load(f'{path}/data/c_test.npy')
y_train = np.load(f'{path}/data/y_train.npy')
y_test = np.load(f'{path}/data/y_test.npy')
dtype = paddle.float32

c_train = paddle.to_tensor(c_train, dtype=dtype)
c_test = paddle.to_tensor(c_test, dtype=dtype)
y_train = paddle.to_tensor(y_train, dtype=dtype)
y_test = paddle.to_tensor(y_test, dtype=dtype)
x = paddle.to_tensor(x, dtype=dtype).reshape([x.shape[0], -1])

print(f'Data loaded.')
print(f'c_train shape = {tuple(c_train.shape)}')
print(f'y_train shape = {tuple(y_train.shape)}')
print(f'c_test shape = {tuple(c_test.shape)}')
print(f'y_test shape = {tuple(y_test.shape)}')
print(f'x shape = {tuple(x.shape)}')
print(f'c_train max = {c_train.max()}, c_train min = {c_train.min()}')
print(f'x max = {x.max()}, x min = {x.min()}')


# Define the trunk/branch KAN parameters.
# Trunk and branch net parameters:
input_dim_trunk = x.shape[1]
input_dim_branch = c_train.shape[1]
HD = 40 # trunk/branch output neurons
hid_trunk = 50 # trunk hidden neurons
num_layer_trunk = 2 # trunk hidden layers
hid_branch = 50 # branch hidden neurons
num_layer_branch = 2 # branch hidden layers
# KAN parameters
grid_opt = False # Grid points trainable or not
apply_base_update = True # apply base update or not
grid_count = 20 # number of grid points
init_scale = 0.01
noise_scale = 0.01
# Training parameters
learning_rate = 1e-2
batch_size = 1024
epochs = 20000
gamma = 0.95
step_size = 500 # step size for learning rate decay
random_seed = 2323
branch_kan_func = 'rbf'
trunk_kan_func = 'rbf'

trunk_min_grid = x.min()
trunk_max_grid = x.max()
branch_min_grid = c_train.min()
branch_max_grid = c_train.max()
width_trunk = kan_width(input_dim_trunk, hid_trunk, num_layer_trunk, HD)
width_branch = kan_width(input_dim_branch, hid_branch, num_layer_branch, HD)

print(f'Trunk net hidden neurons: {width_trunk}, num layers: {num_layer_trunk}, width: {width_trunk}')
print(f'Branch net hidden neurons: {width_branch}, num layers: {num_layer_branch}, width: {width_branch}')

out_0 = c_train.clone().detach()
out_0.stop_gradient = False
c_train = out_0.to(dtype)
out_1 = y_train.clone().detach()
out_1.stop_gradient = not True
y_train = out_1.to(dtype)
out_2 = c_test.clone().detach()
out_2.stop_gradient = not True
c_test = out_2.to(dtype)
out_3 = y_test.clone().detach()
out_3.stop_gradient = not True
y_test = out_3.to(dtype)

model_rbf = KANONet(width_branch=width_branch, width_trunk=width_trunk, 
                 trunk_min_grid=trunk_min_grid, trunk_max_grid=trunk_max_grid,
                 branch_min_grid=branch_min_grid, branch_max_grid=branch_max_grid,
                 grid_count=grid_count, grid_opt=grid_opt,
                 apply_base_update=apply_base_update, noise_scale=noise_scale, dtype=dtype,
                 branch_kan_func=branch_kan_func, trunk_kan_func=trunk_kan_func)
criterion = RMSLoss()
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model_rbf.parameters(), weight_decay=0.0)
tmp_lr = paddle.optimizer.lr.StepDecay(step_size=step_size, gamma=gamma, learning_rate=optimizer.get_lr())
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr

set_seed(random_seed)
num_learnable_parameters = count_learnable_parameters(model_rbf)
print(num_learnable_parameters)
for name, param in model_rbf.named_parameters():
    if not param.stop_gradient:
        print(f"Parameter Name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Number of Parameters: {param.numel()}")


train_losses = []
test_losses = []
start_time = time.time()
mean_test_loss = 0.0
results = []

for epoch in range(epochs):
    model_rbf.train()
    total_loss  =0
    indices = range(0, len(c_train), batch_size)
    if epoch % 500 == 0:
        progress_bar = tqdm(range(0, len(c_train), batch_size), desc=f'Epoch{epoch + 1}/{epochs}')
    else:
        progress_bar = range(0, len(c_train), batch_size)
    for i in progress_bar:
        c_batch = c_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        optimizer.clear_grad()
        y_pred = model_rbf(c_batch, x)
        loss = criterion(y_pred, y_batch)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
        if epoch % 500 == 0:
            progress_bar.set_postfix({'Batch loss': loss.item()})
    avg_loss = total_loss / (len(c_train) // batch_size + 1)
    train_losses.append(avg_loss)
    test_loss = test_model(model=model_rbf, criterion=criterion, c_test=c_test, y_test=y_test, x=x, batch_size=batch_size)
    test_losses.append(test_loss)
    results.append([epoch+1, avg_loss, test_loss])
    if epoch % 500 == 0:
        print(f'After Epoch {epoch}, Average Train Loss: {avg_loss}, Test Loss: {test_loss}')
    scheduler.step()
end_time = time.time()
training_time = end_time - start_time

print(
    '###################################################################################################'
    )
print(
    '###################################################################################################'
    )
print(
    '###################################################################################################'
    )
print(f'Training time: {training_time:.2f} seconds')
print(
    f'The number of learnable parameters in the model: {num_learnable_parameters}'
    )
print(
    '###################################################################################################'
    )
print(
    '###################################################################################################'
    )
print(
    '###################################################################################################'
    )
model_rbf.eval()
y_pred_list = []
batch_size = 100
with paddle.no_grad():
    for i in range(0, len(c_test), batch_size):
        batch_loads = c_test[i:i + batch_size]
        batch_pred = model_rbf(batch_loads, x)
        y_pred_list.append(batch_pred.cpu().numpy())
y_pred_deepokan = np.concatenate(y_pred_list, axis=0)
np.save('y_pred_'+branch_kan_func+'_deepokan.npy', y_pred_deepokan)
print('Predictions generated and saved successfully.')
print(f'Shape of y_pred_deepokan: {tuple(y_pred_deepokan.shape)}')
with open('training_results_'+branch_kan_func+'_deepokan.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
    csv_writer.writerows(results)
print(f"Training results saved in 'loss_data_{branch_kan_func}_deepokan.csv'")

idx = np.random.choice(c_test.shape[0], 1, replace=False)


#plt.title(f'Sample {idx+1}: c1={c_test[idx,0].item():.2f}, c2={c_test[idx,1].item():.2f}, c3={c_test[idx,2].item():.2f}')
plt.show()
with paddle.no_grad():
    y_pred = model_rbf(c_test[idx], x)
    plt.figure()
    
    plt.plot(x.numpy(), y_test[idx].numpy().flatten(), 'r-', label='True')
    plt.plot(x.numpy(), y_pred.numpy().flatten(), 'b--', lw=2, label='Predicted')
    plt.legend()
    #plt.title(f'Sample {idx+1}: c1={c_test[idx,0].item():.2f}, c2={c_test[idx,1].item():.2f}, c3={c_test[idx,2].item():.2f}')
    plt.show()

plt.figure()
results = np.asarray(results)
plt.plot(results[:,0], results[:,1], 'r', label='Training Loss')
plt.plot(results[:,0], results[:,2], 'b--', lw = 2,label='Test Loss')
plt.xscale('log')
plt.xlabel('Epoch')
plt.yscale('log')
plt.ylabel('Loss')
plt.title(f'Loss of B-spline KAN')
plt.legend()
plt.show()