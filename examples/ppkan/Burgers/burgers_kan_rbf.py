import os
import sys

import numpy as np
import paddle

sys.path.append("/shared/KAN_paddle/DOKAN_pp/")
import csv
import random
import time
from statistics import mean, median

import GPUtil
import matplotlib.pyplot as plt
import pandas as pd
from kan_efficiency import *
from kan_rbf import *
from tqdm import tqdm
from utils import *

path = os.path.dirname(os.path.abspath(__file__))

available_gpus = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.5, maxMemory=0.5)
if len(available_gpus) == 0:
    print("No GPU available, using CPU")
    paddle.set_device("cpu")
else:
    selected_gpu = available_gpus[0]
    print(f"Using GPU {selected_gpu}")
    paddle.set_device(f"gpu:{selected_gpu}")

# Load Data
x = np.load("{path}/data/x.npy")
x = x[::8]
u0_train = np.load("{path}/data/u0_train.npy")
u0_train = u0_train[:, ::8]
u0_test = np.load("{path}/data/u0_test.npy")
u0_test = u0_test[:, ::8]
u_train = np.load("{path}/data/u_train.npy")
u_train = u_train[:, ::8]
u_test = np.load("{path}/data/u_test.npy")
u_test = u_test[:, ::8]
dtype = paddle.float32

u0_train = paddle.to_tensor(u0_train, dtype=dtype)
u0_test = paddle.to_tensor(u0_test, dtype=dtype)
u_train = paddle.to_tensor(u_train, dtype=dtype)
u_test = paddle.to_tensor(u_test, dtype=dtype)
x = paddle.to_tensor(x, dtype=dtype).reshape([x.shape[0], -1])

print("Data loaded.")
print(f"u0_train shape = {tuple(u0_train.shape)}")
print(f"u_train shape = {tuple(u_train.shape)}")
print(f"u0_test shape = {tuple(u0_test.shape)}")
print(f"u_test shape = {tuple(u_test.shape)}")
print(f"x shape = {tuple(x.shape)}")
print(f"u0_train max = {u0_train.max()}, u0_train min = {u0_train.min()}")
print(f"x max = {x.max()}, x min = {x.min()}")


# Define the trunk/branch KAN parameters.
# Trunk and branch net parameters:
input_dim_trunk = x.shape[1]
input_dim_branch = u0_train.shape[1]
HD = 40  # trunk/branch output neurons
hid_trunk = 10  # trunk hidden neurons
num_layer_trunk = 1  # trunk hidden layers
hid_branch = 10  # branch hidden neurons
num_layer_branch = 1  # branch hidden layers
# KAN parameters
grid_opt = True  # Grid points trainable or not
apply_base_update = True  # apply base update or not
grid_count = 20  # number of grid points
init_scale = 0.01
noise_scale = 0.01
learning_rate = 1e-3
batch_size = 128
epochs = 10000
gamma = 0.95
step_size = 500  # step size for learning rate decay
random_seed = 2323
branch_kan_func = "rbf"
trunk_kan_func = "rbf"

trunk_min_grid = x.min()
trunk_max_grid = x.max()
branch_min_grid = u0_train.min()
branch_max_grid = u0_train.max()
width_trunk = kan_width(input_dim_trunk, hid_trunk, num_layer_trunk, HD)
width_branch = kan_width(input_dim_branch, hid_branch, num_layer_branch, HD)

print(f"Trunk net hidden neurons: {width_trunk}, num layers: {num_layer_trunk}, width: {width_trunk}")
print(f"Branch net hidden neurons: {width_branch}, num layers: {num_layer_branch}, width: {width_branch}")

out_0 = u0_train.clone().detach()
out_0.stop_gradient = False
u0_train = out_0.to(dtype)
out_1 = u_train.clone().detach()
out_1.stop_gradient = not True
u_train = out_1.to(dtype)
out_2 = u_test.clone().detach()
out_2.stop_gradient = not True
u_test = out_2.to(dtype)
out_3 = u_test.clone().detach()
out_3.stop_gradient = not True
u_test = out_3.to(dtype)


# Define the trunk/branch KAN model, loss function, optimizer and scheduler.
model_rbf = KANONet(
    width_branch=width_branch,
    width_trunk=width_trunk,
    trunk_min_grid=trunk_min_grid,
    trunk_max_grid=trunk_max_grid,
    branch_min_grid=branch_min_grid,
    branch_max_grid=branch_max_grid,
    grid_count=grid_count,
    grid_opt=grid_opt,
    apply_base_update=apply_base_update,
    noise_scale=noise_scale,
    dtype=dtype,
    branch_kan_func=branch_kan_func,
    trunk_kan_func=trunk_kan_func,
)
criterion = RMSLoss()
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model_rbf.parameters(), weight_decay=0.0)
tmp_lr = paddle.optimizer.lr.StepDecay(step_size=step_size, gamma=gamma, learning_rate=optimizer.get_lr())
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr

set_seed(random_seed)
num_learnable_parameters = count_learnable_parameters(model_rbf)
print(num_learnable_parameters)

train_losses = []
test_losses = []
start_time = time.time()
mean_test_loss = 0.0
results = []

for epoch in range(epochs):
    model_rbf.train()
    total_loss = 0
    indices = range(0, len(u0_train), batch_size)
    if epoch % 500 == 0:
        progress_bar = tqdm(range(0, len(u0_train), batch_size), desc=f"Epoch{epoch + 1}/{epochs}")
    else:
        progress_bar = range(0, len(u0_train), batch_size)
    for i in progress_bar:
        u0_batch = u0_train[i : i + batch_size]
        u_batch = u_train[i : i + batch_size]

        optimizer.clear_grad()
        u_pred = model_rbf(u0_batch, x)
        loss = criterion(u_pred, u_batch)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
        if epoch % 500 == 0:
            progress_bar.set_postfix({"Batch loss": loss.item()})
    avg_loss = total_loss / (len(u0_train) // batch_size + 1)
    train_losses.append(avg_loss)
    test_loss = test_model(
        model=model_rbf, criterion=criterion, c_test=u0_test, y_test=u_test, x=x, batch_size=batch_size
    )
    test_losses.append(test_loss)
    results.append([epoch + 1, avg_loss, test_loss])
    if epoch % 500 == 0:
        print(f"After Epoch {epoch}, Average Train Loss: {avg_loss}, Test Loss: {test_loss}")
    scheduler.step()
end_time = time.time()
training_time = end_time - start_time
print("###################################################################################################")
print("###################################################################################################")
print("###################################################################################################")
print(f"Training time: {training_time:.2f} seconds")
print(f"The number of learnable parameters in the model: {num_learnable_parameters}")
print("###################################################################################################")
print("###################################################################################################")
print("###################################################################################################")
model_rbf.eval()
y_pred_list = []
batch_size = 100
with paddle.no_grad():
    for i in range(0, len(u0_test), batch_size):
        batch_loads = u0_test[i : i + batch_size]
        batch_pred = model_rbf(batch_loads, x)
        y_pred_list.append(batch_pred.cpu().numpy())
y_pred_deepokan = np.concatenate(y_pred_list, axis=0)
np.save("u1_pred_" + branch_kan_func + "_deepokan.npy", y_pred_deepokan)
print("Predictions generated and saved successfully.")
print(f"Shape of u1_pred_deepokan: {tuple(y_pred_deepokan.shape)}")
with open("training_results_" + branch_kan_func + "_deepokan.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epoch", "Train Loss", "Test Loss"])
    csv_writer.writerows(results)
print(f"Training results saved in 'loss_data_{branch_kan_func}_deepokan.csv'")

idx = np.random.choice(u_test.shape[0], 1, replace=False)

model_rbf.eval()
with paddle.no_grad():
    u_pred = model_rbf(u0_test[idx], x)
    plt.figure()
    plt.plot(x.numpy(), u_test[idx].numpy().flatten(), "r-", lw=2.5, label="True")
    plt.plot(x.numpy(), u_pred.numpy().flatten(), "b--", lw=1, label="Predicted")
    plt.legend()
    plt.title(f"Sample {idx+1}: Prediction of Burgers equation with RBF-KAN")
    plt.show()

plt.figure()
results = np.asarray(results)
plt.plot(results[:, 0], results[:, 1], "r", label="Training Loss")
plt.plot(results[:, 0], results[:, 2], "b--", lw=2, label="Test Loss")
plt.xscale("log")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Training and Test Loss of RNF-KAN")
plt.show()


"""Bspline KAN for comparison"""
branch_kan_func = "bspline"
trunk_kan_func = "bspline"
model_bspline = KANONet(
    width_branch=width_branch,
    width_trunk=width_trunk,
    trunk_min_grid=trunk_min_grid,
    trunk_max_grid=trunk_max_grid,
    branch_min_grid=branch_min_grid,
    branch_max_grid=branch_max_grid,
    grid_count=grid_count,
    grid_opt=grid_opt,
    apply_base_update=apply_base_update,
    noise_scale=noise_scale,
    dtype=dtype,
    branch_kan_func=branch_kan_func,
    trunk_kan_func=trunk_kan_func,
)
criterion = RMSLoss()
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model_bspline.parameters(), weight_decay=0.0)
tmp_lr = paddle.optimizer.lr.StepDecay(step_size=step_size, gamma=gamma, learning_rate=optimizer.get_lr())
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr
set_seed(232)
num_learnable_parameters = count_learnable_parameters(model_bspline)
print(num_learnable_parameters)
train_losses = []
test_losses = []
start_time = time.time()
mean_test_loss = 0.0
results = []

for epoch in range(epochs):
    # model_bspline().train()
    total_loss = 0
    indices = range(0, len(u0_train), batch_size)
    if epoch % 500 == 0:
        progress_bar = tqdm(range(0, len(u0_train), batch_size), desc=f"Epoch{epoch + 1}/{epochs}")
    else:
        progress_bar = range(0, len(u0_train), batch_size)
    for i in progress_bar:
        u0_batch = u0_train[i : i + batch_size]
        u_batch = u_train[i : i + batch_size]

        optimizer.clear_grad()
        u_pred = model_bspline(u0_batch, x)
        loss = criterion(u_pred, u_batch)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
        if epoch % 500 == 0:
            progress_bar.set_postfix({"Batch loss": loss.item()})
    avg_loss = total_loss / (len(u0_train) // batch_size + 1)
    train_losses.append(avg_loss)
    test_loss = test_model(
        model=model_bspline, criterion=criterion, c_test=u0_test, y_test=u_test, x=x, batch_size=batch_size
    )
    test_losses.append(test_loss)
    results.append([epoch + 1, avg_loss, test_loss])
    if epoch % 500 == 0:
        print(f"After Epoch {epoch}, Average Train Loss: {avg_loss}, Test Loss: {test_loss}")
    scheduler.step()
end_time = time.time()
training_time = end_time - start_time
print("###################################################################################################")
print("###################################################################################################")
print("###################################################################################################")
print(f"Training time: {training_time:.2f} seconds")
print(f"The number of learnable parameters in the model: {num_learnable_parameters}")
print("###################################################################################################")
print("###################################################################################################")
print("###################################################################################################")
model_bspline.eval()
u_pred_list = []
batch_size = 100
with paddle.no_grad():
    for i in range(0, len(u_test), batch_size):
        batch_loads = u0_test[i : i + batch_size]
        batch_pred = model_bspline(batch_loads, x)
        u_pred_list.append(batch_pred.cpu().numpy())
u_pred_deepokan = np.concatenate(u_pred_list, axis=0)
np.save("u_pred_" + branch_kan_func + "_deepokan.npy", u_pred_deepokan)
print("Predictions generated and saved successfully.")
print(f"Shape of u_pred_deepokan: {tuple(u_pred_deepokan.shape)}")
with open("loss_data_" + branch_kan_func + "_deepokan.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epoch", "Train Loss", "Test Loss"])
    csv_writer.writerows(results)
print(f"Training results saved in 'loss_data_{branch_kan_func}_deepokan.csv'")

idx = np.random.choice(u0_test.shape[0], 1, replace=False)

with paddle.no_grad():
    y_pred = model_bspline(u0_test[idx], x)
    plt.figure()

    plt.plot(x.numpy(), u_test[idx].numpy().flatten(), "r-", label="True")
    plt.plot(x.numpy(), y_pred.numpy().flatten(), "b--", lw=2, label="Predicted")
    plt.legend()
    plt.title(f"Sample {idx+1}: Prediction of Burgers equation by B-spline-KAN")
    plt.show()

plt.figure()
results = np.asarray(results)
plt.plot(results[:, 0], results[:, 1], "r", label="Training Loss")
plt.plot(results[:, 0], results[:, 2], "b--", lw=2, label="Test Loss")
plt.xscale("log")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Loss of B-spline KAN")
plt.legend()
plt.show()
