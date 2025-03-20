import numpy as np
import random
import paddle


def generate_sine_data(num_points, num_samples, sinewave_range=[-1.0, 1.0], c_range = [0.0, 1.0]):
    num_points = int(num_points)
    num_samples = int(num_samples)
    x = np.linspace(float(sinewave_range[0]), float(sinewave_range[1]), num_points)
    c = np.random.uniform(float(c_range[0]), float(c_range[1]), size=(num_samples, 3))

    y = np.zeros((num_samples, num_points))

    for i in range(num_samples):
        c1, c2, c3 = c[i]
        y[i] = np.cos(c1 * np.pi * x) - np.sin(c2 * np.pi * x**2) * np.cos(c3 * np.pi * x**3)

    return x, c, y

def kan_width(input_dim, W, repeat_hid, output_dim):
    width = [input_dim] + [W] * repeat_hid + [output_dim]
    return width

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)
    if paddle.device.cuda.device_count() >= 1:
        paddle.seed(seed=seed)
        paddle.seed(seed=seed)

def count_learnable_parameters(model):
    return sum(p.size for p in model.parameters() if not p.stop_gradient)

def test_model(model, criterion, c_test, y_test, x, batch_size):
    model.eval()
    total_loss = 0
    num_batches = len(c_test) // batch_size + 1
    with paddle.no_grad():
        for i in range(0, len(c_test), batch_size):
            c_batch = c_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            pred = model(c_batch, x)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
    loss_avg = total_loss / num_batches
    return loss_avg