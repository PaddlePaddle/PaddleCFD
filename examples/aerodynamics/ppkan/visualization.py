import re
import matplotlib.pyplot as plt
import os
import paddle
import numpy as np

def parse_log_file(log_file_path):
    train_epochs = []
    train_losses = []
    valid_metrics = []
    valid_epochs = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Parse training loss
            if '[Train][Epoch' in line:
                epoch_match = re.search(r'Epoch (\d+)/', line)
                loss_match = re.search(r'Loss\] ([\d\.]+)\(loss\)', line)
                if epoch_match and loss_match:
                    epoch = int(epoch_match.group(1))
                    loss = float(loss_match.group(1))
                    train_epochs.append(epoch)
                    train_losses.append(loss)
            
            # Parse validation metric
            elif '[Valid][Metric]' in line:
                metric_match = re.search(r'Metric\]([\d\.e+-]+)\(metric\)', line)
                if metric_match:
                    metric = float(metric_match.group(1))
                    # The validation happens after the corresponding epoch
                    if len(valid_epochs)==0:
                        valid_epoch = 1
                    else:
                        valid_epoch = len(valid_metrics) * 10  # assuming validation every 10 epochs
                    valid_metrics.append(metric)
                    valid_epochs.append(valid_epoch)
    
    return train_epochs, train_losses, valid_epochs[:-1], valid_metrics[:-1]

def plot_metrics(train_epochs, train_losses, valid_epochs, valid_metrics, output_dir: str=None):
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(train_epochs, train_losses, 'b-', label='Training Loss')
    
    # Plot validation metric
    plt.plot(valid_epochs, valid_metrics, 'r-', label='Validation Metric')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.title('Training Loss and Validation Metric')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        print(f"curve saved as: {os.path.join(output_dir, 'loss_curves.png')}")
    
    plt.show()

def plot_predictions(model, X, y, labels_mean, labels_std, output_dir):
    model.eval()
    X = paddle.to_tensor(X, dtype="float32")

    branch1 = X[:, 0:63]
    print(branch1.shape)
    trunk_x = X[:, 63:]
    print(trunk_x.shape)
    y_pred = model({
        'branch1': branch1,
        'trunk': trunk_x
        })
    print(y_pred.shape)

    epsilon = 1e-4
    relative_error = np.abs(y_pred.numpy() - y) / (np.abs(y) + epsilon)

    y_pred = y_pred.numpy() * labels_std + labels_mean
    y = y * labels_std + labels_mean
    
    coords = trunk_x.numpy()

    plt.figure(figsize=(18, 36))
    # predictions
    plt.subplot(3, 1, 1)
    plt.scatter(coords[:,0], coords[:,1], c=y_pred[:,0], vmin=np.min(y[:,0]), vmax=np.max(y[:,0]),  cmap='jet', s=5)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 0.5)
    plt.title('Predictions')
    plt.colorbar()
    # true labels
    plt.subplot(3, 1, 2)
    plt.scatter(coords[:,0], coords[:,1], c=y[:,0], cmap='jet', s=5)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 0.5)
    plt.title('Ground Truth')
    plt.colorbar()
    # relative error
    plt.subplot(3, 1, 3)
    plt.scatter(coords[:,0], coords[:,1], c=relative_error[:, 0], vmin = 0.0, vmax = 5.0, cmap='hot', s=5)
    plt.title('Relative Error')
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 0.5)
    plt.colorbar()


    plt.tight_layout()
    plt.title(f'Pressure Prediction')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pressure_pred.png'), dpi=300, bbox_inches='tight')
        print(f"curve saved as: {os.path.join(output_dir, './visualization/pressure_pred.png')}")
    plt.close()