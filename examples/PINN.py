import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.datasets import MNIST

# 定义一个MLP网络
class MLP(nn.Layer):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)  # 输入特征大小为784（28x28）
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def pinn_loss(u, x):
    dudx = x.auto_grad()
    return dudx
# 数据预处理
transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])

# 加载MNIST数据集
train_dataset = MNIST(mode='train', transform=transform)
val_dataset = MNIST(mode='test', transform=transform)

# 创建数据加载器
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = MLP(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)

# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch_id, (data, label) in enumerate(train_loader()):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, label)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        if batch_id % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_id}], Loss: {loss.numpy()}')

# 验证模型
model.eval()
correct = 0
total = 0
with paddle.no_grad():
    for data, label in val_loader():
        outputs = model(data)
        _, predicted = paddle.topk(outputs, 1)
        total += label.shape[0]
        correct += (predicted == label.unsqueeze(1)).sum().item()

print(f'Accuracy of the model on the validation images: {100 * correct / total}%')
