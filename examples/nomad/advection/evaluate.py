import numpy as np

# 读取预测
pred = np.load("results/prediction.npz")["prediction"]

# 读取真实数据
data = np.load("./pure_advection_traintest.npz")

s_test = data["solution"][-100:]

# reshape
s_test = s_test.reshape(100, 25600, 1)

errors = []

for i in range(len(pred)):

    err = np.linalg.norm(
        s_test[i,:,0] - pred[i,:,0], 2
    ) / np.linalg.norm(
        s_test[i,:,0], 2
    )

    errors.append(err)

print("Mean error:", np.mean(errors))
print("Std error:", np.std(errors))
print("Min error:", np.min(errors))
print("Max error:", np.max(errors))