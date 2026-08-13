import numpy as np

pred = np.load("results/prediction.npz")["prediction"]
data = np.load("./pure_advection_traintest.npz")

num_test = pred.shape[0]
s_test = data["solution"][-num_test:]

s_test = s_test.reshape(num_test, 25600, 1)

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
