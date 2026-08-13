import numpy as np


def relative_l2(a, b):

    return np.linalg.norm(a - b, 2) / np.linalg.norm(a, 2)


def main():

    d = np.load("./test_SW.npz")

    S_test = d["S_test"]

    num_test = 1000

    Nx = 32
    Ny = 32
    Nt = 5

    ds = 3

    S_test = S_test.reshape(num_test, Nt * Nx * Ny, ds)

    pred = np.load("results/pred_test.npy")

    err_rho = []
    err_u = []
    err_v = []

    for i in range(num_test):

        err_rho.append(relative_l2(S_test[i, :, 0], pred[i, :, 0]))

        err_u.append(relative_l2(S_test[i, :, 1], pred[i, :, 1]))

        err_v.append(relative_l2(S_test[i, :, 2], pred[i, :, 2]))

    print("rho error mean:", np.mean(err_rho))

    print("u error mean:", np.mean(err_u))

    print("v error mean:", np.mean(err_v))


if __name__ == "__main__":

    main()