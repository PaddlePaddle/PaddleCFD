import argparse
import os
import numpy as np
import paddle

from utils import load_model
from ppcfd.models.nomad.advection.operator_model import OperatorModel


def main(n, decoder):

    # =========
    # 参数设置
    # =========
    P = 25600
    m = 256
    dy = 2
    ds = 1

    num_test = 100

    # =========
    # 构建模型结构
    # =========
    if decoder == "nonlinear":

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    elif decoder == "linear":

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [dy, 100, 100, 100, 100, 100, ds * n]

    else:
        raise ValueError("decoder must be 'linear' or 'nonlinear'")

    model = OperatorModel(
        branch_layers,
        trunk_layers,
        m=m,
        P=P,
        n=n,
        decoder=decoder,
        ds=ds,
    )

    # =========
    # 加载模型
    # =========
    load_model(
        model,
        "checkpoints/nomad_advection_model.pdparams",
    )

    # =========
    # 读取数据
    # =========
    data = np.load("./pure_advection_traintest.npz")

    U_test = data["ic"][-num_test:]

    # 转 tensor
    U_test = paddle.to_tensor(U_test, dtype="float32")

    # reshape
    U_test = paddle.reshape(U_test, [num_test, m, 1])

    # 构造 y (t,x)
    Nx = 256
    Nt = 100

    x = np.linspace(0, 2, num=Nx)
    t = np.linspace(0, 1, num=Nt)

    TT, XX = np.meshgrid(t, x, indexing="ij")

    y = np.concatenate(
        (TT.flatten()[:, None], XX.flatten()[:, None]),
        axis=-1,
    )

    y_test = np.tile(y[None, ...], (num_test, 1, 1))

    y_test = paddle.to_tensor(y_test, dtype="float32")

    # =========
    # 推理
    # =========
    model.eval()

    with paddle.no_grad():

        pred = model.predict((U_test, y_test))

    # =========
    # 保存结果
    # =========
    os.makedirs("results", exist_ok=True)

    np.savez(
        "results/prediction.npz",
        prediction=pred.numpy(),
    )

    print("Inference finished.")
    print("Results saved to results/prediction.npz")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NOMAD Advection Inference"
    )

    parser.add_argument(
        "n",
        type=int,
        help="Latent dimension of solution manifold",
    )

    parser.add_argument(
        "decoder",
        type=str,
        help="Decoder type: linear or nonlinear",
    )

    args = parser.parse_args()

    n = args.n
    decoder = args.decoder

    main(n, decoder)