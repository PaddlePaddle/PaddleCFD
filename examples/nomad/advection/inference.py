import argparse
import os
import numpy as np
import paddle

from utils import load_model
from ppcfd.models.nomad.advection.operator_model import OperatorModel


def main(
    n,
    decoder,
    num_test=1000,
    infer_batch_size=100,
    checkpoint="checkpoints/nomad_advection_model.pdparams",
):

    # =========
    # 参数设置
    # =========
    P = 25600
    m = 256
    dy = 2
    ds = 1

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
        checkpoint,
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

    pred_batches = []

    with paddle.no_grad():
        for start in range(0, num_test, infer_batch_size):
            end = min(start + infer_batch_size, num_test)
            pred_batches.append(
                model.predict((U_test[start:end], y_test[start:end])).numpy()
            )

    pred = np.concatenate(pred_batches, axis=0)

    # =========
    # 保存结果
    # =========
    os.makedirs("results", exist_ok=True)

    np.savez(
        "results/prediction.npz",
        prediction=pred,
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
    parser.add_argument(
        "--num-test",
        type=int,
        default=1000,
        help="Number of test samples to run inference on.",
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=100,
        help="Number of test samples per inference batch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/nomad_advection_model.pdparams",
        help="Path to model parameters.",
    )

    args = parser.parse_args()

    main(
        args.n,
        args.decoder,
        args.num_test,
        args.infer_batch_size,
        args.checkpoint,
    )
