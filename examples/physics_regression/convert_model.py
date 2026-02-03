"""
将 PyTorch model.pt 转换为 PaddlePaddle 可加载的框架无关 .pkl 格式。

需要在安装了 PyTorch 的环境中运行。
PaddlePaddle 环境中无法直接 import torch，因此本脚本需在 PyTorch 环境中执行一次。

用法:
    python examples/physics_regression/convert_model.py path/to/model.pt [output.pkl]

示例:
    # 在 PyTorch 环境中运行（仅需一次）
    conda activate <torch_env>
    python examples/physics_regression/convert_model.py ./model.pt ./model.pkl

    # 然后在 PaddlePaddle 环境中加载转换后的文件
    from ppcfd.models.physicsregression import PhyReg
    phyreg = PhyReg("./model.pkl")
"""
import argparse
import os
import pickle
import sys


def convert_pytorch_to_numpy_pkl(src_path: str, dst_path: str = None) -> str:
    """
    将 PyTorch .pt 模型文件转换为框架无关的 numpy pickle (.pkl) 格式。

    转换逻辑:
    - 用 torch.load() 读取 PyTorch 格式文件
    - 将 torch.Tensor 权重通过 .cpu().numpy() 转为 numpy 数组
    - 将 params（argparse.Namespace）转为普通 dict
    - 用标准 pickle.dump() 保存（无框架依赖）

    Args:
        src_path: PyTorch .pt 文件路径
        dst_path: 输出 .pkl 文件路径，默认为同目录下同名 .pkl

    Returns:
        输出文件路径
    """
    try:
        import torch
    except ImportError:
        print("错误: 未找到 PyTorch。请在 PyTorch 环境中运行本脚本。", file=sys.stderr)
        print("提示: conda activate <torch_env>", file=sys.stderr)
        sys.exit(1)

    from argparse import Namespace

    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".pkl"

    print(f"正在加载 PyTorch 模型: {src_path} ...")
    model = torch.load(src_path, map_location="cpu", weights_only=False)

    # 验证文件格式
    required_keys = ("embedder", "encoder", "decoder", "params")
    missing = [k for k in required_keys if k not in model]
    if missing:
        print(f"错误: 文件缺少必要字段: {missing}", file=sys.stderr)
        print(f"文件包含的字段: {list(model.keys())}", file=sys.stderr)
        sys.exit(1)

    # 转换 params: Namespace → dict
    params = model["params"]
    if isinstance(params, Namespace):
        params = vars(params)
    elif not isinstance(params, dict):
        print(f"警告: params 类型为 {type(params)}，尝试直接使用。", file=sys.stderr)

    # 将 state_dict 中的 Tensor 转为 numpy 数组
    def state_dict_to_numpy(sd: dict) -> dict:
        result = {}
        for k, v in sd.items():
            if hasattr(v, "cpu") and hasattr(v, "numpy"):
                arr = v.cpu().numpy()
                result[k] = arr
            else:
                result[k] = v
        return result

    save = {
        "embedder": state_dict_to_numpy(model["embedder"]),
        "encoder":  state_dict_to_numpy(model["encoder"]),
        "decoder":  state_dict_to_numpy(model["decoder"]),
        "params":   params,
    }

    print(f"正在保存到: {dst_path} ...")
    with open(dst_path, "wb") as f:
        pickle.dump(save, f, protocol=4)

    print(f"\n✅ 转换成功！")
    print(f"   源文件: {src_path}")
    print(f"   输出:   {dst_path}")
    print(f"\n现在在 PaddlePaddle 环境中加载转换后的文件:")
    print(f"    from ppcfd.models.physicsregression import PhyReg")
    print(f"    phyreg = PhyReg('{dst_path}')")
    return dst_path


def main():
    parser = argparse.ArgumentParser(
        description="将 PyTorch model.pt 转换为 PaddlePaddle 可加载的 .pkl 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("src", help="PyTorch .pt 文件路径")
    parser.add_argument(
        "dst",
        nargs="?",
        default=None,
        help="输出 .pkl 文件路径（默认与源文件同目录，扩展名改为 .pkl）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        print(f"错误: 文件不存在: {args.src}", file=sys.stderr)
        sys.exit(1)

    convert_pytorch_to_numpy_pkl(args.src, args.dst)


if __name__ == "__main__":
    main()
