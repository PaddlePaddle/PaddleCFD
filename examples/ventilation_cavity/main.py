import logging
import os
import atexit
import shutil
from timeit import default_timer
from pathlib import Path

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from omegaconf import OmegaConf
from paddle.distributed import fleet
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from functions_data import DataLoader
from functions_data import Dataset
from average_meter import AverageMeterDict
from TransKAN import Transolver


# 采用分布式并行训练策略
strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)


# 设置随机种子，保证参数初始化、dropout等操作的可重复性
def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)


def train(cfg: DictConfig, with_val=False):
    # 设置日志文件的保存位置、级别、格式
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    model = Transolver(**cfg.MODEL)
    if cfg.checkpoint:
        param_dict = paddle.load(f"{cfg.checkpoint}.pdparams")
        model.set_state_dict(param_dict)
    model.train()

    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-4
    )
    if cfg.enable_ddp:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    resume_ep = cfg.resume_ep
    if cfg.checkpoint and os.path.exists(f"{cfg.checkpoint}.pdopt"):
        optim_dict = paddle.load(f"{cfg.checkpoint}.pdopt")
        optimizer.set_state_dict(optim_dict)
        resume_ep = optim_dict["LR_Scheduler"]["last_epoch"]

    error_msg = (
        f"training epochs {cfg.epochs} should be greater than resume epoch, "
        f"which is {resume_ep} now."
    )
    assert cfg.epochs > resume_ep, error_msg

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=optimizer.get_lr(),
        T_max=cfg.epochs,
        last_epoch=resume_ep,
    )
    optimizer.set_lr_scheduler(scheduler)
    logging.info(f"lr of {resume_ep+1} is {optimizer.get_lr()}")

    loss_fn = paddle.nn.L1Loss("mean")

    t1 = default_timer()
    dataset = Dataset(cfg.data_path,cfg.t_len_in, cfg.t_len_out,ratio=cfg.ratio,num_conditions=cfg.num_conditions)
    dataloader_train = DataLoader(dataset).dataloader(batch_size=cfg.batch_size, num_workers=0, shuffle=True)
    if with_val:
        dataset = Dataset(cfg.data_path,cfg.t_len_in, cfg.t_len_out,ratio=cfg.ratio,mode="val")
        dataloader_val = DataLoader(dataset).dataloader(batch_size=1, num_workers=0)
    t2 = default_timer()
    logging.info(f"Loading data took {t2 - t1:.2f} seconds.")

    logging.info(f"Start training {cfg.model} ...")

    # 统一loss保存文件路径
    loss_log_path = os.path.join(cfg.output_dir, "all_steps_loss.txt")
    # 训练开始前写入表头（覆盖旧文件）
    with open(loss_log_path, "w") as f:
        f.write("epoch,step,loss\n")

    for ep in range(resume_ep + 1, cfg.epochs):
        t1 = default_timer()
        train_meter = AverageMeterDict()
        data_iterator = iter(dataloader_train)

        with open(loss_log_path, "a") as f:  # 追加写入
            for i in range(cfg.iters):
                try:
                    inputs, labels = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader_train)
                    inputs, labels = next(data_iterator)

                preds = model(inputs)
                loss = loss_fn(preds, labels)

                loss.backward()
                optimizer.step()
                optimizer.clear_grad(set_to_zero=False)

                train_meter.update({"loss": loss})
                f.write(f"{ep+1},{i+1},{loss.item():.6f}\n")

        scheduler.step()
        t2 = default_timer()

        msg = ""
        for k, v in train_meter.avg.items():
            msg += f"{v.item():.4f}({k}), "
        logging.info(
            "[Train][Epoch %d/%d] time: %.2fs, lr: %e [Loss] %s",
            ep + 1,
            cfg.epochs,
            t2 - t1,
            optimizer.get_lr(),
            msg,
        )

        if with_val and (ep + 1) % cfg.save_freq == 0:
            valid(dataloader_val, model)

        if (ep + 1) % cfg.save_freq == 0 or ep == cfg.epochs - 1 or (ep + 1) == 1:
            paddle.save(
                model.state_dict(), f"{cfg.output_dir}/{cfg.model}_{ep}.pdparams"
            )
            if optimizer:
                paddle.save(
                    optimizer.state_dict(), f"{cfg.output_dir}/{cfg.model}_{ep}.pdopt"
                )


# 模型验证
@paddle.no_grad()
def valid(dataloader_val, model):
    metric_fn = paddle.nn.L1Loss("mean")
    meter = AverageMeterDict()
    
    for i, (inputs, labels) in enumerate(dataloader_val):
        preds = model(inputs)
        metric = metric_fn(preds, labels)
        meter.update({"metric": metric})
        if i == 0:
            # to check values
            print("preds",preds)
            print("labels",labels)
        
    msg = "[Valid][Metric] "
    for k, v in meter.avg.items():
        msg += f"{float(v):.4e}({k}), "
    logging.info(msg)


# 模型测试
def test(cfg: DictConfig):
    # 初始化测试数据集
    dataset_test = Dataset(cfg.data_path, cfg.t_len_in, cfg.t_len_out, ratio=cfg.ratio, mode="test")
    dataloader_test = DataLoader(dataset_test).dataloader(batch_size=32, num_workers=0)  # num_workers：控制并行度

    # 加载已训练的模型
    model = Transolver(**cfg.MODEL)
    model_path = './outputs/1.pdparams'
    try:
        model.set_state_dict(paddle.load(model_path))
        print(f"Successfully loaded model ")
    except Exception as e:
        print(f"Error loading model : {str(e)}")
        return

    # 获取标准化参数
    scaler_params = np.load("scaler_params.npz")
    label_mean = scaler_params["label_mean"]
    label_std = scaler_params["label_std"]

    # 模型预测
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    # 选择展示第几个测试工况
    k = 1
    sample_count = 0
    condition_start = (k - 1) * 4851  # 4851为每个测试工况的样本数
    condition_end = k * 4851

    import time
    # 初始化计数器和时间记录
    total_iterations = 0
    start_time = time.time()
    with paddle.no_grad():
        for inputs, labels in dataloader_test:
            total_iterations += 1  # 每次迭代计数器加1

            batch_size = inputs.shape[0]
            batch_start = sample_count
            batch_end = sample_count + batch_size

            # 检查当前batch是否在目标工况范围内
            if batch_end <= condition_start:
                sample_count += batch_size
                continue  # 还没到目标工况，跳过
            if batch_start >= condition_end:
                break  # 已经超过目标工况，结束循环

            # 只处理目标工况范围内的样本
            start_in_batch = max(0, condition_start - batch_start)
            end_in_batch = min(batch_size, condition_end - batch_start)

            inputs = inputs[start_in_batch:end_in_batch]
            labels = labels[start_in_batch:end_in_batch]

            preds = model(inputs)
            # 反归一化（此处使用预处理阶段的数据）
            preds_denorm = preds * np.sqrt(0.2076) + 1.6745e-5
            labels_denorm = labels * np.sqrt(0.2076) + 1.6745e-5

            all_preds.append(preds_denorm.numpy())
            all_labels.append(labels_denorm.numpy())

            sample_count += batch_size
            if sample_count >= condition_end:
                break  # 已经收集完目标工况的所有样本
    # 计算总时间
    total_time = time.time() - start_time

    # 打印统计信息
    print(f"Total iterations: {total_iterations}")
    print(f"Total time for all iterations: {total_time:.2f} seconds")
    print(f"Average time per iteration: {total_time / total_iterations:.4f} seconds")
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 对预测结果进行高斯滤波
    sigma = 6 / 6  # 将窗口大小转换为sigma (经验法则: sigma ≈ window_size/6)
    preds_smoothed = np.zeros_like(preds)
    for i in range(preds.shape[1]):  # 对每个时间步
        for j in range(preds.shape[2]):  # 对每个特征
            preds_smoothed[:, i, j] = gaussian_filter1d(preds[:, i, j], sigma=sigma)
    preds = preds_smoothed

    # 调整为真实的时间序列
    indices = np.arange(0, preds.shape[0], 50)  # 50为输出序列长度
    pred_selected = preds[indices, :50, 1]      # 这里选择展示力还是力矩
    label_selected = labels[indices, :50, 1]
    combined_columns = np.column_stack([
        pred_selected.ravel(),  # 第一列为pre
        label_selected.ravel()  # 第二列为exp
    ])
    ytrue = combined_columns[:, 0]
    ypre = combined_columns[:, 1]
    mse = np.mean((ypre - ytrue) ** 2)
    print(f"MSE: {mse:.4f}")


    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(combined_columns[:, 1], label="True Values", color="black")
    plt.plot(combined_columns[:, 0], label="Predictions", color="red")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# ！！！！！主程序！！！！！
@hydra.main(version_base=None, config_path=".", config_name="config.yaml") # 提取config.yaml的数据
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed) # 固定随机性
    if cfg.mode == "train":
        print("################## training #####################")
        train(cfg, with_val=cfg.with_val) # 调用训练类，开始训练
    elif cfg.mode == "test":
        print("################## test #####################")
        test(cfg) # 模型测试
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)
        atexit.register(cleanup, cfg) # 删去无用的测试日志
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'valid', 'test'], but got '{cfg.mode}'"
        )


def cleanup(cfg):
    output_dir = Path(cfg.output_dir).parent
    print(f"当前输出目录: {output_dir}")
    try:
        # 关闭所有日志处理器
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        # 重试机制
        import time
        for _ in range(3):  # 最多重试3次
            try:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                    print(f"成功删除目录: {output_dir}")
                    break
            except PermissionError as e:
                print(f"删除重试中... ({e})")
                time.sleep(0.5)
    except Exception as e:
        print(f"无法删除目录 {output_dir}: {e}")



if __name__ == "__main__":
    main()
