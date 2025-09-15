# PaddleCFD 0.3.0 Release Notes

## PaddleCFD 0.3.0 概述
PaddleCFD是一个基于飞桨深度学习框架（PaddlePaddle）、围绕计算流体力学（CFD）任务的深度学习工具套件，用于流体力学方程发现、数值模拟计算加速、流动形状优化、流动控制策略发现等。PaddleCFD 0.3.0版本主要聚焦CFD数值模拟计算加速，打造了基于飞桨框架的流体计算代理模型，在模型精度和计算效率上取得了突破，给流体力学领域的科研工作者提供高基线模型、给企业用户提供开箱即用的工具库。PaddleCFD 0.3.0版本的主要特性说明如下：
- **主流前沿代理模型覆盖**：PaddleCFD 0.3.0版本涵盖了傅里叶神经算子（FNO）、DeepONet等科学计算特有模型和Transformer、扩散模型、KAN等深度学习前沿模型，这些模型具备强大的算子学习、预测和生成能力，是AI+流体力学领域的热点研究前沿（数据来源：Clarivate Analytics Essential Science Indicators）。
- **精度/计算效率领先**：PaddleCFD 0.3.0版本在论文公开的模型版本上进行了改进和适配，实现了模型精度和计算效率的双提升。PPFNO模型，通过积分修正，风阻系数测试集预测平均相对误差小于3%（基线模型~8%）；通过算子融合，训练速度提升188%，推理速度提升159%，实现千万网格秒级推理。PPTransfomer模型，通过飞桨框架动转静+神经网络编译器加速训练速度提升29.4%, 双卡并行效率达到90.2%，支持千万级别网格模型并行推理。PPKAN模型，相比传统MLP神经网络，参数量相当的情况下精度提升30%。PPDifusion模型，通过数据并行，实现模型可扩展加速，单机多卡并行效率达99.4%以上。PPDeepONet模型，在MultiONet网络结构基础上，通过二阶优化器SOAP，实现模型精度提升10%左右。
- **全场景支持产业落地**：PaddleCFD 0.3.0版本更加注重产业应用落地，通过产业真实业务场景打磨提升模型精度和计算效率，完善模型功能模块。比如PPFNO模型，针对风阻系数预测任务，开发了完善的功能模块，包括训练/推理数据（体网格&面网格）预处理、分布式训练、离线推理、在线推理等，实现模型训练、推理容器化部署，在高速列车行业头部企业落地应用。另外，PaddleCFD 0.3.0版本提供针对多种CFD数据格式的数据解析模块，能够与多种传统CFD仿真软件无缝衔接。
- **单文件夹策略提升易用性**：PaddleCFD 0.3.0版本参考人工智能领域成功套件HuggingFace的单文件策略，将模型涉及到的模块均放在同一个文件夹下，同时避免了对深度学习框架API的过度封装，降低了用户的学习成本和后期维护成本。

## PaddleCFD 0.3.0 更新
### 新特性
- PPDeepONet模型，增加了[PirateNets网络结构](https://github.com/PaddlePaddle/PaddleCFD/pull/101/files)，提升了模型训练的稳定性和效率。
- PPFNO模型，开源了自定义算子 [fused-segment-csr](https://github.com/PaddlePaddle/PaddleCFD/pull/111/files)，基于该融合算子，模型训练推理吞吐更高，显存占用与样本大小解耦，支持千万级别网格模型训练和推理。

### Bug 修复
- PPTransformer模型，修复了[模型导入错误问题](https://github.com/PaddlePaddle/PaddleCFD/pull/106/files)。
- 通过源码安装自定义算子，修复了套件使用环境[缺少特定动态库的问题](https://github.com/PaddlePaddle/PaddleCFD/pull/111/files)。
- 更新了[PaddleCFD安装文档](https://github.com/PaddlePaddle/PaddleCFD/pull/111/files#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5)，修复了安装过程中可能遇到的错误。
- 修复了[PPFNO模型推理数据预处理模块导入多余包报错的问题](https://github.com/PaddlePaddle/PaddleCFD/pull/114)




## 贡献者名单
[guhaohao0991](https://github.com/guhaohao0991), [HydrogenSulfate](https://github.com/HydrogenSulfate), [KaiCHEN-HT](https://github.com/KaiCHEN-HT), [liaoxin2](https://github.com/liaoxin2), [lijialin03](https://github.com/lijialin03), [wangguan1995](https://github.com/wangguan1995), [XiaoguangHu01](https://github.com/XiaoguangHu01)
