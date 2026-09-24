# ABAR：语音分离改进模块

[English](README.md) | [方法与接口](docs/METHODS.md) | [接入说明](docs/INTEGRATION.md) | [发布前检查](docs/RELEASE_CHECKLIST.md)

本仓库只整理本文在语音分离中使用的 **Band Allocation（BA）** 和 **Band Recalibration（BR）**，便于独立阅读、检查和接入已有模型。

**不包含 TIGER 模型代码、频带编码器、分离器、掩码估计头、完整训练脚本、预训练权重、数据集和音乐分离代码。它是改进模块包，不是完整实验复现仓库。**

## 包含哪些内容

| 文件 | 说明 |
| --- | --- |
| `abar_speech/allocation.py` | 指数带宽分配、低频单 bin 与高频指数分配组合，以及频带索引区间 |
| `abar_speech/recalibration.py` | 时间平均池化、共享 SiLU MLP、2 倍 Sigmoid 和乘性重标定 |
| `configs/speech_ba_br.json` | 论文语音部分的 BA、BR 参数，不含机器路径 |
| `examples/inspect_bands.py` | 打印频带宽度和覆盖范围，无须音频数据 |
| `examples/demo_recalibration.py` | 使用随机特征演示 BR 前向、反向传播，不是分离实验 |
| `tests/` | 频带覆盖、整数修正、门控形状、梯度等单元测试 |
| `docs/` | 算法说明、接入边界、来源说明与发布检查 |
| `.github/workflows/tests.yml` | 上传 GitHub 后可运行的 CPU 测试流程 |
| `CITATION.cff` | 软件引用信息；不虚构已录用论文、DOI 或仓库地址 |

开发时的 HybridBand 对应 BA，BandGate 对应 BR。指数规则是 BA 内部的构建步骤，不是额外提出的第三个模块。包内不含早期 Top-K 或其他主干改动。

## 安装与运行

在本仓库根目录执行，使用 Python 3.10 或更新版本：

```bash
# 只使用 BA，只需要 NumPy。
python -m pip install -e .

# 使用 BA 和 BR；如需特定 CUDA 版本，请先安装对应的 PyTorch。
python -m pip install -e ".[br]"

python examples/inspect_bands.py
python examples/demo_recalibration.py
python -m unittest discover -s tests -v
```

## 论文语音配置

| 项目 | 设置 |
| --- | --- |
| 采样率 | 16 kHz |
| FFT 点数 / 帧移 | 640 / 160 |
| 单边频率 bin | 321 |
| 总频带数 | 67 |
| 低频单 bin 频带 | 60 |
| 高频频带 | 7 |
| 指数增长参数 | 2.0 |
| 编码特征维度 / BR 隐藏维度 | 128 / 64 |

默认频带宽度为：

```python
[1] * 60 + [11, 15, 21, 30, 43, 59, 82]
```

BR 接收 `[批大小, 频带数, 特征维度, 时间帧数]`，输出形状不变。设置 `return_gate=True` 时，同时返回 `[批大小, 频带数]` 的权重。默认配置下 BR 有 8,321 个参数。

## BR 的定义

BR 对每个频带的编码特征沿时间维求平均，通过共享的两层全连接网络生成标量，再经缩放 Sigmoid 得到重标定权重：

```text
p_k = mean_t(z_k[:, t])
u_k = W2 * SiLU(W1 * p_k + b1) + b2
g_k = 2 * sigmoid(u_k)
z_rec_k = g_k * z_k
```

权重的数学范围为 **(0, 2)**。小于 1 时衰减对应频带的编码特征，大于 1 时增强对应特征；权重沿特征维与时间维广播，所有频带均保留。这不是跨频带 Softmax，也不使用 Top-K 筛选或新增残差相加。权重表示学习到的特征缩放，不直接等同于物理意义上的频率重要性。

初始化时，两层偏置均设为 0，第一层权重保留 PyTorch 默认初始化，第二层权重采用均值 0、标准差 `1e-3` 的正态分布。初始 logit 较小时，权重接近 1，从而使重标定接近恒等映射；它不是对任意输入都严格成立的恒等映射。

BA 的边界在训练前确定，不随输入变化；BR 根据输入特征生成连续权重，但不会删除频带。接入时，BA 作用于原有频带组织，BR 放在频带编码之后、分离器之前。原始复数子频带谱仍用于后续掩码相乘，不要用 BR 输出代替原始频谱。

## 重新训练与结果对应

更新 BR 后，相关实验已重新训练。本文档和代码统一使用 **BR** 名称，其含义为上述 SiLU、缩放 Sigmoid 和接近恒等映射的初始化；BA 规则保持不变。名称不变不代表可以直接沿用旧结果，当前包不附新的分离指标或完整模型权重。

旧实现采用 ReLU 和未缩放的 Sigmoid。两种实现的参数名、形状相同，旧权重可能通过严格加载检查，但激活函数和输出缩放不同，不能因此认为计算等价。应使用与重新训练实验匹配的实现和权重。训练设置与结果记录要求见 [实验说明](docs/EXPERIMENTS.md)。

## 发布状态

项目仓库为 [Zlw123-456/SS](https://github.com/Zlw123-456/SS)。当前更新包未新增或变更开源协议；协议仍由作者和老师确认，不默认授予 MIT 或其他许可。合并到现有仓库时，应保留已获批准的作者信息、协议和其他无关修改。

TIGER 仅作为外部参考：[官方仓库](https://github.com/JusperLee/TIGER)。读者需要自行获取适用的主干、训练和评价实现。现有示例不能直接复现论文中的分离指标。
