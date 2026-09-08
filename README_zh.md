**中文** | [English](README.md)

# IMU EEG MobileBCI 去噪基准测试

本项目在 MobileBCI BrainVision EEG/IMU 数据集上对比 **ASR**（伪影子空间重建）与 **iCanClean 风格 CCA**（典型相关分析）回归去噪算法的性能。

## 方法说明

### ASR

ASR 实现为伪影子空间重建去噪器：

1. **校准**：对 ERP 训练数据（`ses-01`）进行带通滤波（0.5–45 Hz），分窗后选取最干净的窗口（RMS 最低的 25% 分位数）计算每个被试的参考协方差矩阵。
2. **变换**：对每个运动录音分窗，使用参考协方差进行白化。白化后协方差矩阵的特征值超过截止阈值的部分被识别为高方差（伪影）子空间，利用剩余的良好子空间进行重建。

### iCanClean

iCanClean 实现为基于 CCA 的噪声参考去噪器：

1. **分解**：对 EEG 和 IMU 运动通道进行带通滤波（0.5–15 Hz）并标准化。通过对白化后的互协方差矩阵进行 SVD 分解，计算典型相关系数和空间滤波器。
2. **回归去除**：选取典型相关系数超过阈值的分量，通过最小二乘投影估计其对宽带 EEG 的回归贡献，并从原始信号中减去。

## 会话映射

| 会话 | 含义 | 速度 |
| --- | --- | --- |
| `ses-01` | ERP 训练会话 | 训练 |
| `ses-02` | 静站 | 0.0 m/s |
| `ses-03` | 慢走 | 0.8 m/s |
| `ses-04` | 快走 | 1.6 m/s |
| `ses-05` | 慢跑 | 2.0 m/s |

## 输出内容

运行主管道会生成：

- 各速度下原始 EEG 与 IMU 的相干性曲线
- ASR 和 iCanClean 在步态频率处的功率降低曲线
- Pz 通道的 ERP 保留度
- 伪影降低与信号保留的权衡关系
- ASR 截止参数敏感性分析
- 各速度下 ERP LDA AUC 与 SSVEP CCA 准确率

运行补充实验会生成：

- 默认稳健指标（ASR vs iCanClean）及配对 Wilcoxon 检验
- ASR 参数扫描（截止值 [3, 5, 8, 10, 15, 20, 30]）
- iCanClean IMU 参考消融实验（加速度、陀螺仪、加速度+陀螺仪、幅值、增强、增强+滞后）
- iCanClean 参数扫描（相关阈值 × 最大分量数）
- 满足保留度约束的最优配置选择

## 安装

```bash
python -m pip install -r requirements.txt
```

可选的可编辑安装：

```bash
python -m pip install -e .
```

## 运行

### 主管道

完整运行（数据目录中的所有被试）：

```bash
python run_pipeline.py --data-dir Motion_eeg_data --out-dir outputs
```

`sub-01` ERP 冒烟测试：

```bash
python run_pipeline.py --quick --tasks ERP --out-dir outputs_quick
```

运行指定被试：

```bash
python run_pipeline.py --subjects sub-01 sub-02 --out-dir outputs_sub01_02
```

修改 ASR 参数：

```bash
python run_pipeline.py --asr-cutoff 8 --asr-sensitivity-cutoffs 3 5 8 10 15 20
```

### 补充实验

```bash
python run_supplementary_experiments.py --data-dir Motion_eeg_data --out-dir outputs_supplement
```

`sub-01` 冒烟测试：

```bash
python run_supplementary_experiments.py --quick --data-dir Motion_eeg_data --out-dir outputs_supplement_quick
```

## 数据来源

- 数据集仓库：https://github.com/ChenfeiJiangSICAU/MobileBCI_Data
- 数据集论文：Lee et al., *Scientific Data* 2021, DOI `10.1038/s41597-021-01094-4`

## 项目结构

```
run_pipeline.py                     # 主管道入口
run_supplementary_experiments.py     # 补充实验入口
requirements.txt
pyproject.toml
src/grf_eeg/
    __init__.py
    config.py        # 会话-速度映射、通道列表、管道配置
    dataio.py         # BrainVision 读取器、录音发现、通道选择
    sigproc.py        # Butterworth 滤波、协方差、PSD、步态频率估计
    denoise.py        # ASR 模型与 iCanClean CCA 分解
    metrics.py        # ERP 保留度、伪影降低、相干性、功率降低
    bci.py            # ERP LDA 分类器与 SSVEP CCA 准确率
    plots.py          # 主管道可视化函数
    pipeline.py       # 主基准管道
    experiments.py    # 补充参数扫描实验
```

## 许可证

MIT
