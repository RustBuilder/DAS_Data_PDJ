# BlockIQ DAS 4米标距离线处理

本项目处理 `113`、`619`、`985` 三组30秒 BlockIQ CSV。主DAS信号固定采用10点、4米标距的圆周差分相位，核心算法和统一入口全部位于 `scripts/`。

当前为第一版 `v0.1.0`。

## 项目结构

```text
DAS_PY新建文件夹/
├── .env
├── README.md
├── requirements.txt
├── scripts/
│   ├── __init__.py
│   ├── pipeline.py       核心读取、恢复、预处理、频谱和导出算法
│   └── process_das.py    唯一命令行入口
├── tests/
└── results/
```

项目不再包含2米标距比较代码、2米结果目录或多个数据集入口脚本。

## 运行

```powershell
git clone https://github.com/RustBuilder/DAS_Data_PDJ.git
cd DAS_Data_PDJ
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，将三个 `DATASET_*_DIR` 和 `OUTPUT_ROOT` 改为本机路径。`.env`、原始数据、HDF5和 `results/` 均被Git忽略，不会上传到仓库。

处理全部三组：

```powershell
python scripts\process_das.py
```

只处理指定数据：

```powershell
python scripts\process_das.py 113
python scripts\process_das.py 619
python scripts\process_das.py 985
```

## 输入语义

- CSV来自 `PD_FDM_IQ_AMP_PHASE` 模式下的 `BlockIQ.amp(0)` 和 `BlockIQ.phase(0)`。
- ADC0/ADC1已由SDK做偏振合成和IQ解调，CSV不是原始ADC或独立I/Q。
- `amp_loc_*` 是SDK原生未标定幅值。
- `phase_loc_*` 是SDK解调后的单点相位，数据特征表明单位为弧度。
- 每5000行是一个0.5秒 BlockIQ 块，单点相位在块边界重新初始化。

## 固定4米标距流程

采样间距为 `0.4 m/点`，算法固定使用10点标距：

```text
10点 × 0.4 m/点 = 4 m
```

处理流程：

```text
CSV和capture_info.json校验
  -> 分5000行读取
  -> 原始SDK幅值和单点相位保存
  -> 单点相位块间连续对齐，仅作诊断
  -> 右端相位减左端相位
  -> 圆周约束到[-π, π)
  -> 每块内沿时间解缠
  -> 差分相位块间连续对齐
  -> 线性去趋势
  -> 空间中值共模投影去除
  -> 5–2000 Hz四阶Butterworth双向零相位滤波
  -> Welch功率谱、RMS和频带功率
  -> 自动选择代表4米标距通道
  -> HDF5、CSV和图片分类导出
```

标距点数在 `scripts/pipeline.py` 中固定为 `10`，不再从 `.env` 切换到2米。

## 输出

每组结果位于 `results/<数据编号>/`：

```text
00_original/     原始SDK幅值和单点相位图
01_recovered/    单点诊断相位、4米差分相位和预处理图
02_comparison/   恢复前后时域与频谱对比
03_data/         HDF5、代表标距CSV、通道指标和JSON报告
```

核心HDF5数据集：

| 数据集 | 含义 | 单位 |
|---|---|---|
| `amplitude/raw_sdk_native` | SDK原始幅值 | SDK原生 |
| `amplitude/preprocessed_relative` | 预处理相对幅值 | 相对量 |
| `phase/raw_block_relative_rad` | 每0.5秒复位的SDK单点相位 | rad |
| `phase/aligned_rad` | 块间对齐单点相位，仅作诊断 | rad |
| `phase/differential_aligned_rad` | 4米圆周差分、解缠和块间对齐相位 | rad |
| `phase/differential_preprocessed_rad` | 最终主DAS差分相位 | rad |
| `phase/differential_block_offsets_rad` | 差分相位块间偏置 | rad |

`channel_summary.csv` 保存101个4米标距通道的左右端索引、中心距离、端点幅值质量、RMS、主频和频带功率。`representative_timeseries.csv` 保存代表标距的完整10 kHz时序。

## 可调参数

`.env` 保留以下处理参数：

- `BANDPASS_LOW_HZ`、`BANDPASS_HIGH_HZ`
- `FILTER_ORDER`
- `COMMON_MODE_REMOVAL`
- `ALIGN_EDGE_SAMPLES`
- `PHASE_BLOCK_UNWRAP`
- `CSV_CHUNK_ROWS`
- `FILTER_CHANNEL_BATCH`
- `PLOT_MAX_TIME_POINTS`
- `HDF5_COMPRESSION_LEVEL`

4米标距不在 `.env` 中修改。

## 验证

```powershell
python -m pytest -q
```

当前输出仍是未标定差分相位，没有换算为应变或声压，因为采集元数据缺少工作波长、有效折射率和光弹系数。
