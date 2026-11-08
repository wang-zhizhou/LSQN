# Linux 运行说明

所有命令均在解压后的 `LSQN` 目录执行。程序使用 CPU，支持 Linux 和 macOS，不需要 CUDA 或显卡。

本公开版可运行 LSQN 和七个基线，共八种方法。AA-I-S-5 和 iGMRES-PI 的实现不随包分发；原实验结果、参数和绘图数据仍保留，见 [基线记录](docs/BASELINE_RECORDS.md)。

## 安装并检查

```bash
conda create -n lsqn-repro python=3.11 pip -y
conda run -n lsqn-repro python -m pip install -e .
conda run --no-capture-output -n lsqn-repro python scripts/prepare_backend.py
conda run --no-capture-output -n lsqn-repro python scripts/run_suite.py smoke --methods all
```

已有 `lsqn-repro` 环境时，无需重复创建。安装需要联网，准备步骤需要 C 编译器。若明确提示找不到编译器，可在独立环境内安装后重试：

```bash
conda install -n lsqn-repro -c conda-forge gcc_linux-64 -y
conda run --no-capture-output -n lsqn-repro python scripts/prepare_backend.py
```

小规模检查运行八种方法、六个小模型，共 48 次求解，并独立核验结果。若失败，请保留终端输出和生成的结果文件。

## 运行公开版实验

```bash
conda run --no-capture-output -n lsqn-repro python scripts/run_paper.py
```

共 828 次求解：八种方法的主实验 768 次、LSQN 官方公共环境实验 24 次、LSQN synthetic scaling 实验 36 次。保留原有参数、初值和停止条件，默认单线程、串行计时，单次软时限为 30/60 秒。超时和失败也会保存。

每次运行创建独立的结果目录和 ZIP，记录机器配置、依赖版本、输入哈希、残差、迭代与查询次数、耗时和汇总，不覆盖原实验数据。中途中断后保留的部分结果不代表完成整套实验。

`--methods all` 指本包可运行的八种方法。新运行不会生成论文的完整十方法排名，也不会用历史记录补齐两种未分发基线的结果。

## 分组运行

```bash
conda run --no-capture-output -n lsqn-repro python scripts/run_suite.py main --methods all
conda run --no-capture-output -n lsqn-repro python scripts/run_suite.py public
conda run --no-capture-output -n lsqn-repro python scripts/run_suite.py scaling
```

只运行 LSQN 的三套实验时，使用 `scripts/run_suite.py all`，共 156 次求解。使用原残差图对应的 LSQN 实现进行主比较时，运行：

```bash
conda run --no-capture-output -n lsqn-repro python scripts/run_suite.py main --methods all --implementation curves
```

默认采用论文最终计时表对应的实现；`curves` 使用原四幅残差图对应的 LSQN 实现。两种运行记录分开保存，不混合挑选较快的结果。机器、数值库和浮点计算差异可能影响耗时、迭代与查询次数。

## 从原记录重画图表

```bash
conda run -n lsqn-repro python -m pip install -e '.[plots]'
conda run -n lsqn-repro python scripts/plot_benchmarks.py
conda run -n lsqn-repro python scripts/plot_runtime_tables.py
```

绘图还需要 Poppler 的 `pdftoppm`。这些命令读取 `benchmarks/` 中的原记录，重画包含全部十种方法的跨平台图表，不运行 AA-I-S-5 或 iGMRES-PI。原完整 Linux 批次包含 1,020 次求解；这一历史记录与公开版新运行的 828 次求解分别保留。

首页 macOS 曲线来自最终计时配置的历史记录，可能与论文较早的残差图轨迹不同；具体来源见 [比较说明](benchmarks/README.md)。历史检查与公开版检查的范围见 [验证记录说明](validation/README.md)。
