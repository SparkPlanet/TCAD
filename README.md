# TCAD 良率估计实验

本仓库保存 18、243 和 1203 维 SRAM 读扰动用例的统一算法和实测数据。HOIST 对候选工艺样本排序，Top-K 梯度确定初始采样轴。真实 SPICE 反馈修正排序并拟合轴向可逆样条。冻结 proposal 后，沿平行线探测局部失效边界，并用重要性采样修正边界积分。

## 三例的沿线估计

每例使用 8,192 个候选点、Top-K = 32、12 轮反馈、每轮 32 次 SPICE。每条线用两次 SPICE 探测边界，再用两次 SPICE 修正积分。反馈种子为 46，最终估计种子为 49。标准误按独立对偶线对计算。总仿真数计入已有训练标签。

| 维度 | 训练标签 | 反馈 | 采样线数 | 总 SPICE | 失效概率 | FOM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 1,024 | 384 | 320 | 2,688 | 5.19436e-5 | 0.01357 |
| 243 | 1,024 | 384 | 320 | 2,688 | 5.76205e-5 | 0.01167 |
| 1203 | 2,048 | 384 | 512 | 4,480 | 5.20249e-5 | 0.05696 |

## 安装

实验使用 Linux、Python 3.9、ngspice 42 和 PyTorch 2.5.1。请安装 ngspice 并保证其可执行文件在 PATH 中。在仓库根目录执行以下命令。

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NGSPICE_NUM_THREADS=1
```

运行过程使用 CPU。记录反馈样本的严格重放会逐点核对浮点结果，建议使用上述依赖版本与线程设置。

## 用保存的 SPICE 数据复算

```bash
python experiments/scripts/report_results.py
```

脚本从逐点测量和权重重算直接 IS、分层控制变量 IS 和沿线估计的失效概率、标准误和总调用数。论文中的直接 MC 值是有限样本参考。补充更精确的参考后，可以复制并修改 `data/references.json`，执行 `python experiments/scripts/report_results.py --references 新文件.json`。参考必须对应相同阈值和工艺分布。HOIST 的点估计取自论文舍入值。

## 使用真实 SPICE 重跑

使用保存的反馈记录重建冻结 proposal。以下命令只重新调用最终阶段的 SPICE。输出目录由脚本创建，每次运行使用新目录。

```bash
for case in case18 case243 case1203; do
    lines=320
    if [ "$case" = case1203 ]; then lines=512; fi
    python experiments/scripts/check_axial_boundary.py \
        "results/conditional_triplet_46_47/${case}_pilot" \
        "runs/${case}_line49" --lines "$lines" --workers 4 --seed 49
done
```

从头生成 384 个真实反馈标签时，使用对应 case 名执行以下命令。训练数据和训练好的 HOIST 权重随仓库提供。命令结束时会打印本次反馈输出目录，该目录可用作上述沿线脚本的第一个参数。

```bash
python src/feedback_flow.py --case case18 --axis gradient --seed 46 \
    --rounds 12 --batch 32 --pool 8192 --k 32 --pilot-fit --conditional-fit
```

普通独立 IS 使用 `validate_frozen_feedback.py`，对应参数为 `--samples 1792 --seed 47`。分层 IS 使用同一入口，参数为 `--samples 1280 --seed 48 --stratified-control`。两个入口的位置参数依次为反馈目录和新输出目录。

## 文件位置

| 位置 | 内容 |
| --- | --- |
| `src/feedback_flow.py` | Top-K、反馈残差、probit 和可逆样条 |
| `experiments/scripts/` | 独立 IS、沿线估计和结果复算 |
| `data/legacy_manifest.json` | 三例的网表、训练数据、权重和阈值映射 |
| `legacy/hoist/` | HOIST 代理模型、SPICE 接口、训练数据、权重和三套电路文件 |
| `results/conditional_triplet_46_47/` | 反馈批次和冻结 proposal 后的直接 IS 对照 |
| `results/stratified_control_46_48/` | 分层 IS 对照、三个 `line_check_*_seed49` 目录的沿线原始数据和汇总 |
