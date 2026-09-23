# RoboFollow

**面向具身智能体指令跟随的诊断性评测基准**

[English](README.md) | 简体中文

<p align="center">
  <a href="https://arxiv.org/abs/2609.25636"><img src="https://img.shields.io/badge/arXiv-2609.25636-b31b1b?style=flat" alt="arXiv 2609.25636" height="25"></a>
  <a href="https://mrc-crm.github.io/RoboFollow/"><img src="https://img.shields.io/badge/Project-Page-007ec6?style=flat" alt="Project Page" height="25"></a>
  <a href="https://huggingface.co/datasets/AutoLab-SJTU/robofollow-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffbd45?style=flat" alt="Hugging Face Dataset" height="25"></a>
</p>

RoboFollow 是基于 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 构建的仿真评测基准，用于研究具身智能体在语言指令条件下的操作能力。同一场景配置对应多种任务，要求策略依据语言选择正确的操作对象、空间关系、运动约束或逻辑分支。

本项目提供四类场景、专家示范采集、L0–L3 分层评测协议，以及分阶段的 **Intent（意图）** 与 **Execution（执行）** 评分，用于区分任务选择错误和物理执行失败。

## 主要特性

- **共享场景下的多任务设计：** 四类场景包含 75 个训练任务和 229 条训练指令变体。
- **结构化泛化评测：** 554 个评测任务覆盖布局变化、语义重组及二者的组合。
- **专家示范采集：** 支持种子筛选、规划轨迹重放、数据完整性校验和断点续采。
- **模型无关的评测接口：** 通过统一策略接口支持本地推理和 TCP 远程推理。
- **可检查的评测结果：** 保存逐条得分、阶段判定原因、可选视频及运行元数据。

## 基准设计

### 场景组成

| 场景 | 评测内容 | 训练任务 | 指令变体 | L0 | L1 | L2 | L3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Scene 1 | 空间关系与物体放置 | 16 | 48 | 16 | 8 | 16 | 76 |
| Scene 2 | 物体属性与动作选择 | 16 | 52 | 16 | 24 | 68 | 24 |
| Scene 3 | 轨迹与朝向约束 | 16 | 48 | 16 | 40 | 64 | 96 |
| Scene 4 | 条件与组合指令 | 27 | 81 | 27 | 14 | 25 | 24 |
| **合计** | | **75** | **229** | **75** | **86** | **173** | **220** |

### 评测层级

| 层级 | 评测目标 |
|---|---|
| **L0** | 分布内的指令跟随能力 |
| **L1** | 场景布局变化下的视觉指代与定位能力 |
| **L2** | 参考布局下的语义重组能力 |
| **L3** | 布局变化与语义变化共同作用下的泛化能力 |

具体评测任务定义见 [evaluation/](evaluation/)；训练任务和指令变体独立定义于 [tasks/](tasks/)。

默认协议采用 ALOHA AgileX 双臂机器人、分辨率为 320 × 240 的三路 RGB 观测和干净背景。协议参数定义于 [protocol.yml](protocol.yml)，由 [config.py](config.py) 组装为运行时配置。任务定义中仍包含各场景自身的物体位置变化。

## 安装

### 1. 准备 RoboTwin

RoboFollow 以子目录形式安装于 RoboTwin 工程。参考 RoboTwin 版本为 `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`。

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout 6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755
git submodule update --init --recursive
```

按照 [RoboTwin 环境安装与资产下载说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)配置仿真环境。参考环境使用 Linux、Python 3.10、支持 CUDA 的 NVIDIA GPU、SAPIEN、mplib 和 cuRobo。专家采集依赖 cuRobo，下面的附加依赖不能替代上游环境。

在已准备好的 RoboTwin 根目录克隆本项目：

```bash
git clone https://github.com/AutoLab-SAI-SJTU/RoboFollow.git robofollow
```

目录结构如下：

```text
RoboTwin/
├── assets/
├── envs/
├── scripts/
└── robofollow/
    ├── README.md
    ├── collect.py
    ├── evaluate.py
    └── ...
```

除非另有说明，后续命令均在 **RoboTwin 根目录**执行，并使用已配置的仿真环境。

### 2. 安装依赖与场景

```bash
conda activate RoboTwin
python -m pip install -r robofollow/requirements-core.txt
python -m pip install toppra==0.6.9
python robofollow/install_scenes.py --dry-run
python robofollow/install_scenes.py
```

场景安装器添加 `envs/scene1.py` 至 `envs/scene4.py`。已存在且内容相同的文件保持不变；存在内容冲突时停止安装，不覆盖目标文件。

在已激活的仿真环境中检查规划器：

```bash
python -c 'import curobo, torch; from curobo.wrap.reacher.motion_gen import MotionGen; print(curobo.__file__); print("CUDA:", torch.cuda.is_available())'
```

cuRobo 的 editable 安装保留源码目录路径。移动或删除源码后，`pip show` 仍可能显示已安装，但实际导入失败。应恢复匹配的源码目录，或使用 `python -m pip install -e /path/to/curobo --no-build-isolation` 重新安装目标副本。 CUDA 扩展应针对当前 Python/PyTorch/CUDA 环境编译。随后运行下文的单条示范采集，确认真实规划和渲染都可用。

### 3. 配置资产

完成上游资产下载与解压后执行：

```bash
python scripts/update_embodiment_config_path.py
python -m robofollow.assets install --dry-run
python -m robofollow.assets install
python -m robofollow.assets status
```

资产安装器应用随项目提供的两个碗视觉模型，校验文件哈希，并备份可识别的原始文件。资产说明见 [ASSETS.md](ASSETS.md)。运行环境同时需要上游物体索引、机器人资产及其他基础资源。

### 4. 检查任务注册

```bash
python -m robofollow.collect --scene all --list
python -m robofollow.evaluate --scene all --list
```

以上命令读取任务注册表，不启动仿真。使用下一节的单条示范采集命令检查规划、渲染及数据生成。

## 专家示范采集

### 单任务采集

```bash
python -m robofollow.collect \
  --scene scene1 --tasks pickup_yellow_block_1_left \
  --episodes 1 --max-seed-attempts 100 --gpu 0 \
  --output data/robofollow_smoke
```

采集器首先筛选成功的专家规划，再重放已保存的轨迹以记录观测和动作。每条 episode 通过校验后才写入最终输出目录。

### 完整训练集

```bash
python -m robofollow.collect \
  --scene all --episodes 50 --seed-start 0 \
  --max-seed-attempts 5000 --gpu 0 \
  --output data/robofollow_train
```

该命令为**每个训练任务采集 50 条示范**，总计 **3,750 条**。`--max-seed-attempts` 限制本次运行中每个任务的规划尝试次数。

中断后，在同一命令上追加 `--resume` 即可续采。采集器会检查已有 episode，并要求源码、协议及记录的依赖元数据一致；改变这些设置时应使用新的输出目录。并行采集进程不得写入同一个任务目录。

### 输出格式

```text
data/robofollow_train/<scene>/<scene>_<task>/
├── collection.json
├── seed.txt
├── seeds.json
├── _traj_data/
├── data/episode0.hdf5
├── instructions/episode0.json
├── scene_info.json
├── video/episode0.mp4
└── complete.json
```

HDF5 episode 在 `joint_action`、`observation` 和 `endpose` 下保留完整记录序列，指令保存在 `instructions/episode<N>.json`。关节向量顺序如下：

```text
[左臂关节 (6), 左夹爪 (1), 右臂关节 (6), 右夹爪 (1)]
```

`joint_action` 保存观测到的关节位置和归一化夹爪值。训练转换将第 t 帧的观测、状态与第 t+1 帧的绝对关节目标配对；采集写入时保留全部帧。图像以 JPEG 编码存储，可用 RoboTwin 的 `decode_image_bit` 解码。`save_freq` 为仿真步采样间隔。`collection.json`、`seeds.json` 和 `complete.json` 记录采集配置、已选种子和完成状态，用于数据校验与断点续采。

## 策略训练

RoboFollow 提供专家示范与基准评测，策略优化由相应模型的训练框架完成。

对于 **π0.5**，使用上游 [XPolicyLab Pi_05 集成](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Pi_05/README.md)与 [OpenPI](https://github.com/Physical-Intelligence/openpi)。模型与仿真依赖存在差异时，应使用独立的 Python 环境。

对接流程如下：

1. 采集并检查 RoboFollow 训练示范。
2. 将原始 HDF5 转换为训练框架要求的格式，例如 LeRobot。
3. 配置三路相机、14 维状态与动作、训练指令，并使用 RoboFollow 训练数据计算归一化统计量。
4. 通过上游训练器微调预训练模型。
5. 使用实现下述策略接口的适配器加载 checkpoint 并运行评测。

使用 OpenPI Aloha 数据配置时，相关字段映射如下：

| RoboFollow 输入 | OpenPI 适配器输入 |
|---|---|
| `observation.head_camera.rgb` | `images.cam_high` |
| `observation.left_camera.rgb` | `images.cam_left_wrist` |
| `observation.right_camera.rgb` | `images.cam_right_wrist` |
| `joint_action.vector` | `state` |
| 传入 `set_instruction` 的指令 | `prompt` |

训练与推理必须保持图像变换、归一化统计量、动作顺序及绝对量／增量约定一致。评测指令不得加入训练数据。

[convert.py](convert.py) 将示范转换为 **Pi HDF5 数据集**：

```bash
python -m robofollow.convert \
  --input data/robofollow_train --output data/robofollow_pi_hdf5
```

图像由 RGB 数组直接经 OpenCV 编码，使用 `cv2.imdecode` 读取时无需再交换红蓝通道。π0.5 训练使用下文的 LeRobot 转换。

已提供可运行的 LeRobot 转换、OpenPI π0.5 训练配置与策略适配器；安装、短程微调和远程评测步骤见 [PI05.md](PI05.md)。转换器支持 `joint_action/observation` 原始序列和 `state/action/vision` 配对数据：前者配对一次，后者保留已有配对。图像默认缩放为 640 × 480，可通过 `--image-width` 和 `--image-height` 设置。

## 策略接口

策略工厂返回实现 `reset`、`set_instruction` 和 `predict` 的对象。最小接口如下：

```python
class Policy:
    def reset(self):
        """Reset episode-specific state."""
        raise NotImplementedError

    def set_instruction(self, text: str):
        """Set the instruction for the next episode."""
        raise NotImplementedError

    def predict(self, observation):
        """Return a finite, nonempty float32 array of shape [T, 14]."""
        raise NotImplementedError
```

输入在 `observation["observation"]` 下提供三路 HWC 顺序的 `uint8` RGB 图像，相机名称见上表；当前 14 维状态位于 `observation["joint_action"]["vector"]`。适配器负责模型专属预处理，并在完成相应输出变换后返回绝对动作目标。

策略还可实现 `close()` 以释放资源。

### 远程推理

在**模型环境**中，从 RoboTwin 根目录启动自定义策略工厂；也可通过其他方式确保 `robofollow` 与适配器模块能够导入：

```bash
python -m robofollow.serve \
  --factory your_policy:create_policy \
  --kwargs '{"checkpoint":"/path/to/checkpoint"}' \
  --host 127.0.0.1 --port 8999
```

将 `your_policy:create_policy` 替换为实际适配器模块及工厂名称。`--kwargs` 中的参数会传递给该工厂。服务使用 RoboFollow 的长度前缀 TCP 协议，不能直接替换为任意上游 HTTP 或 WebSocket 策略服务。

单个服务依次处理客户端。并行评测应配置独立服务和端口。

## 评测

### 接口检查

内置 `HoldPolicy` 保持当前关节位置，用于检查仿真到评分的执行流程，不作为完成任务的模型基线。

```bash
python -m robofollow.evaluate \
  --scene scene1 --levels L0 --limit 1 --rounds 1 \
  --max-steps 1 --actions-per-step 5 \
  --policy-factory robofollow.policies:HoldPolicy \
  --output results/robofollow_hold --video
```

### 评测训练后的策略

策略服务启动后，在**仿真环境**中执行：

```bash
python -m robofollow.evaluate \
  --scene scene1 --levels L0 --limit 1 --rounds 1 \
  --remote --host 127.0.0.1 --port 8999 \
  --runtime fixed --gpu 0 \
  --output results/robofollow_smoke --video
```

检查初步结果后运行完整协议：

```bash
python -m robofollow.evaluate \
  --scene all --levels L0,L1,L2,L3 --rounds 10 --base-seed 42 \
  --max-steps 10 --actions-per-step 50 --runtime fixed --gpu 0 \
  --remote --host 127.0.0.1 --port 8999 \
  --output results/robofollow_full
```

完整协议包含 **5,540 次评测**。追加 `--video` 可保存评测视频。

| 参数 | 含义 |
|---|---|
| `--tasks` | 逗号分隔的评测任务 ID，以 `--list` 输出为准 |
| `--limit` | **每个场景**最多选择的任务数 |
| `--max-steps` | 每条 episode 最多调用策略的次数 |
| `--actions-per-step` | 每次策略调用后最多执行的动作目标数 |
| `--runtime fixed` | 唯一支持的评测配置，省略时也使用该配置 |
| `--video-fps` | 仅影响视频播放速度 |
| `--dry-run` | 显示任务选择与评测次数，不启动仿真 |

默认动作预算上限为 10 × 50 = 500 个目标，并非 500 个物理仿真步。`--direct-step` 启用另一种执行方式，其结果应与默认 TOPP 执行方式分别报告。

每次评测需要新的或空的输出目录，当前不支持评测 `--resume`。

## 结果与可复现性

评测输出如下：

| 文件或目录 | 内容 |
|---|---|
| `run.json` | 运行状态、参数、任务选择、版本、源码指纹、资产元数据和评测次数 |
| `results.json` | 逐条得分、指令、种子、阶段判定原因及可选视频路径 |
| `summary.json` | 按场景和评测层级汇总的指标 |
| `videos/` | 使用 `--video` 时生成的视频 |

查看运行信息与汇总：

```bash
python -m json.tool results/robofollow_full/run.json
python -m json.tool results/robofollow_full/summary.json
```

评测完成时，`status` 为 `complete`，且 `completed_trials == expected_trials`。

| 汇总指标 | 定义 |
|---|---|
| `mean_intent_score` | 分阶段指令理解与指代得分的均值 |
| `mean_exec_score` | 分阶段物理执行得分的均值 |
| `intent_full_rate` | 意图得分达到对应满分 99% 及以上的评测比例 |
| `exec_full_rate` | 执行得分达到对应满分 99% 及以上的评测比例 |
| `completion_rate` / CR | 任务完成率，在每个场景/层级内按任务等权平均 |
| `task_macro_completion_rate` | 与 `completion_rate` 相同的任务宏平均完成率 |
| `episode_micro_completion_rate` | 按完成规则通过的回合数占比 |

CR 按以下规则判断每个回合是否完成，得分采用 0–1 标度：

1. **pickup 类动作**：`pickup`、`sequence_pickups` 以及名称以 `pickup` 开头的动作，要求最终 IS、ES 总分均至少为 0.8。这里使用已有加减分处理后的总分，不先扣除 finishing 阶段得分。
2. **有显式信号的其余动作**：优先读取 `final_target_ok`，否则读取 `retry_bonus.final_target_ok`。选中信号的布尔值直接决定是否完成；明确为 `false` 就判失败，只有信号缺失或为 null 才允许回退。
3. **两项信号均不可用的其余动作**：检查 `stage3_orientation` 的执行分是否大于零；没有该阶段时，按 `stage2_stack`、`stage2_unstack_slot`、`stage2_place_bowl`、`stage2_push` 的顺序，检查第一个存在阶段的执行分是否大于零。该阶段执行分为零或没有匹配阶段，均判失败。阶段得分仍保留原有的前置阶段门控。

这些信号对应的含义和时间范围并不相同：

| 任务类型 | CR 使用的信号 |
|---|---|
| Scene 1 关系放置 | `final_target_ok`：最后 10 个记录帧持续满足指定空间关系和桌面高度条件，不重新检查初次抓取或首次松手 |
| Scene 2 堆叠 / 放入碗 | `retry_bonus.final_target_ok`：最终帧的堆叠或入碗几何条件 |
| Scene 2 推动 | `retry_bonus.final_target_ok`：任意记录时刻，源物体位移在朝向目标初始位置方向上的投影达到 0.005 m；第二阶段推动执行评分另有 0.02 m 的位移阈值 |
| Scene 3 / Scene 4 非 pickup 动作 | 上述对应执行阶段的回退信号 |

CR 不要求每个任务阶段满分，也没有统一的“曾经完成就永久算成功”规则。后续动作可能破坏最终布局，从而改变 CR。在 Scenes 1、2、4 中，`stage3_finish` 在 IS、ES 中各占 0.2；非 pickup 动作的 CR 不单独要求该阶段通过，而 pickup 类动作直接使用包含 finishing 得分和已有加减分的最终总分。

例如，Scene 1 关系放置回合可以各执行阶段均为零分，但最后布局检查通过，获得已有的 +0.3 ES 补分：此时 ES 只有 0.3，CR 仍计为完成。反过来，抓取和放置阶段拿到 0.8 ES 后，如果后续动作破坏了目标关系，CR 仍可判失败。IS/ES 及其 full rate 继续使用完整阶段和已有加减分规则。Scene 4 的 `conditional_full_success` 是全部阶段及 IS/ES 总分满分的独立诊断指标。

CR 采用任务宏平均：先计算每个任务的成功回合比例，再对各任务等权平均。同时保存按回合统计的微平均；各任务回合数相同时，两者一致。IS/ES 的均值仍按回合数加权。

`results.json` 中的 `phases` 保存 `intent_reason`、`exec_reason`、阶段得分及满分值；若记录了视频，`video` 字段为相对于本次输出目录的路径。

## 项目结构

```text
robofollow/
├── tasks/                 # 训练任务与指令变体
├── scene_sources/         # 专家场景实现
├── evaluation/            # 评测任务与分阶段评分
├── asset_overrides/       # 碗视觉模型与哈希清单
├── collect.py             # 专家示范采集
├── episodes.py            # Episode 校验与写入
├── routes.py              # 专家路线检查
├── evaluate.py            # 评测入口
├── simulation.py          # 评测环境构建
├── policies.py            # 策略接口校验
├── remote_policy.py       # TCP 推理客户端
├── serve.py               # 策略服务
├── convert.py             # Pi HDF5 转换
├── install_scenes.py       # 场景安装
├── assets.py              # 资产安装与恢复
├── geometry.py            # 场景几何工具
├── registry.py            # 任务注册表访问
├── config.py              # 运行配置
├── protocol.yml           # 基准协议
└── runinfo.py              # 运行元数据
```

## 致谢与第三方声明

RoboFollow 基于 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 构建。策略训练可使用 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 和 [OpenPI](https://github.com/Physical-Intelligence/openpi)。

上游 RoboTwin 的 MIT 许可声明保留于 [LICENSE.RoboTwin](LICENSE.RoboTwin)。资产配置说明见 [ASSETS.md](ASSETS.md)；模型权重及其他第三方资源通过各自提供方获取。

## 引用

```bibtex
@misc{guo2026robofollow,
  title = {RoboFollow: Unveiling the Instruction Following Mirage in Embodied Agents},
  author = {Guo, Chang and Xie, Yukun and Tan, Bohan and Chang, Zheng and Yin, Zhaokai and Ma, Qianli and Wang, Yingqiao and Liang, Chao and Zhang, Zhipeng},
  year = {2026},
  eprint = {2609.25636},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.25636}
}
```
