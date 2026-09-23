# RoboFollow

**A diagnostic benchmark for instruction following in embodied agents**

English | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="https://arxiv.org/abs/2609.25636"><img src="https://img.shields.io/badge/arXiv-2609.25636-b31b1b?style=flat" alt="arXiv 2609.25636" height="25"></a>
  <a href="https://mrc-crm.github.io/RoboFollow/"><img src="https://img.shields.io/badge/Project-Page-007ec6?style=flat" alt="Project Page" height="25"></a>
  <a href="https://huggingface.co/datasets/AutoLab-SJTU/robofollow-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffbd45?style=flat" alt="Hugging Face Dataset" height="25"></a>
</p>

RoboFollow is a simulation benchmark built on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) for evaluating instruction-conditioned manipulation by embodied agents. Multiple tasks share the same scene configuration, requiring policies to use language to select the appropriate object, spatial relation, motion constraint, or logical branch.

The benchmark provides four scene families, expert demonstration collection, an L0–L3 evaluation protocol, and stage-wise **Intent** and **Execution** scores. These complementary measures help distinguish incorrect task selection from incomplete physical execution.

## Features

- **Shared-scene task diversity:** 75 training tasks with 229 instruction variants across four scene families.
- **Structured generalization evaluation:** 554 evaluation tasks covering layout changes, semantic recombination, and their combination.
- **Expert demonstration collection:** seed selection, planned-trajectory replay, episode validation, and resumable collection.
- **Model-independent evaluation:** a common policy interface with local and TCP-based inference.
- **Inspectable results:** per-episode scores, stage-level reasons, optional videos, and run metadata.

## Benchmark

### Scene families

| Scene | Focus | Training tasks | Instruction variants | L0 | L1 | L2 | L3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Scene 1 | Spatial relations and object placement | 16 | 48 | 16 | 8 | 16 | 76 |
| Scene 2 | Object attributes and action selection | 16 | 52 | 16 | 24 | 68 | 24 |
| Scene 3 | Trajectory and orientation constraints | 16 | 48 | 16 | 40 | 64 | 96 |
| Scene 4 | Conditional and compositional instructions | 27 | 81 | 27 | 14 | 25 | 24 |
| **Total** | | **75** | **229** | **75** | **86** | **173** | **220** |

### Evaluation levels

| Level | Evaluation objective |
|---|---|
| **L0** | In-distribution instruction following |
| **L1** | Visual grounding under changed scene layouts |
| **L2** | Semantic recombination under the reference layouts |
| **L3** | Combined layout and semantic changes |

Exact task specifications are defined in [evaluation/](evaluation/); training tasks and instruction variants are defined separately in [tasks/](tasks/).

The default protocol uses an ALOHA AgileX dual-arm robot, three RGB views at 320 × 240 resolution, and clean backgrounds. Protocol settings are defined in [protocol.yml](protocol.yml) and assembled by [config.py](config.py). Scene-specific object placement variation remains part of the task definitions.

## Installation

### 1. Prepare RoboTwin

RoboFollow is installed as a directory inside a RoboTwin checkout. The reference RoboTwin revision is `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout 6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755
git submodule update --init --recursive
```

Follow the [RoboTwin installation and asset download instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for the simulation environment. The reference setup uses Linux, Python 3.10, a CUDA-capable NVIDIA GPU, SAPIEN, mplib, and cuRobo. Expert collection requires cuRobo; the additional dependencies below do not replace the upstream environment.

Clone this project into the prepared RoboTwin checkout:

```bash
git clone https://github.com/AutoLab-SAI-SJTU/RoboFollow.git robofollow
```

The resulting directory layout is:

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

All subsequent commands are run from the **RoboTwin root directory**, with its simulation environment activated, unless stated otherwise.

### 2. Install dependencies and scenes

```bash
conda activate RoboTwin
python -m pip install -r robofollow/requirements-core.txt
python -m pip install toppra==0.6.9
python robofollow/install_scenes.py --dry-run
python robofollow/install_scenes.py
```

The scene installer adds `envs/scene1.py` through `envs/scene4.py`. Identical files are retained; conflicting files cause installation to stop without overwriting them.

Check the planner in the activated simulator environment:

```bash
python -c 'import curobo, torch; from curobo.wrap.reacher.motion_gen import MotionGen; print(curobo.__file__); print("CUDA:", torch.cuda.is_available())'
```

An editable cuRobo installation retains the source directory path. If the checkout was moved or deleted, `pip show` alone can still report an installed package while the import fails. Restore the matching source directory, or install the intended checkout again with `python -m pip install -e /path/to/curobo --no-build-isolation`. Build CUDA extensions for the current Python/PyTorch/CUDA environment. Then run the one-episode collection below to verify actual planning and rendering.

### 3. Configure assets

After downloading and extracting the upstream assets:

```bash
python scripts/update_embodiment_config_path.py
python -m robofollow.assets install --dry-run
python -m robofollow.assets install
python -m robofollow.assets status
```

The asset installer applies two bundled bowl visual overrides, verifies file hashes, and preserves recognized originals as backups. See [ASSETS.md](ASSETS.md) for asset details. The upstream object indices, robot assets, and other base resources are also required.

### 4. Verify task registration

```bash
python -m robofollow.collect --scene all --list
python -m robofollow.evaluate --scene all --list
```

These commands inspect the task registries without starting the simulator. Use the single-episode collection example below to verify planning, rendering, and data generation.

## Demonstration Collection

### Single-task collection

```bash
python -m robofollow.collect \
  --scene scene1 --tasks pickup_yellow_block_1_left \
  --episodes 1 --max-seed-attempts 100 --gpu 0 \
  --output data/robofollow_smoke
```

The collector first identifies successful expert plans, then replays saved trajectories to record observations and actions. Episodes are validated before publication to the output directory.

### Full training set

```bash
python -m robofollow.collect \
  --scene all --episodes 50 --seed-start 0 \
  --max-seed-attempts 5000 --gpu 0 \
  --output data/robofollow_train
```

This requests **50 demonstrations per training task**, or **3,750 demonstrations** in total. `--max-seed-attempts` limits planning attempts per task for the current invocation.

To resume collection, repeat the command with `--resume`. The collector checks existing episodes and requires matching source, protocol, and recorded dependency metadata. Use a new output directory when changing these settings. Concurrent workers must not write to the same task directory.

### Output format

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

Each HDF5 episode preserves the full recorded sequence under `joint_action`, `observation`, and `endpose`. Instructions are stored in `instructions/episode<N>.json`. The joint vector uses this ordering:

```text
[left arm joints (6), left gripper (1), right arm joints (6), right gripper (1)]
```

`joint_action` stores observed joint positions and normalized gripper values. Training conversion pairs observation and state at frame t with the absolute joint target at frame t+1. The writer retains all frames. Images are stored as JPEG buffers and decoded with RoboTwin’s `decode_image_bit`. `save_freq` is the interval in simulation steps. `collection.json`, `seeds.json`, and `complete.json` record collection settings, accepted seeds, and completion status for validation and resuming collection.

## Policy Training

RoboFollow provides demonstration data and benchmark evaluation. Policy optimization is delegated to the model's training framework.

For **π0.5**, use the upstream [XPolicyLab Pi_05 integration](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Pi_05/README.md) and [OpenPI](https://github.com/Physical-Intelligence/openpi). Keep the model environment separate from the simulation environment where necessary.

The integration workflow is:

1. Collect and inspect RoboFollow training demonstrations.
2. Convert the original HDF5 data into the format required by the trainer, such as LeRobot.
3. Configure the three camera views, 14-dimensional state/action layout, and training instructions; compute normalization statistics from the RoboFollow training data.
4. Fine-tune the pretrained model using the upstream trainer.
5. Load the checkpoint through a policy adapter implementing the interface below.

For an OpenPI Aloha data configuration, the relevant field mapping is:

| RoboFollow input | OpenPI adapter input |
|---|---|
| `observation.head_camera.rgb` | `images.cam_high` |
| `observation.left_camera.rgb` | `images.cam_left_wrist` |
| `observation.right_camera.rgb` | `images.cam_right_wrist` |
| `joint_action.vector` | `state` |
| Instruction passed to `set_instruction` | `prompt` |

Use the same image transforms, normalization statistics, action ordering, and absolute/delta conventions for training and inference. Evaluation instructions must remain excluded from training data.

[convert.py](convert.py) converts demonstrations to **Pi HDF5 datasets**:

```bash
python -m robofollow.convert \
  --input data/robofollow_train --output data/robofollow_pi_hdf5
```

Images are encoded directly from RGB arrays with OpenCV. Read them with `cv2.imdecode` without swapping red and blue channels. For π0.5 training, use the LeRobot conversion below.

Runnable LeRobot conversion, an OpenPI π0.5 training configuration, and a policy adapter are provided; see [PI05.md](PI05.md) for installation, short training, and remote evaluation. The converters accept both `joint_action/observation` raw sequences and `state/action/vision` paired data. Raw sequences are paired once; existing pairs are retained. Images are resized to 640 × 480 by default; `--image-width` and `--image-height` select another size.

## Policy Interface

A policy factory returns an object implementing `reset`, `set_instruction`, and `predict`. A minimal interface is:

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

The observation contains three `uint8` RGB images in HWC order under `observation["observation"]`, using the camera names listed above, and the current 14-dimensional state at `observation["joint_action"]["vector"]`. The adapter handles model-specific preprocessing and returns absolute action targets after the appropriate output transforms.

An optional `close()` method can release policy resources.

### Remote inference

In the **model environment**, run a user-provided factory from the RoboTwin root, or ensure that both `robofollow` and the adapter are importable:

```bash
python -m robofollow.serve \
  --factory your_policy:create_policy \
  --kwargs '{"checkpoint":"/path/to/checkpoint"}' \
  --host 127.0.0.1 --port 8999
```

Replace `your_policy:create_policy` with the adapter's actual module and factory. The `--kwargs` object is passed to that factory. The service uses RoboFollow's length-prefixed TCP protocol; an arbitrary upstream HTTP or WebSocket policy endpoint is not directly interchangeable.

A server processes one client at a time. Use separate servers and ports for concurrent evaluation workers.

## Evaluation

### Interface smoke test

The included `HoldPolicy` maintains the current joint positions and checks the simulator-to-evaluator pipeline. It is not a task-solving baseline.

```bash
python -m robofollow.evaluate \
  --scene scene1 --levels L0 --limit 1 --rounds 1 \
  --max-steps 1 --actions-per-step 5 \
  --policy-factory robofollow.policies:HoldPolicy \
  --output results/robofollow_hold --video
```

### Evaluate a trained policy

With the policy server running, execute the following in the **simulation environment**:

```bash
python -m robofollow.evaluate \
  --scene scene1 --levels L0 --limit 1 --rounds 1 \
  --remote --host 127.0.0.1 --port 8999 \
  --runtime fixed --gpu 0 \
  --output results/robofollow_smoke --video
```

Run the full protocol after inspecting the initial result:

```bash
python -m robofollow.evaluate \
  --scene all --levels L0,L1,L2,L3 --rounds 10 --base-seed 42 \
  --max-steps 10 --actions-per-step 50 --runtime fixed --gpu 0 \
  --remote --host 127.0.0.1 --port 8999 \
  --output results/robofollow_full
```

The full protocol contains **5,540 trials**. Add `--video` to retain evaluation videos.

| Option | Meaning |
|---|---|
| `--tasks` | Comma-separated evaluation task IDs; use the IDs from `--list` |
| `--limit` | Maximum number of selected tasks **per scene** |
| `--max-steps` | Maximum policy calls per episode |
| `--actions-per-step` | Maximum action targets executed per policy call |
| `--runtime fixed` | The only supported evaluation runtime; also used when omitted |
| `--video-fps` | Video playback rate only |
| `--dry-run` | Display selected tasks and trial count without simulation |

The default action budget is at most 10 × 50 = 500 targets, not 500 physics steps. `--direct-step` selects an alternative execution mode; its results should be reported separately from the default TOPP-based execution.

Each evaluation requires a new or empty output directory. Evaluation does not currently support `--resume`.

## Results and Reproducibility

Evaluation produces:

| Output | Contents |
|---|---|
| `run.json` | Run status, arguments, task selection, versions, source fingerprint, asset metadata, and trial counts |
| `results.json` | Per-trial scores, instructions, seeds, stage-level reasons, and optional video paths |
| `summary.json` | Aggregates grouped by scene and evaluation level |
| `videos/` | Videos generated when `--video` is enabled |

Inspect a run:

```bash
python -m json.tool results/robofollow_full/run.json
python -m json.tool results/robofollow_full/summary.json
```

A completed run has `status == "complete"` and `completed_trials == expected_trials`.

| Summary metric | Definition |
|---|---|
| `mean_intent_score` | Mean stage-based instruction-grounding score |
| `mean_exec_score` | Mean stage-based physical execution score |
| `intent_full_rate` | Fraction of trials reaching at least 99% of the applicable maximum Intent score |
| `exec_full_rate` | Fraction of trials reaching at least 99% of the applicable maximum Execution score |
| `completion_rate` / CR | Task completion rate, averaged equally across tasks within each scene/level |
| `task_macro_completion_rate` | Same task-weighted rate as `completion_rate` |
| `episode_micro_completion_rate` | Fraction of episodes passing the completion rule |

CR classifies each episode using the following rules, with scores on the 0–1 scale:

1. **Pickup-like actions:** `pickup`, `sequence_pickups`, and action names starting with `pickup` require both final total IS and ES to be at least 0.8. These totals include existing score adjustments; finishing-stage credit is not subtracted.
2. **Other actions with an explicit signal:** use `final_target_ok` when available, otherwise `retry_bonus.final_target_ok`. The selected Boolean is the completion result. An explicit `false` means failure; only an absent or null signal permits a fallback.
3. **Other actions without either signal:** check whether `stage3_orientation` has positive execution credit. If that stage is absent, check the first existing stage in this order: `stage2_stack`, `stage2_unstack_slot`, `stage2_place_bowl`, `stage2_push`. Zero execution credit or no matching stage means failure. Stage scores retain their existing prerequisite gates.

The selected signals have different meanings and time windows:

| Task family | Signal used by CR |
|---|---|
| Scene 1 relation placement | `final_target_ok`: the specified relation and table-height condition hold throughout the last 10 recorded frames; the initial grasp and first release are not rechecked |
| Scene 2 stack / bowl placement | `retry_bonus.final_target_ok`: final stack or bowl-placement geometry |
| Scene 2 push | `retry_bonus.final_target_ok`: at any recorded time, the source's displacement projected toward the target's initial position reaches at least 0.005 m; stage-2 push execution separately uses a 0.02 m displacement threshold |
| Scene 3 / non-pickup Scene 4 | The applicable execution-stage fallback above |

CR does not require every task stage to receive full credit, and success is not universally preserved once achieved. Later actions can invalidate a final-layout signal. In Scenes 1, 2, and 4, `stage3_finish` contributes 0.2 to each of IS and ES. It is not a separate CR requirement for non-pickup actions, while pickup-like actions use the final totals including any finish credit and score adjustments.

For example, a Scene 1 relation-placement episode can have zero execution-stage scores, satisfy the final-layout check, and receive the existing +0.3 ES bonus: CR counts it as complete even though ES is only 0.3. Conversely, an episode with 0.8 ES from grasp and placement can fail CR if the relation is later broken. IS/ES and their full-score rates still use all stages and existing score adjustments. Scene 4's `conditional_full_success` is a separate diagnostic requiring every stage and both totals to receive full credit.

CR uses a task macro average: compute each task's success fraction, then average those fractions equally. The episode micro average is also saved; the two rates coincide when every task has the same number of episodes. IS/ES averages remain weighted by episode count.

Inspect `phases` in `results.json` for `intent_reason`, `exec_reason`, stage scores, and maxima. When available, the `video` field gives a path relative to the run directory.

## Project Structure

```text
robofollow/
├── tasks/                 # Training tasks and instruction variants
├── scene_sources/         # Bundled expert scene implementations
├── evaluation/            # Evaluation tasks and stage-wise scorers
├── asset_overrides/       # Bowl visual overrides and hash manifest
├── collect.py             # Expert demonstration collection
├── episodes.py            # Episode validation and publication
├── routes.py              # Expert route validation
├── evaluate.py            # Evaluation entry point
├── simulation.py          # Evaluation environment construction
├── policies.py            # Policy interface validation
├── remote_policy.py       # TCP inference client
├── serve.py               # Policy server
├── convert.py             # Pi HDF5 conversion
├── install_scenes.py       # Scene installation
├── assets.py              # Asset installation and restoration
├── geometry.py            # Scene geometry utilities
├── registry.py            # Task registry access
├── config.py              # Runtime configuration
├── protocol.yml           # Benchmark protocol
└── runinfo.py              # Run metadata
```

## Acknowledgements and Third-Party Notices

RoboFollow builds on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). Policy training can use [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) and [OpenPI](https://github.com/Physical-Intelligence/openpi).

The upstream RoboTwin MIT notice is preserved in [LICENSE.RoboTwin](LICENSE.RoboTwin). Asset setup is documented in [ASSETS.md](ASSETS.md); model weights and other third-party resources are obtained from their respective providers.

## Citation

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
