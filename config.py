"""Load benchmark protocol settings and RoboTwin environment paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def environment_args(scene, *, seed=0, output=None, evaluation=False):
    args = yaml.safe_load((ROOT / "robofollow/protocol.yml").read_text())
    assets = ROOT / "assets"
    robot_file = assets / "embodiments/aloha-agilex"
    robot = yaml.safe_load((robot_file / "config.yml").read_text())
    camera = yaml.safe_load((ROOT / "env_cfg/task_config/_camera_config.yml").read_text())
    spec = camera[args["camera"]["head_camera_type"]]
    args.update(
        task_name=scene, seed=seed, now_ep_num=0, render_freq=0,
        eval_mode=evaluation, need_plan=True, save_data=False,
        left_robot_file=str(robot_file), right_robot_file=str(robot_file),
        left_embodiment_config=robot, right_embodiment_config=robot,
        dual_arm_embodied=True, embodiment_name="aloha-agilex",
        task_config="robofollow", head_camera_h=spec["h"], head_camera_w=spec["w"],
        eval_video_log=False,
        save_path=str(output or ROOT / "data/robofollow"),
    )
    return args
