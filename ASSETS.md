# Simulation assets

RoboFollow uses the ALOHA AgileX robot and object assets from RoboTwin. Download them following the [RoboTwin installation instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

## Bowl visuals

The following files replace the corresponding bowl visuals in RoboTwin:

| File | Appearance |
|---|---|
| `asset_overrides/002_bowl/visual/base1.glb` | Yellow bowl |
| `asset_overrides/002_bowl/visual/base3.glb` | Green bowl |

Scene 2 uses both bowls; Scene 4 uses model 3. Collision meshes and model metadata remain unchanged. `asset_overrides/manifest.json` records the SHA-256 hashes of the bundled files and the RoboTwin files they replace.

## Installation

Run from the RoboTwin root after downloading the assets:

```bash
python scripts/update_embodiment_config_path.py
python -m robofollow.assets install
python -m robofollow.assets status
```

The installer verifies hashes and saves backups in `assets/.robofollow-backups/`. To restore the backed-up visuals:

```bash
python -m robofollow.assets restore
```

The RoboTwin code license is preserved in [LICENSE.RoboTwin](LICENSE.RoboTwin).
