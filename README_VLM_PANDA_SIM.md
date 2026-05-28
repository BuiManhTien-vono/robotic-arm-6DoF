# VLM Panda Pick-and-Place Simulation

This package contains the PyBullet Panda simulation integrated with VLM target selection
and the trained GraspNet baseline checkpoint.

## What It Runs

```text
PyBullet scene render
  -> VLM selects target bbox from RGB image
  -> segmentation mask maps bbox to PyBullet object_id
  -> PyBullet RGB-D render creates the target point cloud
  -> trained GraspNet checkpoint predicts a 6DoF grasp pose
  -> Franka Panda executes that GraspNet-based grasp
  -> object is placed into the blue bin
```

## Main Scripts

```text
scripts/06_run_panda_pick_place_sim.py
```

Runs the notebook-style Panda pick-and-place simulation without VLM.

```text
scripts/07_run_vlm_panda_pick_place.py
```

Runs the VLM + GraspNet integrated Panda simulation.

```text
scripts/08_vlm_panda_textbox_app.py
```

Opens a Tkinter text-box UI. Type a natural-language command, then press `Run Command`;
the robot renders the scene, sends the image and command to the VLM, selects the target
object, and executes pick-and-place in PyBullet.

## Setup

The default VLM backend is local Qwen2.5-VL, so the pipeline no longer needs an API key.
Create `.env` from `.env.example` only if you want to override the model or backend:

```powershell
Copy-Item .env.example .env
notepad .env
```

Local default:

```env
VLM_BACKEND=qwen-local
QWEN_VL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

Install the GraspNet/PyBullet-side dependencies in the environment that runs
`graspnet-baseline`:

```powershell
python -m pip install -r requirements_project.txt
```

Create a separate VLM environment for Qwen2.5-VL local:

```powershell
conda create -n env_vlm python=3.10 -y
conda activate env_vlm
```

If `conda` is not available but Python 3.10 is installed:

```powershell
py -3.10 -m venv .venv_vlm
.\.venv_vlm\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Do not use Python 3.12+ for this project setup. Some pinned packages used by
the GraspNet/Qwen split environments do not provide compatible wheels there, so
pip may try to build packages such as `numpy` from source on Windows.

Install a PyTorch build matching your machine before installing the local VLM
requirements. Example for CUDA 12.1:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements_vlm_local.txt
```

For CPU-only testing, install PyTorch from the default index, but Qwen2.5-VL will be slow.
For lower VRAM, set `QWEN_VL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct` and optionally `QWEN_VL_4BIT=1`
after installing `bitsandbytes`.

Do not install `requirements_vlm_local.txt` inside the old GraspNet `.venv`.
GraspNet often needs an older PyTorch stack, while Qwen2.5-VL needs a newer one.
Run `scripts/01_test_vlm.py` in `env_vlm` to produce `data/outputs/vlm_result.json`,
then run the GraspNet/PyBullet side in the GraspNet environment with
`--use-existing-vlm --vlm-result data/outputs/vlm_result.json`.

The default GraspNet checkpoint path is:

```text
graspnet-baseline/checkpoints/checkpoint-rs.tar
```

## Run

Run VLM-integrated simulation:

```powershell
python .\scripts\07_run_vlm_panda_pick_place.py --gui --command "Hay gap mot vat the nho tren ban."
```

Run only the VLM bbox step:

```powershell
python .\scripts\01_test_vlm.py --image .\graspnet-baseline\doc\example_data\color.png --command "Hay gap cai lon mau do."
```

Use Gemini instead of local Qwen only when needed:

```powershell
python .\scripts\01_test_vlm.py --vlm-backend gemini --vlm-model gemini-2.5-flash
```

Run the text-box UI:

```powershell
python .\scripts\08_vlm_panda_textbox_app.py --speed-scale 3.0
```

In the UI, edit the command in the text box and press `Run Command`. Use a larger
`--speed-scale` value for slower robot motion, for example `4.0` or `5.0`.

Run without GUI:

```powershell
python .\scripts\07_run_vlm_panda_pick_place.py --command "Hay gap mot vat the nho tren ban."
```

Test without loading any VLM by using PyBullet segmentation as a mock bbox:

```powershell
python .\scripts\07_run_vlm_panda_pick_place.py --mock-object-index 0
python .\scripts\08_vlm_panda_textbox_app.py --mock --speed-scale 3.0
```

Run the old heuristic grasp without GraspNet:

```powershell
python .\scripts\07_run_vlm_panda_pick_place.py --gui --no-graspnet
python .\scripts\08_vlm_panda_textbox_app.py --no-graspnet
```

Keep GUI open after the task:

```powershell
python .\scripts\07_run_vlm_panda_pick_place.py --gui --keep-open --realtime-sleep
```

## Outputs

Results are saved to:

```text
data/outputs/vlm_panda_sim/
```

Key files:

```text
01_render_rgb.png
01_render_depth.png
02_vlm_result.json
03_selected_bbox.png
04_graspnet/best_grasp.json
04_graspnet/target_cloud_for_graspnet.ply
vlm_panda_result.json
```

## Notes

The VLM may sometimes choose a static scene object such as the bin or table. The script explicitly asks the VLM to ignore robot/table/bin, and also uses the PyBullet segmentation mask plus object semantic metadata to map imperfect bboxes to the correct movable object.

By default the UI uses GraspNet for the grasp pose. The robot still uses a stable downward Panda gripper orientation unless `--use-graspnet-orientation` is supplied, because the raw GraspNet gripper frame can be harder for the Panda IK solver in this simplified simulation.
