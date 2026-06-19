# robotic-arm-6DoF

Robotic arm 6DoF project integrating VLM object localization, GraspNet grasp pose estimation, and PyBullet robot simulation.

## Main Components

- Local/API VLM object localization in `src/vlm_localizer.py`
- RGB-D to point cloud conversion and GraspNet inference
- PyBullet Panda/xArm pick-and-place simulation scripts
- Textbox UI for natural-language commands in `scripts/08_vlm_panda_textbox_app.py`

## Local Run Notes

The project uses separate Python environments for the VLM stack and the GraspNet/PyBullet stack. See:

- `README_VLM_PANDA_SIM.md`
- `requirements_project.txt`
- `requirements_vlm_local.txt`

Generated outputs, local virtual environments, model caches, checkpoints, and `.env` files are intentionally ignored by Git.

## Run With Gemini API

Add `GEMINI_API_KEY` and `GEMINI_MODEL=gemini-2.5-flash` to `.env`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_textbox_gemini_api.ps1
```

This keeps VLM inference in the API while GraspNet and PyBullet continue running locally.
