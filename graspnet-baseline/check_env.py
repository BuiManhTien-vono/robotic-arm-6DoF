import importlib
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
for rel in ("models", "dataset", "utils", "pointnet2", "knn"):
    sys.path.append(os.path.join(ROOT, rel))


def check(name, import_name=None):
    import_name = import_name or name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "")
        print(f"[OK] {name} {version}".rstrip())
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


print("Python:", sys.version)
torch_ok = check("torch")
if torch_ok:
    import torch

    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)

check("open3d")
check("scipy")
check("Pillow", "PIL")
check("graspnetAPI")
pointnet2_ok = check("pointnet2_utils")
if pointnet2_ok:
    import pointnet2_utils

    print("PointNet2 native extension:", pointnet2_utils.POINTNET2_EXT_AVAILABLE)

knn_ok = check("knn_modules")
if knn_ok:
    import knn_modules

    print("KNN native extension:", knn_modules.knn_pytorch is not None)

checkpoint = os.path.join(ROOT, "checkpoints", "checkpoint-rs.tar")
print("Checkpoint:", checkpoint)
print("Checkpoint exists:", os.path.exists(checkpoint))
