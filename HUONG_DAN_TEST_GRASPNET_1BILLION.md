# Huong dan test checkpoint tren GraspNet-1Billion

Tai lieu nay huong dan chay checkpoint:

```text
graspnet-baseline/checkpoints/checkpoint-rs.tar
```

tren ba nhom test chinh thuc cua GraspNet-1Billion:

| Tap test | Scene |
|---|---|
| Seen | `scene_0100` den `scene_0129` |
| Similar | `scene_0130` den `scene_0159` |
| Novel | `scene_0160` den `scene_0189` |

Checkpoint hien tai la checkpoint RealSense, epoch 18. Vi vay phai dung:

```text
--camera realsense
```

## 1. Cau hinh de nghi

Voi may GTX 1650 4 GB va RAM 8 GB:

| Tham so | Gia tri |
|---|---:|
| `camera` | `realsense` |
| `num_point` | `20000` |
| `num_view` | `300` |
| `batch_size` | `1` |
| `collision_thresh` | `0.01` |
| `voxel_size` | `0.01` |
| `num_workers` | `1` |

Khong thay doi `num_view=300`, vi kien truc checkpoint duoc tao voi 300 view.

`collision_thresh=0.01` la cau hinh dung de so sanh voi ket qua chinh thuc cua repo. Co the dat `-1` de chay inference nhanh, nhung ket qua AP se khong con so sanh cong bang voi bang benchmark co collision detection.

## 2. Chuan bi dataset

Dat dataset tai mot thu muc, vi du:

```text
D:\Datasets\graspnet
```

Cau truc toi thieu can co:

```text
D:\Datasets\graspnet
|-- scenes
|   |-- scene_0100
|   |   `-- realsense
|   |       |-- rgb
|   |       |-- depth
|   |       |-- label
|   |       |-- meta
|   |       |-- annotations
|   |       |-- camera_poses.npy
|   |       `-- cam0_wrt_table.npy
|   `-- scene_0189
|-- models
|-- dex_models
|-- grasp_label
`-- collision_label
```

Moi scene test can co 256 frame, tu `0000` den `0255`.

Kiem tra nhanh:

```powershell
Test-Path "D:\Datasets\graspnet\scenes\scene_0100\realsense\depth\0000.png"
Test-Path "D:\Datasets\graspnet\scenes\scene_0189\realsense\depth\0255.png"
Test-Path "D:\Datasets\graspnet\models"
Test-Path "D:\Datasets\graspnet\dex_models"
```

Tat ca lenh tren phai tra ve `True`.

## 3. Cai official GraspNet API

Thu muc `graspnet-baseline/graspnetAPI` trong project hien chi la fallback nho cho demo. No khong the tinh AP benchmark.

Can cai official API vao moi truong GraspNet:

```powershell
cd D:\CV_2026\VLM_2026

git clone https://github.com/graspnet/graspnetAPI.git .\external\graspnetAPI

.\graspnet-baseline\.venv\Scripts\python.exe -m pip install -e .\external\graspnetAPI
```

Kiem tra:

```powershell
.\graspnet-baseline\.venv\Scripts\python.exe -c "from graspnetAPI import GraspNetEval; print(GraspNetEval)"
```

Neu import van tro den file fallback:

```text
graspnet-baseline\graspnetAPI\__init__.py
```

thi can doi ten thu muc fallback:

```powershell
Rename-Item `
  .\graspnet-baseline\graspnetAPI `
  graspnetAPI_demo_fallback
```

Sau do chay lai lenh kiem tra import.

## 4. Chinh so worker cho may 8 GB RAM

Trong file:

```text
graspnet-baseline/test.py
```

tim:

```python
num_workers=4
```

doi thanh:

```python
num_workers=cfgs.num_workers
```

Khi chay, dung:

```text
--num_workers 1
```

Tren Windows, nhieu DataLoader worker co the lam tang RAM va gay loi paging file.

## 5. Chay day du ba tap test

File `test.py` dang dung:

```python
split="test"
```

Split nay bao gom ca 90 scene test. Ham `eval_all()` se tu tach ket qua thanh:

- AP Seen
- AP Similar
- AP Novel
- AP tong

Mo PowerShell va chay:

```powershell
cd D:\CV_2026\VLM_2026\graspnet-baseline

.\.venv\Scripts\python.exe .\test.py `
  --dataset_root "D:\Datasets\graspnet" `
  --checkpoint_path ".\checkpoints\checkpoint-rs.tar" `
  --dump_dir ".\outputs\eval_checkpoint_rs" `
  --camera realsense `
  --num_point 20000 `
  --num_view 300 `
  --batch_size 1 `
  --collision_thresh 0.01 `
  --voxel_size 0.01 `
  --num_workers 1
```

Thay:

```text
D:\Datasets\graspnet
```

bang duong dan dataset that tren may.

## 6. Qua trinh chay

Lenh tren gom hai giai do.

### Giai do 1: Inference

Model chay tren 90 scene x 256 frame:

```text
90 x 256 = 23040 frame
```

Moi frame tao file grasp:

```text
outputs/eval_checkpoint_rs/scene_0100/realsense/0000.npy
...
outputs/eval_checkpoint_rs/scene_0189/realsense/0255.npy
```

Day la qua trinh rat dai. Khong dong PowerShell trong luc dang inference.

### Giai do 2: Evaluation

Sau khi inference xong, `GraspNetEval.eval_all()` se doc cac file `.npy` va tinh AP.

Ket qua chi tiet duoc luu tai:

```text
graspnet-baseline/outputs/eval_checkpoint_rs/ap_realsense.npy
```

## 7. Cach doc ket qua

Cuoi terminal se co dang:

```text
Evaluation Result:
----------
realsense, AP=..., AP Seen=..., AP Similar=..., AP Novel=...
```

Y nghia:

| Chi so | Y nghia |
|---|---|
| `AP` | Trung binh tren tat ca 90 scene |
| `AP Seen` | Vat the da xuat hien trong train |
| `AP Similar` | Vat the khac nhung tuong tu train |
| `AP Novel` | Vat the moi |

Ket qua tham khao cua pretrained RealSense trong repo:

| Split | AP | AP 0.8 | AP 0.4 |
|---|---:|---:|---:|
| Seen | 47.47 | 55.90 | 41.33 |
| Similar | 42.27 | 51.01 | 35.40 |
| Novel | 16.61 | 20.84 | 8.30 |

Sai khac nho co the den tu phien ban PyTorch, CUDA, Open3D, collision detector va sampling ngau nhien.

## 8. Chay inference nhanh truoc khi benchmark

De kiem tra pipeline ma khong chay collision detection:

```powershell
.\.venv\Scripts\python.exe .\test.py `
  --dataset_root "D:\Datasets\graspnet" `
  --checkpoint_path ".\checkpoints\checkpoint-rs.tar" `
  --dump_dir ".\outputs\eval_checkpoint_rs_fast" `
  --camera realsense `
  --num_point 20000 `
  --num_view 300 `
  --batch_size 1 `
  --collision_thresh -1 `
  --voxel_size 0.01 `
  --num_workers 1
```

Day chi la che do kiem tra toc do/pipeline. De bao cao benchmark, chay lai voi:

```text
--collision_thresh 0.01
```

## 9. Chay rieng Seen, Similar, Novel

Mã hien tai da tinh ca ba split trong mot lan chay, day la cach nen dung.

Neu can inference rieng, doi dong sau trong `test.py`:

```python
split="test"
```

thanh mot trong ba gia tri:

```python
split="test_seen"
split="test_similar"
split="test_novel"
```

Sau do evaluation phai goi ham tuong ung:

```python
ge.eval_seen(...)
ge.eval_similar(...)
ge.eval_novel(...)
```

Moi split phai dung `dump_dir` rieng. Khong nen ghi de ket qua giua cac split.

## 10. Loi thuong gap

### `ImportError: Local graspnetAPI fallback only supports GraspGroup`

Chua cai official `graspnetAPI`, hoac Python dang import nham fallback trong repo.

### `FileNotFoundError` trong `scene_0100`

Sai `dataset_root`, dataset chua tai du scene, hoac chua tai camera RealSense.

### `CUDA out of memory`

Giu:

```text
--batch_size 1
--num_point 20000
--num_workers 1
```

Dong Qwen worker, PyBullet, Chrome va Jupyter truoc khi benchmark.

### `The paging file is too small`

Dong cac ung dung ton RAM va tang Windows pagefile. Benchmark GraspNet khong can mo Qwen hay app textbox.

### Ket qua thap bat thuong

Kiem tra:

- checkpoint RealSense phai di voi `--camera realsense`;
- `num_view` phai la `300`;
- dung `collision_thresh=0.01` khi so sanh benchmark;
- dataset phai co day du `models`, `dex_models`, annotations va camera poses;
- output cu trong `dump_dir` khong bi tron voi lan test khac.

## 11. Xoa output va chay lai

Chi xoa dung thu muc output benchmark:

```powershell
Remove-Item `
  -LiteralPath "D:\CV_2026\VLM_2026\graspnet-baseline\outputs\eval_checkpoint_rs" `
  -Recurse `
  -Force
```

Sau do chay lai lenh trong muc 5.

