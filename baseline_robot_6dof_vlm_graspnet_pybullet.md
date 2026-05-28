# BASELINE TRIỂN KHAI TỪNG BƯỚC  
# Hệ thống gắp thả robot 6DOF tích hợp VLM + GraspNet + PyBullet

**Dự án:** Đề tài Nghiên cứu Khoa học cấp Đại học  
**Bài toán:** Robotic Semantic Grasping – gắp vật thể dựa trên mô tả ngôn ngữ tự nhiên  
**Mục tiêu baseline:** Tạo được một pipeline tối thiểu có thể chạy được từ lệnh người dùng đến thao tác gắp-thả trong mô phỏng.

---

## 0. Tóm tắt ý tưởng baseline

Baseline của hệ thống được chia thành 6 mô-đun chính:

```text
Người dùng nhập lệnh
        ↓
Camera RGB-D chụp ảnh khu vực làm việc
        ↓
VLM xác định vật thể mục tiêu theo ngữ nghĩa
        ↓
Cắt vùng vật thể và tạo point cloud
        ↓
GraspNet dự đoán tư thế gắp 6DoF
        ↓
PyBullet tính động học ngược và điều khiển robot 6DOF
```

Ví dụ lệnh đầu vào:

```text
Hãy gắp cái cốc màu xanh ở bên trái.
```

Đầu ra mong muốn:

```text
Robot xác định đúng cái cốc màu xanh → tìm tư thế gắp tốt nhất → di chuyển tay gắp đến vật thể → đóng kẹp → nâng vật thể → thả tại vị trí đích.
```

---

## 1. Phạm vi baseline

Baseline này không cố gắng xây dựng toàn bộ hệ thống hoàn hảo ngay từ đầu. Mục tiêu là tạo một phiên bản **chạy được, kiểm thử được, giải thích được**.

### 1.1. Phiên bản tối thiểu cần hoàn thành

| Mốc | Nội dung | Kết quả cần có |
|---|---|---|
| M1 | Cài được GraspNet Baseline | Chạy được `demo.py` hoặc inference mẫu |
| M2 | Có VLM nhận ảnh + câu lệnh | Trả về vật thể mục tiêu và bounding box |
| M3 | Chuyển RGB-D sang point cloud | Tạo được point cloud của vùng vật thể |
| M4 | Dùng GraspNet dự đoán grasp pose | Có danh sách grasp candidates |
| M5 | Chọn grasp tốt nhất | Lấy được pose `(x, y, z, roll, pitch, yaw)` |
| M6 | Đưa pose vào PyBullet | Robot 6DOF di chuyển đến vị trí gắp |
| M7 | Mô phỏng gắp-thả | Robot đóng/mở kẹp và thả vật thể |

### 1.2. Những gì chưa cần làm ở baseline

Ở giai đoạn baseline, chưa cần làm các phần sau:

- Chưa cần huấn luyện lại GraspNet từ đầu.
- Chưa cần tự xây dựng mô hình VLM.
- Chưa cần chạy trên robot thật.
- Chưa cần tối ưu tốc độ real-time.
- Chưa cần nhận diện mọi loại vật thể phức tạp.
- Chưa cần đánh giá benchmark đầy đủ như paper gốc.

---

## 2. Kiến trúc tổng thể

### 2.1. Sơ đồ pipeline

```text
+-------------------+
| User Command      |
| "gắp cốc xanh"    |
+---------+---------+
          |
          v
+-------------------+
| RGB Image         |
| Depth Image       |
+---------+---------+
          |
          v
+-------------------+
| VLM Gemini        |
| Object Reasoning  |
+---------+---------+
          |
          v
+-----------------------------+
| Target Object Localization  |
| bbox + object description   |
+---------+-------------------+
          |
          v
+-----------------------------+
| RGB-D to Point Cloud        |
| Crop point cloud by bbox    |
+---------+-------------------+
          |
          v
+-----------------------------+
| GraspNet Baseline           |
| 6DoF grasp candidates       |
+---------+-------------------+
          |
          v
+-----------------------------+
| Best Grasp Selection        |
| score + collision filtering |
+---------+-------------------+
          |
          v
+-----------------------------+
| Coordinate Transform        |
| camera frame → robot/world  |
+---------+-------------------+
          |
          v
+-----------------------------+
| PyBullet IK + Motion        |
| 6DOF robot control          |
+---------+-------------------+
          |
          v
+-----------------------------+
| Pick and Place Result       |
+-----------------------------+
```

### 2.2. Dữ liệu đầu vào và đầu ra của từng mô-đun

| Mô-đun | Đầu vào | Đầu ra |
|---|---|---|
| VLM | Ảnh RGB + câu lệnh | `target_object`, `bbox`, `reason` |
| RGB-D Processing | RGB, Depth, bbox, camera intrinsics | point cloud vùng vật thể |
| GraspNet | point cloud | danh sách grasp poses |
| Grasp Selection | grasp poses + score | grasp pose tốt nhất |
| Transform | pose trong camera frame | pose trong robot/world frame |
| PyBullet IK | end-effector pose | góc các khớp robot |
| Motion Control | joint angles | robot thực hiện gắp-thả |

---

## 3. Cấu trúc thư mục đề xuất

Nên tách code của bạn khỏi repo gốc `graspnet-baseline` để dễ quản lý.

```text
robotic-semantic-grasping/
│
├── README.md
├── .env
├── requirements_project.txt
├── environment.yml
│
├── external/
│   ├── graspnet-baseline/
│   └── graspnetAPI/
│
├── checkpoints/
│   ├── checkpoint-rs.tar
│   └── checkpoint-kn.tar
│
├── data/
│   ├── sample_rgb/
│   ├── sample_depth/
│   ├── sample_intrinsics/
│   └── outputs/
│
├── configs/
│   ├── camera.yaml
│   ├── robot.yaml
│   ├── graspnet.yaml
│   └── pipeline.yaml
│
├── src/
│   ├── main.py
│   ├── vlm_localizer.py
│   ├── rgbd_to_pointcloud.py
│   ├── graspnet_adapter.py
│   ├── grasp_selector.py
│   ├── coordinate_transform.py
│   ├── pybullet_controller.py
│   └── utils.py
│
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_test_vlm.py
│   ├── 02_test_rgbd_pointcloud.py
│   ├── 03_test_graspnet_demo.sh
│   ├── 04_test_pybullet_robot.py
│   └── 05_run_full_pipeline.py
│
└── docs/
    ├── baseline_steps.md
    ├── experiment_log.md
    └── troubleshooting.md
```

---

## 4. Chuẩn bị môi trường

### 4.1. Khuyến nghị hệ điều hành

Nên dùng:

```text
Ubuntu 20.04 / Ubuntu 22.04
hoặc Windows + WSL2 Ubuntu
```

Không khuyến nghị cài trực tiếp trên Windows nếu chưa quen, vì bước build `pointnet2` và `knn` cần biên dịch C++/CUDA.

### 4.2. Yêu cầu phần cứng

| Thành phần | Khuyến nghị |
|---|---|
| CPU | Intel i5/i7 hoặc AMD Ryzen 5/7 trở lên |
| RAM | Tối thiểu 16GB, khuyến nghị 32GB |
| GPU | NVIDIA GPU có CUDA |
| VRAM | Tối thiểu 6GB, khuyến nghị 8GB trở lên |
| Camera | RGB-D camera, ví dụ Intel RealSense, hoặc dùng dữ liệu mẫu trước |
| Robot | Giai đoạn đầu dùng robot mô phỏng trong PyBullet |

### 4.3. Tạo môi trường Conda

```bash
conda create -n semantic_grasp python=3.8 -y
conda activate semantic_grasp
```

Kiểm tra Python:

```bash
python --version
```

Kết quả mong muốn:

```text
Python 3.8.x
```

> Ghi chú: Repo gốc GraspNet Baseline khá cũ và yêu cầu PyTorch 1.6 trong tài liệu gốc. Nếu máy của bạn dùng CUDA mới, có thể cần điều chỉnh phiên bản PyTorch/CUDA hoặc dùng fork tương thích hơn. Tuy nhiên, baseline nên bắt đầu từ repo gốc để dễ giải thích trong báo cáo.

---

## 5. Cài đặt GraspNet Baseline

### 5.1. Clone repo chính thức

Từ thư mục dự án chính:

```bash
mkdir -p external
cd external

git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```

### 5.2. Cài requirements

```bash
pip install -r requirements.txt
```

Nếu lỗi do version quá cũ, có thể cài từng gói chính:

```bash
pip install numpy scipy pillow tqdm tensorboard
pip install open3d
```

### 5.3. Cài PyTorch phù hợp CUDA

Kiểm tra CUDA:

```bash
nvidia-smi
nvcc --version
```

Kiểm tra PyTorch:

```bash
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY
```

Nếu `torch.cuda.is_available()` trả về `False`, GraspNet vẫn có thể import một phần, nhưng inference sẽ rất chậm hoặc lỗi nếu code yêu cầu CUDA.

---

## 6. Build PointNet2 và KNN

Đây là bước rất quan trọng vì GraspNet dùng toán tử mở rộng bằng C++/CUDA.

### 6.1. Build PointNet2

Từ thư mục `external/graspnet-baseline`:

```bash
cd pointnet2
python setup.py install
cd ..
```

### 6.2. Build KNN

```bash
cd knn
python setup.py install
cd ..
```

### 6.3. Kiểm tra import sau khi build

Tạo file:

```bash
nano ../../scripts/00_check_env.py
```

Nội dung:

```python
import torch
import open3d as o3d

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Open3D:", o3d.__version__)

try:
    import pointnet2
    print("pointnet2 import OK")
except Exception as e:
    print("pointnet2 import FAILED:", e)

try:
    import knn
    print("knn import OK")
except Exception as e:
    print("knn import FAILED:", e)
```

Chạy:

```bash
cd ../../
python scripts/00_check_env.py
```

Kết quả mong muốn:

```text
CUDA available: True
pointnet2 import OK
knn import OK
```

---

## 7. Cài GraspNet API

GraspNet API dùng để load, xử lý, visualize và đánh giá kết quả GraspNet.

Từ thư mục `external`:

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
cd ../..
```

Kiểm tra:

```bash
python - << 'PY'
from graspnetAPI import GraspGroup
print("graspnetAPI import OK")
PY
```

---

## 8. Tải checkpoint pretrained

Repo GraspNet Baseline cung cấp checkpoint pretrained:

```text
checkpoint-rs.tar  : huấn luyện với dữ liệu camera RealSense
checkpoint-kn.tar  : huấn luyện với dữ liệu camera Kinect
```

Khuyến nghị baseline:

```text
Dùng checkpoint-rs.tar trước vì phù hợp hơn nếu bạn dùng RealSense hoặc dữ liệu RGB-D phổ biến.
```

Tạo thư mục:

```bash
mkdir -p checkpoints
```

Sau khi tải, đặt file như sau:

```text
robotic-semantic-grasping/checkpoints/checkpoint-rs.tar
```

---

## 9. Chạy thử GraspNet demo gốc

Trước khi tích hợp VLM, bắt buộc phải chạy được demo gốc.

### 9.1. Chạy demo

Từ thư mục `external/graspnet-baseline`:

```bash
python demo.py --checkpoint_path ../../checkpoints/checkpoint-rs.tar
```

Nếu repo có `command_demo.sh`, có thể chạy:

```bash
bash command_demo.sh
```

Nhưng cần mở file `command_demo.sh` để chỉnh lại đường dẫn checkpoint nếu cần.

### 9.2. Kết quả mong muốn

Kết quả mong muốn:

```text
- Load được checkpoint
- Load được RGB-D sample
- Sinh ra nhiều grasp candidates
- Visualize được grasp bằng Open3D
```

Nếu chưa chạy được demo gốc, không nên vội tích hợp VLM/PyBullet.

---

## 10. Chuẩn bị dữ liệu RGB-D

### 10.1. Dữ liệu cần có cho mỗi frame

Một frame đầu vào cần gồm:

```text
rgb.png
depth.png
camera_intrinsics.json
```

Ví dụ:

```text
data/sample_rgb/frame_0001.png
data/sample_depth/frame_0001.png
data/sample_intrinsics/realsense_intrinsics.json
```

### 10.2. Camera intrinsics

File `configs/camera.yaml`:

```yaml
camera:
  name: "realsense"
  width: 640
  height: 480
  fx: 615.0
  fy: 615.0
  cx: 320.0
  cy: 240.0
  depth_scale: 1000.0
```

Trong đó:

| Tham số | Ý nghĩa |
|---|---|
| `fx`, `fy` | tiêu cự theo pixel |
| `cx`, `cy` | tâm ảnh |
| `depth_scale` | hệ số đổi depth raw sang mét |

Nếu depth lưu theo milimet:

```python
z = depth_raw / 1000.0
```

---

## 11. Mô-đun VLM xác định vật thể mục tiêu

### 11.1. Mục tiêu

VLM nhận:

```text
- Ảnh RGB toàn cảnh
- Câu lệnh người dùng
```

VLM trả về:

```json
{
  "target_object": "blue cup",
  "bbox": [120, 80, 260, 310],
  "confidence": 0.91,
  "reason": "The blue cup is on the left side of the table."
}
```

Trong đó:

```text
bbox = [x_min, y_min, x_max, y_max]
```

### 11.2. Cài Gemini SDK

```bash
pip install google-genai python-dotenv pydantic
```

Tạo file `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Nếu đề tài của bạn cố định dùng Gemini 1.5 Flash, có thể đặt:

```env
GEMINI_MODEL=gemini-1.5-flash
```

Nếu API báo model cũ không còn được hỗ trợ, chỉ cần đổi biến `GEMINI_MODEL` sang model Flash mới hơn mà không cần sửa pipeline.

### 11.3. File `src/vlm_localizer.py`

```python
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)


def localize_target_object(image_path: str, user_command: str) -> dict:
    """
    Input:
        image_path: đường dẫn ảnh RGB
        user_command: câu lệnh người dùng, ví dụ "gắp cái cốc màu xanh"

    Output:
        dict gồm target_object, bbox, confidence, reason
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = f"""
Bạn là mô-đun nhận thức ngữ nghĩa cho robot gắp thả.
Nhiệm vụ: xác định vật thể mục tiêu trong ảnh dựa trên câu lệnh người dùng.

Câu lệnh người dùng:
"{user_command}"

Hãy trả về JSON duy nhất theo schema:
{{
  "target_object": "tên vật thể mục tiêu",
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": số từ 0 đến 1,
  "reason": "giải thích ngắn"
}}

Yêu cầu:
- bbox dùng tọa độ pixel trên ảnh gốc.
- Không thêm markdown.
- Không giải thích ngoài JSON.
- Nếu không chắc, vẫn chọn vật thể phù hợp nhất và giảm confidence.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ],
        config={
            "response_mime_type": "application/json"
        }
    )

    try:
        return json.loads(response.text)
    except Exception:
        raise ValueError(f"VLM response is not valid JSON: {response.text}")
```

### 11.4. File test `scripts/01_test_vlm.py`

```python
from src.vlm_localizer import localize_target_object

image_path = "data/sample_rgb/frame_0001.png"
command = "Hãy gắp cái cốc màu xanh ở bên trái."

result = localize_target_object(image_path, command)
print(result)
```

Chạy:

```bash
python scripts/01_test_vlm.py
```

---

## 12. Chuyển RGB-D sang point cloud

### 12.1. Công thức chuyển đổi

Với pixel `(u, v)` và giá trị độ sâu `z`, tọa độ 3D trong camera frame:

```text
X = (u - cx) * z / fx
Y = (v - cy) * z / fy
Z = z
```

Điểm 3D:

```text
p = (X, Y, Z)
```

### 12.2. File `src/rgbd_to_pointcloud.py`

```python
import numpy as np
import cv2
import open3d as o3d


def load_depth(depth_path: str, depth_scale: float = 1000.0):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(depth_path)

    depth_m = depth_raw.astype(np.float32) / depth_scale
    return depth_m


def crop_rgbd_by_bbox(rgb_path: str, depth_path: str, bbox: list, depth_scale: float):
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(rgb_path)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = load_depth(depth_path, depth_scale)

    x_min, y_min, x_max, y_max = map(int, bbox)

    rgb_crop = rgb[y_min:y_max, x_min:x_max]
    depth_crop = depth[y_min:y_max, x_min:x_max]

    return rgb_crop, depth_crop, (x_min, y_min)


def create_point_cloud_from_crop(
    rgb_crop,
    depth_crop,
    offset_xy,
    intrinsics: dict,
    max_depth: float = 2.0
):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x_offset, y_offset = offset_xy

    points = []
    colors = []

    h, w = depth_crop.shape

    for v in range(h):
        for u in range(w):
            z = depth_crop[v, u]

            if z <= 0 or z > max_depth:
                continue

            u_global = u + x_offset
            v_global = v + y_offset

            x = (u_global - cx) * z / fx
            y = (v_global - cy) * z / fy

            points.append([x, y, z])
            colors.append(rgb_crop[v, u] / 255.0)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_point_cloud(pcd, output_path: str):
    o3d.io.write_point_cloud(output_path, pcd)
```

### 12.3. File test `scripts/02_test_rgbd_pointcloud.py`

```python
import open3d as o3d
from src.rgbd_to_pointcloud import crop_rgbd_by_bbox, create_point_cloud_from_crop, save_point_cloud

rgb_path = "data/sample_rgb/frame_0001.png"
depth_path = "data/sample_depth/frame_0001.png"

bbox = [120, 80, 260, 310]

intrinsics = {
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0,
}

rgb_crop, depth_crop, offset_xy = crop_rgbd_by_bbox(
    rgb_path=rgb_path,
    depth_path=depth_path,
    bbox=bbox,
    depth_scale=1000.0
)

pcd = create_point_cloud_from_crop(
    rgb_crop=rgb_crop,
    depth_crop=depth_crop,
    offset_xy=offset_xy,
    intrinsics=intrinsics,
    max_depth=2.0
)

save_point_cloud(pcd, "data/outputs/target_object.ply")
o3d.visualization.draw_geometries([pcd])
```

Chạy:

```bash
mkdir -p data/outputs
python scripts/02_test_rgbd_pointcloud.py
```

---

## 13. Tích hợp point cloud vào GraspNet

### 13.1. Cách làm đúng ở baseline

Không nên sửa toàn bộ code GraspNet ngay lập tức. Nên đi theo 3 mức:

```text
Mức 1: Chạy được demo.py gốc.
Mức 2: Sửa hàm get_and_process_data() trong demo.py để dùng RGB-D của mình.
Mức 3: Tách thành adapter riêng graspnet_adapter.py.
```

### 13.2. Mức 1 – chạy demo gốc

```bash
cd external/graspnet-baseline
python demo.py --checkpoint_path ../../checkpoints/checkpoint-rs.tar
```

### 13.3. Mức 2 – sửa `get_and_process_data()`

Trong repo GraspNet Baseline, file `demo.py` có phần xử lý dữ liệu demo. Ý tưởng sửa:

```python
def get_and_process_data(data_dir):
    """
    Thay vì dùng dữ liệu mẫu của repo,
    hàm này sẽ đọc:
    - RGB image của bạn
    - Depth image của bạn
    - Camera intrinsics của bạn
    - Workspace mask nếu có
    """
```

Dữ liệu cần đưa về format GraspNet mong đợi:

```text
point_clouds
coors
feats
cloud
```

Ở baseline, có thể giữ cấu trúc xử lý giống `demo.py`, chỉ thay nguồn ảnh/depth.

### 13.4. Mức 3 – tạo `src/graspnet_adapter.py`

Mục tiêu của adapter:

```python
class GraspNetAdapter:
    def __init__(self, checkpoint_path):
        ...

    def predict(self, rgb_path, depth_path, bbox, intrinsics):
        ...
        return grasp_candidates
```

Skeleton:

```python
class GraspNetAdapter:
    def __init__(self, checkpoint_path: str, repo_root: str):
        self.checkpoint_path = checkpoint_path
        self.repo_root = repo_root
        self.model = self._load_model()

    def _load_model(self):
        """
        TODO:
        - import GraspNet model từ repo gốc
        - load checkpoint
        - set eval mode
        """
        pass

    def predict_from_rgbd(self, rgb_path, depth_path, bbox, intrinsics):
        """
        Input:
            rgb_path
            depth_path
            bbox
            intrinsics

        Output:
            list hoặc GraspGroup chứa các grasp candidates
        """
        # 1. Crop vùng vật thể bằng bbox
        # 2. Tạo point cloud
        # 3. Format dữ liệu theo GraspNet
        # 4. Inference
        # 5. Decode grasp predictions
        # 6. Return grasp candidates
        pass
```

---

## 14. Biểu diễn grasp pose

Một grasp candidate nên được chuẩn hóa về dạng thống nhất trong hệ thống của bạn.

### 14.1. Format đề xuất

```python
grasp = {
    "translation": [x, y, z],
    "rotation_matrix": [
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ],
    "width": 0.05,
    "score": 0.89
}
```

### 14.2. Chuyển rotation matrix sang Euler angle

```python
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_euler(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler("xyz", degrees=False)
    return roll, pitch, yaw
```

Sau khi chuyển:

```python
pose_6dof = {
    "x": x,
    "y": y,
    "z": z,
    "roll": roll,
    "pitch": pitch,
    "yaw": yaw,
    "width": width,
    "score": score
}
```

---

## 15. Lọc và chọn grasp tốt nhất

### 15.1. Tiêu chí chọn

Không nên chỉ lấy grasp có score cao nhất. Baseline nên lọc theo:

| Tiêu chí | Mục đích |
|---|---|
| `score` cao | tư thế gắp đáng tin cậy |
| không va chạm mặt bàn | tránh gắp xuyên bàn |
| nằm trong workspace robot | robot với tới được |
| hướng tiếp cận hợp lý | tránh gắp từ dưới lên |
| độ mở kẹp phù hợp | phù hợp kích thước vật thể |

### 15.2. File `src/grasp_selector.py`

```python
def is_inside_workspace(translation, workspace):
    x, y, z = translation

    return (
        workspace["x_min"] <= x <= workspace["x_max"] and
        workspace["y_min"] <= y <= workspace["y_max"] and
        workspace["z_min"] <= z <= workspace["z_max"]
    )


def is_above_table(translation, table_z=0.0, margin=0.02):
    return translation[2] > table_z + margin


def select_best_grasp(grasp_candidates, workspace, table_z=0.0):
    valid_grasps = []

    for g in grasp_candidates:
        translation = g["translation"]
        score = g["score"]

        if score < 0.3:
            continue

        if not is_inside_workspace(translation, workspace):
            continue

        if not is_above_table(translation, table_z):
            continue

        valid_grasps.append(g)

    if len(valid_grasps) == 0:
        return None

    valid_grasps = sorted(valid_grasps, key=lambda x: x["score"], reverse=True)
    return valid_grasps[0]
```

### 15.3. Workspace mẫu

File `configs/robot.yaml`:

```yaml
workspace:
  x_min: -0.5
  x_max: 0.5
  y_min: -0.5
  y_max: 0.5
  z_min: 0.02
  z_max: 0.8

table:
  z: 0.0
```

---

## 16. Chuyển hệ tọa độ camera sang robot/world

### 16.1. Vấn đề

GraspNet dự đoán pose trong hệ tọa độ camera:

```text
camera frame
```

Nhưng PyBullet/robot cần pose trong hệ tọa độ world hoặc base robot:

```text
world frame / robot base frame
```

Do đó cần ma trận biến đổi:

```text
T_world_camera
```

### 16.2. Công thức

Nếu điểm trong camera frame là:

```text
p_camera = [x, y, z, 1]^T
```

Thì điểm trong world frame là:

```text
p_world = T_world_camera × p_camera
```

### 16.3. File `src/coordinate_transform.py`

```python
import numpy as np


def transform_point(T, point):
    point_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    point_world = T @ point_h
    return point_world[:3]


def transform_rotation(T, R_camera):
    R_world_camera = T[:3, :3]
    R_world = R_world_camera @ R_camera
    return R_world


def transform_grasp_camera_to_world(grasp, T_world_camera):
    grasp_world = grasp.copy()

    grasp_world["translation"] = transform_point(
        T_world_camera,
        grasp["translation"]
    ).tolist()

    grasp_world["rotation_matrix"] = transform_rotation(
        T_world_camera,
        np.array(grasp["rotation_matrix"])
    ).tolist()

    return grasp_world
```

### 16.4. Ma trận transform tạm cho baseline

Nếu chưa calibrate thật, có thể dùng transform giả định trong mô phỏng:

```python
T_world_camera = np.array([
    [1, 0, 0, 0.0],
    [0, 1, 0, 0.0],
    [0, 0, 1, 0.5],
    [0, 0, 0, 1.0],
], dtype=np.float32)
```

> Trong báo cáo cần ghi rõ: ở baseline mô phỏng, ma trận camera-to-world có thể được thiết lập thủ công. Khi chuyển sang robot thật, cần thực hiện camera calibration / hand-eye calibration.

---

## 17. Thiết lập PyBullet robot 6DOF

### 17.1. Cài PyBullet

```bash
pip install pybullet
```

### 17.2. Tạo mô phỏng cơ bản

File `src/pybullet_controller.py`:

```python
import time
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R


class PyBulletRobotController:
    def __init__(self, gui=True):
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.end_effector_link_index = None
        self.arm_joint_indices = []

    def connect(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def load_scene(self):
        p.loadURDF("plane.urdf")

        # Có thể thay bằng URDF robot 6DOF của bạn.
        # KUKA iiwa là robot mẫu phổ biến trong PyBullet.
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )

        self.arm_joint_indices = list(range(7))
        self.end_effector_link_index = 6

    def get_quaternion_from_rotation_matrix(self, rotation_matrix):
        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
        return quat_xyzw.tolist()

    def compute_ik(self, target_position, target_orientation_quat):
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link_index,
            targetPosition=target_position,
            targetOrientation=target_orientation_quat
        )

        return joint_poses

    def move_joints(self, joint_poses, steps=240):
        for i, joint_index in enumerate(self.arm_joint_indices):
            if i >= len(joint_poses):
                break

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500
            )

        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def move_to_grasp(self, grasp_world):
        target_position = grasp_world["translation"]
        rotation_matrix = np.array(grasp_world["rotation_matrix"])

        target_quat = self.get_quaternion_from_rotation_matrix(rotation_matrix)
        joint_poses = self.compute_ik(target_position, target_quat)

        self.move_joints(joint_poses)

    def disconnect(self):
        p.disconnect()
```

### 17.3. Test PyBullet

File `scripts/04_test_pybullet_robot.py`:

```python
from src.pybullet_controller import PyBulletRobotController

controller = PyBulletRobotController(gui=True)
controller.connect()
controller.load_scene()

dummy_grasp = {
    "translation": [0.4, 0.0, 0.4],
    "rotation_matrix": [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
}

controller.move_to_grasp(dummy_grasp)
input("Press Enter to exit...")
controller.disconnect()
```

Chạy:

```bash
python scripts/04_test_pybullet_robot.py
```

Kết quả mong muốn:

```text
Robot KUKA trong PyBullet di chuyển end-effector đến pose mục tiêu.
```

---

## 18. Mô phỏng gripper

### 18.1. Cách làm đơn giản cho baseline

Ở baseline, có thể mô phỏng thao tác gắp bằng 3 mức:

```text
Mức 1: Chỉ di chuyển end-effector đến vị trí gắp.
Mức 2: Thêm gripper giả bằng object constraint.
Mức 3: Dùng URDF robot có gripper thật.
```

### 18.2. Mức 1 – chỉ kiểm tra pose

Mục tiêu là xem robot có tới đúng vị trí không.

```text
Không cần đóng/mở kẹp thật.
```

### 18.3. Mức 2 – gắp bằng constraint

Khi end-effector đến gần vật thể, tạo constraint để gắn vật thể vào end-effector:

```python
constraint_id = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=end_effector_link_index,
    childBodyUniqueId=object_id,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)
```

Khi thả vật thể:

```python
p.removeConstraint(constraint_id)
```

### 18.4. Mức 3 – dùng gripper thật

Khi đã ổn định, thay robot mẫu bằng URDF có gripper:

```text
UR5 + Robotiq 2F-85
Franka Panda
xArm + gripper
KUKA + custom gripper
```

---

## 19. Full pipeline baseline

### 19.1. File `src/main.py`

```python
import numpy as np

from src.vlm_localizer import localize_target_object
from src.rgbd_to_pointcloud import crop_rgbd_by_bbox, create_point_cloud_from_crop
from src.grasp_selector import select_best_grasp
from src.coordinate_transform import transform_grasp_camera_to_world
from src.pybullet_controller import PyBulletRobotController


def main():
    user_command = "Hãy gắp cái cốc màu xanh ở bên trái."

    rgb_path = "data/sample_rgb/frame_0001.png"
    depth_path = "data/sample_depth/frame_0001.png"

    intrinsics = {
        "fx": 615.0,
        "fy": 615.0,
        "cx": 320.0,
        "cy": 240.0,
    }

    workspace = {
        "x_min": -0.5,
        "x_max": 0.5,
        "y_min": -0.5,
        "y_max": 0.5,
        "z_min": 0.02,
        "z_max": 0.8
    }

    print("[1] Running VLM localization...")
    vlm_result = localize_target_object(rgb_path, user_command)
    bbox = vlm_result["bbox"]

    print("VLM result:", vlm_result)

    print("[2] Creating target point cloud...")
    rgb_crop, depth_crop, offset_xy = crop_rgbd_by_bbox(
        rgb_path=rgb_path,
        depth_path=depth_path,
        bbox=bbox,
        depth_scale=1000.0
    )

    pcd = create_point_cloud_from_crop(
        rgb_crop=rgb_crop,
        depth_crop=depth_crop,
        offset_xy=offset_xy,
        intrinsics=intrinsics
    )

    print("[3] Running GraspNet...")
    # TODO:
    # graspnet = GraspNetAdapter(
    #     checkpoint_path="checkpoints/checkpoint-rs.tar",
    #     repo_root="external/graspnet-baseline"
    # )
    # grasp_candidates = graspnet.predict_from_rgbd(rgb_path, depth_path, bbox, intrinsics)

    # Baseline giả lập để test PyBullet trước
    grasp_candidates = [
        {
            "translation": [0.4, 0.0, 0.4],
            "rotation_matrix": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ],
            "width": 0.05,
            "score": 0.9
        }
    ]

    print("[4] Selecting best grasp...")
    best_grasp = select_best_grasp(
        grasp_candidates=grasp_candidates,
        workspace=workspace,
        table_z=0.0
    )

    if best_grasp is None:
        print("No valid grasp found.")
        return

    print("Best grasp:", best_grasp)

    print("[5] Transforming camera frame to world frame...")
    T_world_camera = np.array([
        [1, 0, 0, 0.0],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1.0],
    ], dtype=np.float32)

    grasp_world = transform_grasp_camera_to_world(best_grasp, T_world_camera)

    print("[6] Running PyBullet execution...")
    controller = PyBulletRobotController(gui=True)
    controller.connect()
    controller.load_scene()
    controller.move_to_grasp(grasp_world)

    input("Press Enter to exit...")
    controller.disconnect()


if __name__ == "__main__":
    main()
```

### 19.2. Chạy full pipeline

```bash
python src/main.py
```

Kết quả baseline ban đầu:

```text
- VLM trả về bbox.
- Point cloud được tạo.
- Grasp pose được chọn.
- Robot trong PyBullet di chuyển đến pose.
```

---

## 20. Thứ tự triển khai khuyến nghị

Không nên làm tất cả cùng lúc. Nên làm theo thứ tự sau:

### Giai đoạn 1 – Chạy GraspNet độc lập

```text
Mục tiêu: chứng minh GraspNet chạy được.
```

Checklist:

- [ ] Clone `graspnet-baseline`
- [ ] Cài requirements
- [ ] Build `pointnet2`
- [ ] Build `knn`
- [ ] Cài `graspnetAPI`
- [ ] Tải checkpoint
- [ ] Chạy được `demo.py`
- [ ] Visualize được grasp candidates

### Giai đoạn 2 – Chạy VLM độc lập

```text
Mục tiêu: VLM hiểu lệnh và xác định đúng vật thể.
```

Checklist:

- [ ] Cài `google-genai`
- [ ] Tạo API key
- [ ] Gửi ảnh RGB + câu lệnh
- [ ] Nhận JSON hợp lệ
- [ ] Parse được `bbox`
- [ ] Vẽ bbox lên ảnh để kiểm tra

### Giai đoạn 3 – Tạo point cloud từ bbox

```text
Mục tiêu: chỉ lấy point cloud của vật thể mục tiêu.
```

Checklist:

- [ ] Đọc RGB image
- [ ] Đọc Depth image
- [ ] Đọc camera intrinsics
- [ ] Crop theo bbox
- [ ] Chuyển depth sang point cloud
- [ ] Visualize point cloud bằng Open3D

### Giai đoạn 4 – Tích hợp GraspNet với dữ liệu của mình

```text
Mục tiêu: GraspNet dự đoán grasp trên point cloud/cảnh của mình.
```

Checklist:

- [ ] Sửa `demo.py` để dùng RGB-D của mình
- [ ] Đưa đúng camera intrinsics
- [ ] Kiểm tra depth scale
- [ ] Sinh grasp candidates
- [ ] Lọc grasp theo score
- [ ] Xuất best grasp pose

### Giai đoạn 5 – PyBullet IK

```text
Mục tiêu: Robot mô phỏng di chuyển đến pose bất kỳ.
```

Checklist:

- [ ] Load robot URDF
- [ ] Xác định end-effector link index
- [ ] Test `calculateInverseKinematics`
- [ ] Di chuyển joint bằng `POSITION_CONTROL`
- [ ] Test nhiều target pose khác nhau

### Giai đoạn 6 – Full pipeline

```text
Mục tiêu: nối các mô-đun lại với nhau.
```

Checklist:

- [ ] User command
- [ ] VLM bbox
- [ ] RGB-D crop
- [ ] Point cloud
- [ ] GraspNet inference
- [ ] Grasp selection
- [ ] Coordinate transform
- [ ] PyBullet motion
- [ ] Pick-and-place simulation

---

## 21. Tiêu chí đánh giá baseline

### 21.1. Đánh giá VLM

| Tiêu chí | Cách đo |
|---|---|
| Chọn đúng vật thể | So sánh với nhãn thủ công |
| Bbox hợp lý | IoU hoặc kiểm tra trực quan |
| JSON ổn định | Tỷ lệ output parse được |
| Xử lý mô tả phức tạp | Test màu sắc, vị trí, chức năng |

Ví dụ test:

```text
1. "gắp cái cốc màu xanh"
2. "gắp vật nằm bên trái cái bát"
3. "gắp vật dùng để uống nước"
4. "gắp khối màu đỏ gần camera nhất"
```

### 21.2. Đánh giá GraspNet

| Tiêu chí | Cách đo |
|---|---|
| Có sinh grasp không | số lượng grasp candidates |
| Grasp có hợp lý không | visualize bằng Open3D |
| Score cao không | trung bình score top-k |
| Có va chạm không | kiểm tra collision hoặc quan sát mô phỏng |

### 21.3. Đánh giá PyBullet

| Tiêu chí | Cách đo |
|---|---|
| IK tìm được nghiệm | robot đến gần target |
| Sai số vị trí | khoảng cách end-effector với target |
| Sai số hướng | chênh lệch quaternion/euler |
| Chuyển động mượt | không giật, không vượt giới hạn khớp |
| Gắp-thả thành công | vật thể được nâng và thả đúng vị trí |

### 21.4. Đánh giá toàn hệ thống

| Tiêu chí | Cách đo |
|---|---|
| Task success rate | số lần gắp thành công / tổng số lần thử |
| Semantic success | chọn đúng vật thể theo câu lệnh |
| Grasp success | vật không bị rơi khi nâng |
| Execution success | robot không va chạm, không lỗi IK |

---

## 22. Gợi ý thí nghiệm cho báo cáo NCKH

### 22.1. Thí nghiệm 1 – Nhận thức ngữ nghĩa

Mục tiêu:

```text
Đánh giá khả năng VLM chọn đúng vật thể theo ngôn ngữ tự nhiên.
```

Thiết lập:

```text
- 10 cảnh đơn giản
- Mỗi cảnh có 3–5 vật thể
- Các câu lệnh có màu sắc, vị trí, tên vật thể
```

Ví dụ bảng kết quả:

| STT | Câu lệnh | Vật thể đúng | VLM chọn | Đúng/Sai |
|---|---|---|---|---|
| 1 | Gắp cốc xanh | cốc xanh | cốc xanh | Đúng |
| 2 | Gắp hộp đỏ | hộp đỏ | hộp đỏ | Đúng |

### 22.2. Thí nghiệm 2 – Dự đoán tư thế gắp

Mục tiêu:

```text
Đánh giá GraspNet có tạo được tư thế gắp hợp lý với vật thể mục tiêu không.
```

Chỉ số:

```text
- Số grasp candidates
- Score top-1
- Score top-5
- Quan sát trực quan bằng Open3D
```

### 22.3. Thí nghiệm 3 – Điều khiển robot mô phỏng

Mục tiêu:

```text
Đánh giá robot có di chuyển đến pose gắp được không.
```

Chỉ số:

```text
- Sai số vị trí end-effector
- Sai số hướng
- Tỷ lệ IK thành công
```

### 22.4. Thí nghiệm 4 – Pipeline end-to-end

Mục tiêu:

```text
Đánh giá toàn bộ hệ thống từ câu lệnh đến gắp-thả.
```

Chỉ số:

```text
- Tỷ lệ chọn đúng vật thể
- Tỷ lệ gắp thành công
- Tỷ lệ thả đúng vị trí
- Thời gian xử lý trung bình
```

---

## 23. Các lỗi thường gặp và cách xử lý

### 23.1. Lỗi build `pointnet2`

Biểu hiện:

```text
error: command 'gcc' failed
nvcc not found
CUDA_HOME is not set
```

Cách xử lý:

```bash
which nvcc
echo $CUDA_HOME
nvidia-smi
```

Thêm biến môi trường nếu cần:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 23.2. Lỗi PyTorch không nhận CUDA

Kiểm tra:

```bash
python - << 'PY'
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

Nếu trả về `False`, cần cài lại PyTorch đúng CUDA.

### 23.3. Lỗi Open3D không mở cửa sổ

Nếu chạy trên server không có GUI:

```text
Open3D không visualize được.
```

Cách xử lý:

```text
- Chạy local có màn hình.
- Hoặc lưu point cloud ra file `.ply`.
- Hoặc dùng offscreen rendering.
```

### 23.4. VLM trả về không đúng JSON

Cách xử lý:

```text
- Bắt buộc prompt ghi “Return JSON only”.
- Dùng `response_mime_type="application/json"`.
- Validate output bằng `json.loads`.
- Nếu lỗi, gọi lại model một lần.
```

### 23.5. Bbox lệch

Nguyên nhân:

```text
- VLM trả bbox theo ảnh resize thay vì ảnh gốc.
- Prompt chưa nói rõ dùng pixel ảnh gốc.
- Ảnh gửi lên đã bị resize.
```

Cách xử lý:

```text
- Ghi rõ trong prompt: bbox theo ảnh gốc.
- Không resize ảnh trước khi gửi.
- Vẽ bbox để kiểm tra.
```

### 23.6. Point cloud bị méo

Nguyên nhân:

```text
- Sai camera intrinsics.
- Sai depth scale.
- Depth image bị mất dữ liệu.
```

Cách xử lý:

```text
- Kiểm tra `fx`, `fy`, `cx`, `cy`.
- Kiểm tra depth đang là mm hay meter.
- Visualize point cloud toàn cảnh trước khi crop.
```

### 23.7. Robot không tới được grasp pose

Nguyên nhân:

```text
- Pose nằm ngoài workspace.
- Sai hệ tọa độ camera/world.
- Sai orientation của end-effector.
- IK ra nghiệm không ổn định.
```

Cách xử lý:

```text
- Test với dummy pose trước.
- Giới hạn workspace.
- Thêm pre-grasp pose phía trên vật thể.
- Kiểm tra transform camera-to-world.
```

---

## 24. Lộ trình hoàn thành trong báo cáo

### 24.1. Mục tiêu tuần 1

```text
- Cài môi trường.
- Chạy GraspNet demo.
- Ghi lại ảnh kết quả visualize grasp.
```

Sản phẩm:

```text
- Ảnh chụp demo GraspNet
- Log cài đặt
- Mô tả môi trường
```

### 24.2. Mục tiêu tuần 2

```text
- Xây dựng mô-đun VLM.
- Test trên 10 ảnh đơn giản.
- Vẽ bbox kết quả.
```

Sản phẩm:

```text
- File `vlm_localizer.py`
- Bảng test câu lệnh
- Ảnh bbox
```

### 24.3. Mục tiêu tuần 3

```text
- Chuyển RGB-D sang point cloud.
- Crop point cloud bằng bbox từ VLM.
- Visualize point cloud vật thể.
```

Sản phẩm:

```text
- File `rgbd_to_pointcloud.py`
- File `.ply`
- Ảnh visualize point cloud
```

### 24.4. Mục tiêu tuần 4

```text
- Tích hợp dữ liệu của mình vào GraspNet.
- Sinh grasp candidates.
- Chọn best grasp.
```

Sản phẩm:

```text
- File `graspnet_adapter.py`
- Ảnh visualize top-k grasps
- Log score
```

### 24.5. Mục tiêu tuần 5

```text
- Thiết lập PyBullet.
- Load robot 6DOF.
- Test IK với dummy pose.
```

Sản phẩm:

```text
- File `pybullet_controller.py`
- Video robot di chuyển đến target pose
```

### 24.6. Mục tiêu tuần 6

```text
- Nối full pipeline.
- Demo end-to-end trong mô phỏng.
- Ghi lại kết quả thử nghiệm.
```

Sản phẩm:

```text
- File `main.py`
- Video demo end-to-end
- Bảng đánh giá baseline
```

---

## 25. Nội dung nên viết trong báo cáo NCKH

### 25.1. Cách mô tả GraspNet trong báo cáo

Có thể viết:

> Trong đề tài, nhóm sử dụng GraspNet Baseline làm mô-đun dự đoán tư thế gắp 6DoF từ dữ liệu point cloud RGB-D. Thay vì huấn luyện lại mô hình từ đầu, baseline sử dụng checkpoint huấn luyện sẵn để sinh các ứng viên tư thế gắp. Các ứng viên này được lọc theo điểm tin cậy, vùng thao tác của robot và điều kiện tránh va chạm trước khi truyền sang mô-đun điều khiển chuyển động.

### 25.2. Cách mô tả VLM trong báo cáo

Có thể viết:

> Vision-Language Model được sử dụng để liên kết câu lệnh ngôn ngữ tự nhiên với vùng ảnh chứa vật thể mục tiêu. Mô-đun này cho phép hệ thống hiểu các mô tả như màu sắc, vị trí tương đối hoặc chức năng của vật thể, thay vì chỉ dựa trên nhãn cố định.

### 25.3. Cách mô tả PyBullet trong báo cáo

Có thể viết:

> PyBullet được sử dụng để mô phỏng môi trường robot 6DOF và giải bài toán động học ngược. Tư thế gắp đầu ra từ GraspNet được chuyển đổi sang hệ tọa độ robot, sau đó bộ giải IK tính toán các góc khớp tương ứng để robot di chuyển end-effector tới vị trí gắp.

### 25.4. Cách mô tả điểm mới

Có thể viết:

> Điểm mới của đề tài nằm ở việc tích hợp nhận thức ngữ nghĩa bằng Vision-Language Model, nhận thức hình học 3D bằng GraspNet và điều khiển chuyển động bằng PyBullet trong một pipeline thống nhất cho bài toán gắp thả vật thể theo ngôn ngữ tự nhiên. Hệ thống cho phép robot lựa chọn vật thể dựa trên ý nghĩa câu lệnh, sau đó tự động suy luận tư thế gắp và thực hiện hành động trong môi trường mô phỏng.

---

## 26. Checklist hoàn thành baseline

### 26.1. Cài đặt

- [ ] Tạo môi trường Conda
- [ ] Cài PyTorch
- [ ] Cài Open3D
- [ ] Clone GraspNet Baseline
- [ ] Build PointNet2
- [ ] Build KNN
- [ ] Cài GraspNet API
- [ ] Tải checkpoint
- [ ] Cài Google GenAI SDK
- [ ] Cài PyBullet

### 26.2. Mô-đun VLM

- [ ] Nhận ảnh RGB
- [ ] Nhận câu lệnh người dùng
- [ ] Trả JSON hợp lệ
- [ ] Có `bbox`
- [ ] Có `target_object`
- [ ] Có `confidence`
- [ ] Vẽ bbox kiểm tra

### 26.3. Mô-đun RGB-D

- [ ] Đọc depth image
- [ ] Đổi depth sang mét
- [ ] Đọc camera intrinsics
- [ ] Crop theo bbox
- [ ] Tạo point cloud
- [ ] Lưu `.ply`
- [ ] Visualize bằng Open3D

### 26.4. Mô-đun GraspNet

- [ ] Chạy demo gốc
- [ ] Load checkpoint
- [ ] Dùng dữ liệu RGB-D của mình
- [ ] Sinh grasp candidates
- [ ] Lọc theo score
- [ ] Chọn best grasp
- [ ] Xuất pose 6DoF

### 26.5. Mô-đun PyBullet

- [ ] Load scene
- [ ] Load robot URDF
- [ ] Xác định end-effector link
- [ ] Tính IK
- [ ] Điều khiển joint
- [ ] Test pre-grasp pose
- [ ] Test grasp pose
- [ ] Test place pose

### 26.6. Full pipeline

- [ ] Nhập câu lệnh
- [ ] VLM tìm vật thể
- [ ] Tạo point cloud
- [ ] GraspNet dự đoán grasp
- [ ] Chọn grasp tốt nhất
- [ ] Transform pose
- [ ] PyBullet thực thi
- [ ] Lưu log kết quả
- [ ] Quay video demo

---

## 27. Kết luận baseline

Baseline cần chứng minh được 3 năng lực chính:

```text
1. Robot hiểu yêu cầu ngữ nghĩa của người dùng thông qua VLM.
2. Robot tìm được tư thế gắp 6DoF bằng GraspNet từ dữ liệu RGB-D.
3. Robot thực thi chuyển động gắp-thả trong PyBullet bằng động học ngược.
```

Khi hoàn thành baseline, đề tài đã có nền tảng đủ mạnh để phát triển tiếp:

```text
- Tối ưu chọn grasp.
- Cải thiện calibration.
- Thêm nhiều loại câu lệnh.
- Chạy trên camera thật.
- Chạy trên robot thật.
- So sánh với các phương pháp không dùng VLM.
```

---

## 28. Tài liệu tham khảo kỹ thuật

- GraspNet Baseline: https://github.com/graspnet/graspnet-baseline
- GraspNet API: https://github.com/graspnet/graspnetAPI
- GraspNet API documentation: https://graspnetapi.readthedocs.io/
- Google GenAI Python SDK: https://googleapis.github.io/python-genai/
- Gemini API documentation: https://ai.google.dev/gemini-api/docs
- PyBullet: https://pybullet.org/
- PyBullet Quickstart Guide: https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
