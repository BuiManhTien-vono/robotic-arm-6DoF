# BASELINE TRIỂN KHAI HỆ THỐNG ROBOTIC SEMANTIC GRASPING 6DOF  
## Qwen2.5-VL Local + GraspNet Baseline + PyBullet

**Mục tiêu:** xây dựng một pipeline baseline hoàn chỉnh cho bài toán **gắp thả vật thể theo ngữ nghĩa** bằng robot 6DOF, không phụ thuộc API key.  
**Phạm vi:** hệ thống sử dụng **Qwen2.5-VL chạy cục bộ** để hiểu ảnh và câu lệnh, **GraspNet Baseline** để dự đoán tư thế gắp 6DoF từ point cloud, và **PyBullet** để mô phỏng/điều khiển robot bằng động học ngược.

---

# 1. Ý tưởng tổng thể

Bài toán cần giải quyết:

> Người dùng nhập câu lệnh tự nhiên như:  
> **"Gắp cái cốc màu xanh bên trái"**  
> Robot phải xác định đúng vật thể, tìm tư thế gắp ổn định, sau đó thực hiện gắp thả.

Pipeline baseline:

```text
User Command
    ↓
RGB Image
    ↓
Qwen2.5-VL Local
    ↓
Target Object + 2D Bounding Box
    ↓
RGB-D Processing
    ↓
Object Point Cloud
    ↓
GraspNet Baseline
    ↓
Best 6DoF Grasp Pose
    ↓
Camera-to-Robot Transform
    ↓
PyBullet Inverse Kinematics
    ↓
Robot 6DOF Pick-and-Place
```

---

# 2. Thành phần chính của hệ thống

| Thành phần | Công nghệ đề xuất | Vai trò |
|---|---|---|
| Nhận thức ngữ nghĩa | Qwen2.5-VL-7B-Instruct hoặc Qwen2.5-VL-3B-Instruct | Hiểu ảnh RGB + câu lệnh, trả về bounding box vật thể |
| Nhận thức không gian 3D | RGB-D camera / depth map | Chuyển vùng ảnh 2D thành point cloud |
| Dự đoán tư thế gắp | GraspNet Baseline | Sinh các tư thế gắp 6DoF |
| Lọc grasp | Confidence score, collision check, workspace filter | Chọn tư thế gắp phù hợp nhất |
| Mô phỏng robot | PyBullet | Load URDF robot, giải IK, điều khiển khớp |
| Robot | Robot 6DOF mô phỏng hoặc robot thật | Thực hiện pick-and-place |

---

# 3. Vì sao chọn Qwen2.5-VL local?

Trong đề tài này, VLM cần làm tốt tác vụ:

```text
Ảnh RGB + câu lệnh tự nhiên → xác định vật thể mục tiêu + bbox
```

Qwen2.5-VL phù hợp vì:

- có thể chạy cục bộ, không cần API key;
- hỗ trợ image-text reasoning;
- có khả năng visual localization, tức định vị vật thể bằng bounding box hoặc point;
- có thể ép đầu ra dạng JSON để dễ nối với pipeline xử lý sau;
- có bản 3B nhẹ hơn và bản 7B cân bằng tốt hơn giữa hiệu năng và độ chính xác.

Khuyến nghị:

```text
Máy GPU yếu / VRAM thấp: Qwen2.5-VL-3B-Instruct
Máy GPU tốt hơn: Qwen2.5-VL-7B-Instruct
```

Trong baseline NCKH, nên bắt đầu với **Qwen2.5-VL-7B-Instruct** nếu máy có GPU đủ mạnh. Nếu lỗi bộ nhớ, chuyển xuống **Qwen2.5-VL-3B-Instruct** hoặc dùng quantization 4-bit.

---

# 4. Kiến trúc thư mục đề xuất

Tạo project như sau:

```text
semantic_grasping_6dof/
│
├── README.md
├── requirements_common.txt
│
├── configs/
│   ├── camera_intrinsics.json
│   ├── hand_eye_calibration.json
│   ├── robot_config.json
│   └── vlm_config.json
│
├── data/
│   ├── rgb/
│   │   └── scene_001.jpg
│   ├── depth/
│   │   └── scene_001.png
│   ├── pointcloud/
│   │   └── object_cloud_001.ply
│   └── outputs/
│       ├── vlm_result.json
│       ├── grasp_candidates.json
│       └── selected_grasp.json
│
├── external/
│   ├── graspnet-baseline/
│   └── graspnetAPI/
│
├── src/
│   ├── 01_vlm_localize.py
│   ├── 02_rgbd_to_pointcloud.py
│   ├── 03_graspnet_infer.py
│   ├── 04_select_grasp.py
│   ├── 05_transform_pose.py
│   ├── 06_pybullet_execute.py
│   └── run_pipeline.py
│
├── scripts/
│   ├── setup_vlm_env.sh
│   ├── setup_graspnet_env.sh
│   └── run_demo.sh
│
└── docs/
    ├── baseline_pipeline.md
    ├── experiment_log.md
    └── error_notes.md
```

---

# 5. Lưu ý quan trọng về môi trường

## 5.1. Không nên ép tất cả vào một môi trường duy nhất

Qwen2.5-VL thường cần môi trường `transformers` mới, còn `graspnet-baseline` gốc dùng stack cũ hơn, đặc biệt có các extension C++/CUDA như `pointnet2` và `knn`.

Do đó baseline nên chia thành 2 môi trường:

```text
env_vlm       → chạy Qwen2.5-VL local
env_graspnet  → chạy GraspNet Baseline
```

Hai môi trường trao đổi dữ liệu qua file JSON/PLY:

```text
VLM output: bbox JSON
RGB-D module output: object_cloud.ply
GraspNet output: grasp pose JSON
PyBullet input: selected_grasp.json
```

Cách này dễ debug hơn, tránh lỗi xung đột PyTorch/CUDA.

---

# 6. Bước 1 — Tạo môi trường cho VLM local

## 6.1. Tạo môi trường Conda

```bash
conda create -n env_vlm python=3.10 -y
conda activate env_vlm
```

## 6.2. Cài PyTorch

Chọn bản PyTorch phù hợp với CUDA máy của bạn. Ví dụ với CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Nếu máy không có GPU NVIDIA, vẫn có thể chạy CPU nhưng rất chậm:

```bash
pip install torch torchvision torchaudio
```

## 6.3. Cài thư viện Qwen2.5-VL

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils
pip install pillow opencv-python numpy
```

Nếu muốn chạy video hoặc tối ưu xử lý visual input:

```bash
pip install qwen-vl-utils[decord]
```

Nếu VRAM thấp và muốn thử quantization:

```bash
pip install bitsandbytes
```

---

# 7. Bước 2 — Test Qwen2.5-VL local

Tạo file:

```text
src/01_vlm_localize.py
```

Nội dung baseline:

```python
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def extract_json(text: str) -> dict:
    """
    Cố gắng trích JSON từ output của VLM.
    VLM đôi khi trả thêm text ngoài JSON, nên cần regex.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Không tìm thấy JSON trong output:\n{text}")

    return json.loads(match.group(0))


def build_prompt(command: str) -> str:
    return f"""
You are the perception module of a robotic semantic grasping system.

Task:
Given an RGB image and a user command, identify the target object that best matches the command.

User command:
"{command}"

Return only valid JSON. Do not include markdown. Do not explain outside JSON.

Required JSON format:
{{
  "target_object": "short object name",
  "bbox_2d": [x_min, y_min, x_max, y_max],
  "confidence": 0.0,
  "reason": "brief reason"
}}

Rules:
- bbox_2d must be pixel coordinates in the original image.
- Use [x_min, y_min, x_max, y_max].
- If the target is uncertain, still return the most likely object with lower confidence.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh RGB")
    parser.add_argument("--command", required=True, help="Câu lệnh người dùng")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", default="data/outputs/vlm_result.json")
    args = parser.parse_args()

    image_path = str(Path(args.image).resolve())
    prompt = build_prompt(args.command)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("Raw VLM output:")
    print(output_text)

    result = extract_json(output_text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved VLM result to: {output_path}")


if __name__ == "__main__":
    main()
```

Chạy thử:

```bash
conda activate env_vlm

python src/01_vlm_localize.py \
  --image data/rgb/scene_001.jpg \
  --command "Gắp cái cốc màu xanh bên trái" \
  --output data/outputs/vlm_result.json
```

Output kỳ vọng:

```json
{
  "target_object": "blue cup",
  "bbox_2d": [120, 80, 260, 310],
  "confidence": 0.86,
  "reason": "The blue cup is on the left side of the table."
}
```

---

# 8. Bước 3 — Tạo môi trường cho GraspNet Baseline

## 8.1. Tạo môi trường

```bash
conda create -n env_graspnet python=3.8 -y
conda activate env_graspnet
```

> Ghi chú: repo GraspNet Baseline gốc khá cũ. Nếu gặp lỗi với Python/PyTorch mới, nên dùng Python 3.8 hoặc 3.9 trước để giảm rủi ro.

## 8.2. Clone GraspNet Baseline

```bash
mkdir -p external
cd external

git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```

## 8.3. Cài requirements

```bash
pip install -r requirements.txt
```

## 8.4. Build PointNet2

```bash
cd pointnet2
python setup.py install
cd ..
```

## 8.5. Build KNN

```bash
cd knn
python setup.py install
cd ..
```

## 8.6. Cài GraspNet API nếu cần evaluation

```bash
cd ..
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

---

# 9. Bước 4 — Tải checkpoint GraspNet

GraspNet Baseline cung cấp checkpoint huấn luyện sẵn:

```text
checkpoint-rs.tar  → model cho RealSense
checkpoint-kn.tar  → model cho Kinect
```

Khuyến nghị dùng:

```text
checkpoint-rs.tar
```

vì RealSense thường gần với dữ liệu RGB-D phổ biến trong demo robot.

Tạo thư mục:

```bash
mkdir -p external/graspnet-baseline/checkpoints
```

Đặt checkpoint vào:

```text
external/graspnet-baseline/checkpoints/checkpoint-rs.tar
```

---

# 10. Bước 5 — Chuyển bbox + RGB-D thành point cloud vật thể

Sau khi VLM trả:

```json
{
  "bbox_2d": [x_min, y_min, x_max, y_max]
}
```

Ta dùng bbox này để lấy vùng depth tương ứng và chuyển sang point cloud.

Công thức từ pixel sang 3D:

```text
Z = depth(u, v) / depth_scale
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Trong đó:

```text
fx, fy, cx, cy  → camera intrinsics
depth_scale     → hệ số chuyển depth raw sang mét
```

Tạo file cấu hình:

```text
configs/camera_intrinsics.json
```

Ví dụ:

```json
{
  "fx": 615.0,
  "fy": 615.0,
  "cx": 320.0,
  "cy": 240.0,
  "depth_scale": 1000.0,
  "width": 640,
  "height": 480
}
```

Tạo file:

```text
src/02_rgbd_to_pointcloud.py
```

Nội dung baseline:

```python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_to_pointcloud(depth, bbox, intrinsics):
    x_min, y_min, x_max, y_max = bbox

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    depth_scale = intrinsics["depth_scale"]

    h, w = depth.shape

    x_min = max(0, min(w - 1, int(x_min)))
    x_max = max(0, min(w - 1, int(x_max)))
    y_min = max(0, min(h - 1, int(y_min)))
    y_max = max(0, min(h - 1, int(y_max)))

    points = []

    for v in range(y_min, y_max):
        for u in range(x_min, x_max):
            z = float(depth[v, u]) / depth_scale

            if z <= 0.0 or z > 2.0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])

    if len(points) == 0:
        raise ValueError("Không tạo được point cloud: bbox hoặc depth không hợp lệ.")

    return np.asarray(points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True, help="Depth image, ví dụ PNG 16-bit")
    parser.add_argument("--vlm-json", required=True, help="Output JSON từ VLM")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON")
    parser.add_argument("--output", default="data/pointcloud/object_cloud.ply")
    args = parser.parse_args()

    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Không đọc được depth image: {args.depth}")

    vlm_result = load_json(args.vlm_json)
    intrinsics = load_json(args.intrinsics)

    bbox = vlm_result["bbox_2d"]
    points = bbox_to_pointcloud(depth, bbox, intrinsics)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample để giảm nhiễu và giảm số điểm
    pcd = pcd.voxel_down_sample(voxel_size=0.003)

    # Lọc outlier cơ bản
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(output_path), pcd)

    print(f"Saved object point cloud to: {output_path}")
    print(f"Number of points: {len(pcd.points)}")


if __name__ == "__main__":
    main()
```

Chạy thử:

```bash
conda activate env_graspnet

python src/02_rgbd_to_pointcloud.py \
  --depth data/depth/scene_001.png \
  --vlm-json data/outputs/vlm_result.json \
  --intrinsics configs/camera_intrinsics.json \
  --output data/pointcloud/object_cloud_001.ply
```

---

# 11. Bước 6 — Tích hợp object point cloud với GraspNet

Có 2 cách triển khai baseline.

## Cách A — Sửa trực tiếp `demo.py` của GraspNet

Trong repo:

```text
external/graspnet-baseline/demo.py
```

Tìm hàm:

```python
get_and_process_data()
```

Mục tiêu là thay phần đọc dữ liệu demo bằng dữ liệu RGB-D/point cloud của mình.

Baseline chỉnh logic:

```text
Input:
- RGB image
- Depth image
- Camera intrinsics
- Object bbox từ VLM

Process:
- crop vùng object point cloud
- đưa point cloud vào GraspNet
- lấy danh sách grasp candidates
```

Cách này nhanh nhưng code dễ bị phụ thuộc vào cấu trúc repo gốc.

## Cách B — Viết adapter riêng

Khuyến nghị cho báo cáo NCKH:

```text
src/03_graspnet_infer.py
```

File này đóng vai trò adapter giữa point cloud của hệ thống và model GraspNet.

Pseudocode:

```python
"""
Pseudocode cho GraspNet inference adapter.

Do repo GraspNet Baseline có cấu trúc riêng, cần tham chiếu demo.py gốc
để import đúng class và hàm xử lý.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_object_pointcloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points).astype(np.float32)
    return points


def run_graspnet(points, checkpoint_path):
    """
    TODO:
    - Load GraspNet model.
    - Load checkpoint.
    - Chuẩn hóa point cloud đúng format mà demo.py yêu cầu.
    - Forward model.
    - Decode grasp candidates.
    - Collision detection nếu cần.
    """

    # Output giả lập để định nghĩa format chuẩn của hệ thống.
    # Khi tích hợp thật, thay phần này bằng output từ GraspNet.
    grasp_candidates = [
        {
            "translation": [0.42, 0.05, 0.18],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            "width": 0.06,
            "score": 0.91
        }
    ]

    return grasp_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="data/outputs/grasp_candidates.json")
    args = parser.parse_args()

    points = load_object_pointcloud(args.cloud)
    grasps = run_graspnet(points, args.checkpoint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grasps, f, ensure_ascii=False, indent=2)

    print(f"Saved grasp candidates to: {output_path}")


if __name__ == "__main__":
    main()
```

Lưu ý khi tích hợp thật:

```text
Không nên tự viết lại toàn bộ GraspNet.
Nên dựa vào demo.py gốc của graspnet-baseline để đảm bảo đúng format input/output.
```

---

# 12. Bước 7 — Chuẩn hóa output grasp pose

Mỗi grasp candidate nên được chuẩn hóa về JSON:

```json
{
  "translation": [x, y, z],
  "rotation_matrix": [
    [r11, r12, r13],
    [r21, r22, r23],
    [r31, r32, r33]
  ],
  "width": 0.06,
  "score": 0.91
}
```

Ý nghĩa:

| Trường | Ý nghĩa |
|---|---|
| translation | Tọa độ điểm gắp trong hệ camera |
| rotation_matrix | Hướng của gripper |
| width | Độ mở kẹp |
| score | Độ tin cậy của tư thế gắp |

---

# 13. Bước 8 — Chọn grasp tốt nhất

Tạo file:

```text
src/04_select_grasp.py
```

Baseline:

```python
import argparse
import json
from pathlib import Path


def is_valid_grasp(g):
    """
    Lọc cơ bản:
    - score đủ cao
    - z không âm
    - width trong khoảng hợp lý
    """
    score = g.get("score", 0.0)
    width = g.get("width", 0.0)
    z = g.get("translation", [0, 0, -1])[2]

    if score < 0.3:
        return False

    if z <= 0.0:
        return False

    if width < 0.005 or width > 0.12:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasps", required=True)
    parser.add_argument("--output", default="data/outputs/selected_grasp.json")
    args = parser.parse_args()

    with open(args.grasps, "r", encoding="utf-8") as f:
        grasps = json.load(f)

    valid_grasps = [g for g in grasps if is_valid_grasp(g)]

    if not valid_grasps:
        raise ValueError("Không có grasp hợp lệ sau khi lọc.")

    best = max(valid_grasps, key=lambda g: g["score"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("Selected best grasp:")
    print(json.dumps(best, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

Chạy:

```bash
python src/04_select_grasp.py \
  --grasps data/outputs/grasp_candidates.json \
  --output data/outputs/selected_grasp.json
```

---

# 14. Bước 9 — Chuyển hệ tọa độ camera sang robot

GraspNet thường trả pose trong hệ tọa độ camera:

```text
T_camera_grasp
```

Robot cần pose trong hệ tọa độ base:

```text
T_robot_grasp
```

Cần calibration:

```text
T_robot_camera
```

Công thức:

```text
T_robot_grasp = T_robot_camera × T_camera_grasp
```

Tạo file:

```text
configs/hand_eye_calibration.json
```

Ví dụ:

```json
{
  "T_robot_camera": [
    [1.0, 0.0, 0.0, 0.30],
    [0.0, 1.0, 0.0, 0.00],
    [0.0, 0.0, 1.0, 0.50],
    [0.0, 0.0, 0.0, 1.00]
  ]
}
```

Tạo file:

```text
src/05_transform_pose.py
```

Baseline:

```python
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def make_transform(rotation_matrix, translation):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output", default="data/outputs/robot_grasp_pose.json")
    args = parser.parse_args()

    with open(args.grasp, "r", encoding="utf-8") as f:
        grasp = json.load(f)

    with open(args.calib, "r", encoding="utf-8") as f:
        calib = json.load(f)

    T_robot_camera = np.asarray(calib["T_robot_camera"], dtype=np.float64)

    T_camera_grasp = make_transform(
        grasp["rotation_matrix"],
        grasp["translation"]
    )

    T_robot_grasp = T_robot_camera @ T_camera_grasp

    position = T_robot_grasp[:3, 3].tolist()
    quat_xyzw = R.from_matrix(T_robot_grasp[:3, :3]).as_quat().tolist()

    output = {
        "position": position,
        "orientation_quat_xyzw": quat_xyzw,
        "width": grasp.get("width", 0.06),
        "score": grasp.get("score", 0.0),
        "T_robot_grasp": T_robot_grasp.tolist()
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved robot grasp pose to: {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
```

Chạy:

```bash
python src/05_transform_pose.py \
  --grasp data/outputs/selected_grasp.json \
  --calib configs/hand_eye_calibration.json \
  --output data/outputs/robot_grasp_pose.json
```

---

# 15. Bước 10 — Thực thi bằng PyBullet

Tạo file cấu hình robot:

```text
configs/robot_config.json
```

Ví dụ với robot KUKA demo trong PyBullet:

```json
{
  "robot_urdf": "kuka_iiwa/model.urdf",
  "base_position": [0, 0, 0],
  "base_orientation_euler": [0, 0, 0],
  "end_effector_link_index": 6,
  "arm_joint_indices": [0, 1, 2, 3, 4, 5, 6],
  "gripper_open": 0.08,
  "gripper_close": 0.01
}
```

Tạo file:

```text
src/06_pybullet_execute.py
```

Baseline:

```python
import argparse
import json
import time

import pybullet as p
import pybullet_data


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def move_to_pose(robot_id, ee_link_index, joint_indices, target_pos, target_orn, steps=240):
    joint_positions = p.calculateInverseKinematics(
        robot_id,
        ee_link_index,
        target_pos,
        target_orn
    )

    for _ in range(steps):
        for i, joint_idx in enumerate(joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, help="robot_grasp_pose.json")
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    pose = load_json(args.pose)
    cfg = load_json(args.robot_config)

    if args.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    base_pos = cfg.get("base_position", [0, 0, 0])
    base_euler = cfg.get("base_orientation_euler", [0, 0, 0])
    base_orn = p.getQuaternionFromEuler(base_euler)

    robot_id = p.loadURDF(
        cfg["robot_urdf"],
        base_pos,
        base_orn,
        useFixedBase=True
    )

    ee_link = cfg["end_effector_link_index"]
    joint_indices = cfg["arm_joint_indices"]

    target_pos = pose["position"]
    target_orn = pose["orientation_quat_xyzw"]

    # Tạo pre-grasp cao hơn điểm gắp để robot tiếp cận an toàn
    pre_grasp_pos = [
        target_pos[0],
        target_pos[1],
        target_pos[2] + 0.10
    ]

    print("Move to pre-grasp pose")
    move_to_pose(robot_id, ee_link, joint_indices, pre_grasp_pos, target_orn)

    print("Move down to grasp pose")
    move_to_pose(robot_id, ee_link, joint_indices, target_pos, target_orn)

    print("Close gripper - baseline placeholder")
    # TODO: nếu URDF có gripper, thêm điều khiển joint gripper tại đây.

    print("Lift object")
    lift_pos = [
        target_pos[0],
        target_pos[1],
        target_pos[2] + 0.15
    ]
    move_to_pose(robot_id, ee_link, joint_indices, lift_pos, target_orn)

    print("Done")

    if args.gui:
        time.sleep(5)

    p.disconnect()


if __name__ == "__main__":
    main()
```

Chạy:

```bash
pip install pybullet scipy

python src/06_pybullet_execute.py \
  --pose data/outputs/robot_grasp_pose.json \
  --robot-config configs/robot_config.json \
  --gui
```

---

# 16. Bước 11 — Viết script chạy toàn bộ pipeline

Tạo file:

```text
scripts/run_demo.sh
```

Nội dung:

```bash
#!/bin/bash

set -e

IMAGE="data/rgb/scene_001.jpg"
DEPTH="data/depth/scene_001.png"
COMMAND="Gắp cái cốc màu xanh bên trái"

echo "Step 1: VLM localize target object"
conda run -n env_vlm python src/01_vlm_localize.py \
  --image "$IMAGE" \
  --command "$COMMAND" \
  --output data/outputs/vlm_result.json

echo "Step 2: Convert RGB-D bbox to object point cloud"
conda run -n env_graspnet python src/02_rgbd_to_pointcloud.py \
  --depth "$DEPTH" \
  --vlm-json data/outputs/vlm_result.json \
  --intrinsics configs/camera_intrinsics.json \
  --output data/pointcloud/object_cloud_001.ply

echo "Step 3: Run GraspNet inference"
conda run -n env_graspnet python src/03_graspnet_infer.py \
  --cloud data/pointcloud/object_cloud_001.ply \
  --checkpoint external/graspnet-baseline/checkpoints/checkpoint-rs.tar \
  --output data/outputs/grasp_candidates.json

echo "Step 4: Select best grasp"
conda run -n env_graspnet python src/04_select_grasp.py \
  --grasps data/outputs/grasp_candidates.json \
  --output data/outputs/selected_grasp.json

echo "Step 5: Transform camera pose to robot base pose"
conda run -n env_graspnet python src/05_transform_pose.py \
  --grasp data/outputs/selected_grasp.json \
  --calib configs/hand_eye_calibration.json \
  --output data/outputs/robot_grasp_pose.json

echo "Step 6: Execute in PyBullet"
conda run -n env_graspnet python src/06_pybullet_execute.py \
  --pose data/outputs/robot_grasp_pose.json \
  --robot-config configs/robot_config.json \
  --gui
```

Cấp quyền chạy:

```bash
chmod +x scripts/run_demo.sh
```

Chạy demo:

```bash
./scripts/run_demo.sh
```

---

# 17. Checklist hoàn thành baseline

## Giai đoạn 1 — VLM local

- [ ] Cài được `env_vlm`.
- [ ] Load được Qwen2.5-VL-3B hoặc 7B.
- [ ] Nhập ảnh + câu lệnh và nhận được output JSON.
- [ ] JSON có trường `bbox_2d`.
- [ ] Kiểm tra bbox bằng cách vẽ lên ảnh.

## Giai đoạn 2 — RGB-D và point cloud

- [ ] Có ảnh RGB.
- [ ] Có depth image tương ứng.
- [ ] Có camera intrinsics.
- [ ] Chuyển bbox sang object point cloud.
- [ ] Mở được file `.ply` bằng Open3D.
- [ ] Point cloud chỉ chứa chủ yếu vật thể mục tiêu.

## Giai đoạn 3 — GraspNet

- [ ] Clone được `graspnet-baseline`.
- [ ] Cài được `requirements.txt`.
- [ ] Build được `pointnet2`.
- [ ] Build được `knn`.
- [ ] Tải được checkpoint.
- [ ] Chạy được demo gốc của GraspNet.
- [ ] Chỉnh được input sang dữ liệu RGB-D/point cloud của hệ thống.
- [ ] Xuất được grasp candidates.

## Giai đoạn 4 — Robot/PyBullet

- [ ] Load được URDF robot.
- [ ] Xác định đúng link end-effector.
- [ ] Giải IK đến pose mẫu.
- [ ] Robot di chuyển được đến pre-grasp.
- [ ] Robot di chuyển được đến grasp.
- [ ] Robot lift vật thể sau khi gắp.

## Giai đoạn 5 — Full pipeline

- [ ] Chạy được từ câu lệnh người dùng đến bbox.
- [ ] Chạy được từ bbox đến point cloud.
- [ ] Chạy được từ point cloud đến grasp pose.
- [ ] Chạy được từ grasp pose đến PyBullet IK.
- [ ] Có video demo hoặc ảnh minh họa từng bước.

---

# 18. Các lỗi thường gặp và cách xử lý

## 18.1. Lỗi `KeyError: 'qwen2_5_vl'`

Nguyên nhân: bản `transformers` chưa hỗ trợ Qwen2.5-VL.

Cách xử lý:

```bash
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers accelerate
```

---

## 18.2. GPU bị thiếu VRAM khi load Qwen2.5-VL-7B

Cách xử lý theo thứ tự:

```text
1. Giảm về Qwen2.5-VL-3B-Instruct.
2. Giảm max_pixels trong AutoProcessor.
3. Dùng quantization 4-bit.
4. Chạy qua vLLM/SGLang nếu có máy mạnh hơn.
```

Ví dụ giảm số visual tokens:

```python
min_pixels = 256 * 28 * 28
max_pixels = 768 * 28 * 28

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
```

---

## 18.3. Build `pointnet2` lỗi

Nguyên nhân thường gặp:

```text
- CUDA Toolkit không khớp với PyTorch.
- Không có nvcc.
- Compiler C++ không tương thích.
- PyTorch quá mới so với code gốc.
```

Kiểm tra:

```bash
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Hướng xử lý:

```text
- Dùng Python 3.8/3.9.
- Chọn PyTorch + CUDA tương thích.
- Nếu máy quá mới, cân nhắc dùng fork GraspNet đã nâng cấp cho PyTorch/CUDA mới.
```

---

## 18.4. VLM trả bbox sai

Cách xử lý:

```text
- Ép prompt trả JSON nghiêm ngặt.
- Bổ sung ví dụ few-shot trong prompt.
- Chụp ảnh rõ hơn, ít vật bị che.
- Tăng độ phân giải ảnh đầu vào.
- Thêm bước kiểm tra bbox bằng Grounding DINO hoặc SAM nếu cần.
```

---

## 18.5. Point cloud bị lẫn nền/bàn

Cách xử lý:

```text
- Co bbox nhỏ lại một chút.
- Lọc depth theo khoảng cách.
- Dùng mask segmentation thay vì bbox nếu có.
- Dùng plane removal để bỏ mặt bàn.
```

Ví dụ co bbox:

```python
def shrink_bbox(bbox, ratio=0.08):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return [
        x1 + ratio * w,
        y1 + ratio * h,
        x2 - ratio * w,
        y2 - ratio * h
    ]
```

---

## 18.6. Robot IK ra tư thế kỳ lạ

Cách xử lý:

```text
- Kiểm tra lại hệ tọa độ camera và robot.
- Kiểm tra đơn vị: mét hay milimét.
- Kiểm tra quaternion [x, y, z, w].
- Kiểm tra index end-effector link.
- Thêm joint limits vào calculateInverseKinematics.
```

---

# 19. Mốc demo tối thiểu để bảo vệ NCKH

## Demo mức 1 — Nhận thức ngữ nghĩa

Đầu vào:

```text
Ảnh bàn có nhiều vật + câu lệnh "Gắp cái cốc màu xanh"
```

Đầu ra:

```text
bbox đúng vật thể
```

Minh chứng:

```text
Ảnh có vẽ bbox + JSON output
```

---

## Demo mức 2 — Nhận thức không gian 3D

Đầu vào:

```text
bbox + depth image
```

Đầu ra:

```text
object point cloud
```

Minh chứng:

```text
file .ply hoặc ảnh Open3D của point cloud
```

---

## Demo mức 3 — Grasp pose

Đầu vào:

```text
object point cloud
```

Đầu ra:

```text
top-k grasp poses
```

Minh chứng:

```text
ảnh visualize grasp candidates
```

---

## Demo mức 4 — Robot simulation

Đầu vào:

```text
selected grasp pose
```

Đầu ra:

```text
robot di chuyển đến pre-grasp và grasp pose trong PyBullet
```

Minh chứng:

```text
video PyBullet
```

---

## Demo mức 5 — Full pipeline

Đầu vào:

```text
"Gắp cái cốc màu xanh bên trái"
```

Đầu ra:

```text
Robot mô phỏng thực hiện pick-and-place
```

Minh chứng:

```text
video toàn bộ pipeline
```

---

# 20. Nội dung có thể viết trong báo cáo NCKH

Có thể trình bày như sau:

> Đề tài xây dựng một hệ thống gắp thả vật thể theo ngữ nghĩa cho robot 6DOF dựa trên ba mô-đun chính: mô-đun nhận thức ngữ nghĩa sử dụng Vision-Language Model chạy cục bộ, mô-đun dự đoán tư thế gắp 6DoF sử dụng GraspNet Baseline, và mô-đun thực thi chuyển động sử dụng PyBullet.  
>
> Ở giai đoạn nhận thức ngữ nghĩa, Qwen2.5-VL được sử dụng để phân tích ảnh RGB kết hợp với câu lệnh ngôn ngữ tự nhiên của người dùng nhằm xác định vật thể mục tiêu và trả về bounding box 2D. Bounding box này được ánh xạ sang dữ liệu độ sâu để trích xuất point cloud của vật thể. Point cloud sau đó được đưa vào GraspNet để sinh ra các ứng viên tư thế gắp 6DoF. Tư thế có điểm tin cậy cao nhất sau khi lọc được chuyển đổi từ hệ tọa độ camera sang hệ tọa độ robot và được truyền vào PyBullet để giải bài toán động học ngược, từ đó điều khiển robot 6DOF thực hiện thao tác gắp thả.

---

# 21. Điểm mới có thể nhấn mạnh

Không nên nói rằng đề tài tự train lại VLM hoặc tự xây GraspNet từ đầu. Nên nhấn mạnh:

```text
Điểm mới nằm ở tích hợp pipeline:
VLM local → object localization → point cloud extraction → GraspNet → PyBullet IK
```

Cách viết học thuật:

> Điểm đóng góp của đề tài nằm ở việc thiết kế và triển khai một pipeline tích hợp giữa mô hình Vision-Language mã nguồn mở chạy cục bộ và mô hình dự đoán tư thế gắp 6DoF, cho phép robot lựa chọn vật thể dựa trên mô tả ngôn ngữ tự nhiên thay vì chỉ dựa trên nhãn cố định hoặc tọa độ được lập trình trước.

---

# 22. Phân biệt baseline và phần mở rộng

## Baseline bắt buộc

```text
- Qwen2.5-VL local trả bbox
- bbox + depth → object point cloud
- GraspNet sinh grasp pose
- PyBullet robot đi đến grasp pose
```

## Phần mở rộng nếu còn thời gian

```text
- Thay bbox bằng segmentation mask
- Dùng SAM để tách vật thể chính xác hơn
- Fine-tune nhẹ Qwen2.5-VL bằng LoRA
- Collision checking tốt hơn
- Thêm trajectory planning bằng MoveIt hoặc OMPL
- Điều khiển robot thật sau khi sim ổn định
```

---

# 23. Không nên làm trong giai đoạn baseline

Tránh các mục sau nếu thời gian hạn chế:

```text
- Train VLM từ đầu
- Train lại GraspNet từ đầu
- Làm robot thật ngay khi mô phỏng chưa ổn định
- Viết lại toàn bộ PointNet++/GraspNet
- Tích hợp quá nhiều model cùng lúc
```

---

# 24. Công thức pipeline tóm tắt

Ký hiệu:

```text
I_rgb      : ảnh RGB
I_depth    : ảnh độ sâu
C_user     : câu lệnh người dùng
B_target   : bounding box vật thể mục tiêu
P_object   : point cloud vật thể
G_i        : ứng viên tư thế gắp
G_best     : tư thế gắp tốt nhất
T_c_g      : pose grasp trong hệ camera
T_r_c      : transform từ robot base sang camera
T_r_g      : pose grasp trong hệ robot
Q          : vector góc khớp robot
```

Pipeline:

```text
B_target = VLM(I_rgb, C_user)

P_object = RGBD_to_PointCloud(I_depth, B_target)

{G_i} = GraspNet(P_object)

G_best = argmax score(G_i)

T_r_g = T_r_c × T_c_g

Q = IK(T_r_g)

Robot.execute(Q)
```

---

# 25. Tiêu chí đánh giá baseline

| Tiêu chí | Cách đo |
|---|---|
| Độ đúng vật thể | VLM bbox có trùng vật thể mục tiêu không |
| Độ chính xác bbox | IoU nếu có nhãn thủ công |
| Chất lượng point cloud | Quan sát bằng Open3D |
| Chất lượng grasp | Score từ GraspNet + kiểm tra va chạm |
| Khả năng IK | Robot có đến được pose không |
| Tỷ lệ demo thành công | Số lần gắp đúng / tổng số thử nghiệm |

Ví dụ bảng thử nghiệm:

| Lần thử | Câu lệnh | VLM đúng? | Point cloud tốt? | Grasp hợp lệ? | IK thành công? | Kết quả |
|---|---|---:|---:|---:|---:|---|
| 1 | Gắp cốc xanh | Có | Có | Có | Có | Thành công |
| 2 | Gắp chai nước | Có | Có | Không | - | Thất bại |
| 3 | Gắp vật đỏ bên phải | Không | - | - | - | Thất bại |

---

# 26. Tài liệu tham khảo kỹ thuật

- Qwen2.5-VL-7B-Instruct — Hugging Face: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL Blog — Qwen: https://qwenlm.github.io/blog/qwen2.5-vl/
- GraspNet Baseline — GitHub: https://github.com/graspnet/graspnet-baseline
- GraspNet API — GitHub: https://github.com/graspnet/graspnetAPI
- PyBullet Quickstart Guide — Bullet Physics: https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html

---

# 27. Kết luận triển khai

Baseline hợp lý nhất cho đề tài:

```text
Qwen2.5-VL-7B-Instruct local
        ↓
bbox vật thể mục tiêu
        ↓
RGB-D crop + point cloud extraction
        ↓
GraspNet Baseline với checkpoint-rs.tar
        ↓
best grasp pose 6DoF
        ↓
camera-to-robot transform
        ↓
PyBullet IK
        ↓
robot 6DOF pick-and-place
```

Đây là hướng triển khai vừa đủ mới về mặt nghiên cứu, vừa thực tế để hoàn thành trong phạm vi NCKH cấp đại học.
