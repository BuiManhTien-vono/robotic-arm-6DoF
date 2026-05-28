# Giải thích cách hệ thống dùng VLM

Tài liệu này giải thích riêng phần VLM trong hệ thống robot gắp vật thể. Mục tiêu là làm rõ VLM nhận dữ liệu gì, trả về gì, nó nằm ở bước nào trong pipeline, và nó khác gì với GraspNet.

Cập nhật triển khai: backend mặc định hiện là `qwen-local`, tức Qwen2.5-VL chạy trực tiếp trên máy. Gemini vẫn còn trong code như backend tùy chọn nếu gọi với `--vlm-backend gemini`.

## 1. Vai trò của VLM trong hệ thống

Trong hệ thống này, VLM không trực tiếp điều khiển robot và cũng không sinh tư thế gắp 6DoF.

VLM được dùng như một mô-đun hiểu ngữ nghĩa từ ảnh và câu lệnh:

```text
Ảnh RGB của scene + câu lệnh người dùng
  -> VLM hiểu câu lệnh
  -> VLM chọn vật thể mục tiêu
  -> VLM trả về bbox 2D của vật thể đó trên ảnh
```

Sau đó các mô-đun khác mới xử lý phần robot:

```text
VLM bbox
  -> map sang object_id trong PyBullet
  -> lấy RGB-D/point cloud của object
  -> GraspNet trained checkpoint sinh grasp pose 6DoF
  -> Panda robot dùng IK để gắp
```

Nói ngắn gọn:

```text
VLM = chọn đúng vật theo ngôn ngữ tự nhiên
GraspNet = tính tư thế gắp 6DoF
PyBullet/Panda = mô phỏng và điều khiển chuyển động robot
```

## 2. Các file liên quan đến VLM

Phần VLM chính nằm trong:

```text
src/vlm_localizer.py
```

Các hàm quan trọng:

```text
build_localization_prompt(...)
localize_target_object(...)
```

UI text box gọi VLM ở:

```text
scripts/08_vlm_panda_textbox_app.py
```

Pipeline VLM + PyBullet + GraspNet nằm ở:

```text
scripts/07_run_vlm_panda_pick_place.py
```

## 3. Dữ liệu đầu vào của VLM

Khi bạn nhập lệnh trong text box, ví dụ:

```text
gắp vật thể hình tròn
```

Hệ thống làm các bước sau:

1. PyBullet render một ảnh RGB từ camera ảo.
2. Ảnh được lưu thành:

```text
data/outputs/vlm_panda_textbox/01_render_rgb.png
```

3. Text command của bạn và ảnh RGB được đưa vào VLM local Qwen2.5-VL.

Trong code UI, bước này nằm ở:

```text
scripts/08_vlm_panda_textbox_app.py
```

Luồng chính:

```python
render_data = vlm_panda.render_camera_data(...)
vlm_panda.save_rgb(rgb, render_path)
vlm_result = localize_target_object(render_path, command, extra_rules=...)
```

## 4. Prompt gửi cho VLM

Prompt được tạo trong:

```text
src/vlm_localizer.py
```

Hàm:

```python
build_localization_prompt(user_command, image_width, image_height, extra_rules)
```

Nội dung prompt yêu cầu VLM làm đúng một việc:

```text
Find the single target object in the image that best matches the command.
Return exactly one JSON object.
```

Schema JSON bắt buộc:

```json
{
  "target_object": "short object name",
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": 0.0,
  "reason": "short reason"
}
```

Ý nghĩa từng field:

```text
target_object: tên vật thể VLM chọn, ví dụ "green sphere"
bbox: khung 2D bao quanh object trên ảnh RGB
confidence: độ tự tin từ 0.0 đến 1.0
reason: lý do ngắn gọn vì sao chọn object đó
```

Ví dụ output thực tế:

```json
{
  "target_object": "green sphere",
  "bbox": [492, 362, 528, 398],
  "confidence": 0.95,
  "reason": "The green object is a sphere, which is the best match for circular object."
}
```

## 5. Extra rules cho scene robot

Vì ảnh PyBullet có cả robot, bàn, sàn và khay xanh, tôi thêm `extra_rules` khi gọi VLM.

Trong `scripts/08_vlm_panda_textbox_app.py` và `scripts/07_run_vlm_panda_pick_place.py`, VLM được nhắc thêm:

```text
- Select only one small movable object resting on the gray table.
- Do not select the robot arm, gripper, floor, table, gray platform, or blue bin/tray.
- If the user asks for a color and the blue tray/bin matches, ignore the tray/bin and choose the best small tabletop object instead.
```

Mục đích là tránh trường hợp VLM chọn nhầm:

```text
robot arm
gripper
table
floor
blue bin/tray
```

VLM chỉ nên chọn vật nhỏ nằm trên bàn.

## 6. Cách gọi VLM local Qwen2.5-VL

Hàm gọi VLM nằm trong:

```text
src/vlm_localizer.py
```

Hàm chính:

```python
localize_target_object(...)
```

Các bước bên trong với backend mặc định:

1. Load `.env`.
2. Đọc backend từ `VLM_BACKEND`, mặc định là:

```text
qwen-local
```

3. Đọc model từ `QWEN_VL_MODEL`, mặc định là:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

4. Đọc kích thước ảnh.
5. Tạo prompt.
6. Đưa prompt + ảnh vào `Qwen2_5_VLForConditionalGeneration.generate(...)`.
7. Parse JSON và validate schema `target_object`, `bbox`, `confidence`, `reason`.

Nếu cần dùng lại API, gọi script với `--vlm-backend gemini` và cấu hình `GEMINI_API_KEY`.

## 7. Kiểm tra output của VLM

Output từ VLM được validate bằng Pydantic model:

```python
class TargetLocalization(BaseModel):
    target_object: str
    bbox: list[int]
    confidence: float
    reason: str
```

Hệ thống kiểm tra:

```text
bbox phải có 4 số
confidence phải nằm trong [0.0, 1.0]
bbox được clamp vào kích thước ảnh
```

Nếu VLM trả bbox vượt khỏi ảnh, hệ thống tự giới hạn lại trong ảnh.

Nếu bbox sai định dạng, code sẽ báo lỗi thay vì tiếp tục gắp nhầm.

## 8. Vì sao cần map bbox sang object_id

VLM chỉ biết ảnh 2D, nên nó trả về bbox pixel:

```text
[x_min, y_min, x_max, y_max]
```

Nhưng PyBullet điều khiển object bằng `object_id`, ví dụ:

```text
object_id = 13
```

Vì vậy cần bước chuyển:

```text
VLM bbox 2D -> PyBullet object_id
```

Bước này nằm trong:

```text
scripts/07_run_vlm_panda_pick_place.py
```

Hàm chính:

```python
select_object_from_bbox(...)
```

## 9. Cách map bbox sang object_id

Khi PyBullet render ảnh, nó có thể render thêm segmentation mask.

Segmentation mask cho biết mỗi pixel thuộc object nào:

```text
pixel (u, v) -> object_id
```

Hệ thống làm như sau:

1. Lấy bbox VLM trả về.
2. Cắt vùng segmentation tương ứng bbox.
3. Đếm pixel của từng object trong bbox.
4. Object nào có nhiều pixel nhất thì được chọn.

Ví dụ:

```json
"pixel_counts": {
  "13": 481
}
```

Nghĩa là trong bbox có 481 pixel thuộc `object_id = 13`, nên hệ thống chọn object 13.

## 10. Xử lý khi bbox của VLM bị lệch

VLM có thể hiểu đúng vật nhưng bbox hơi lệch.

Ví dụ log trước đó:

```text
VLM target: green sphere bbox=[492, 362, 528, 398]
Selection fallback: no object pixels or IoU...
```

Nghĩa là VLM biết đúng "green sphere", nhưng bbox không đè đúng segmentation object.

Để xử lý, tôi thêm fallback theo 3 mức:

```text
1. Nếu bbox chứa pixel object: chọn object có nhiều pixel nhất.
2. Nếu bbox không chứa pixel object: thử chọn object có IoU cao nhất với bbox.
3. Nếu vẫn không được: dùng semantic hint từ command + target_object + reason.
```

Semantic hint nghĩa là hệ thống đọc lại chữ như:

```text
gắp vật thể hình tròn
green sphere
The green object is a sphere
```

Sau đó ưu tiên object metadata phù hợp.

## 11. Metadata object để hỗ trợ VLM

Trong scene PyBullet, mỗi object được gán metadata:

```json
{
  "id": 13,
  "type": "sphere",
  "shape": "sphere",
  "color_name": "blue",
  "color_rgba": [0.05, 0.2, 0.9, 1.0]
}
```

Metadata được tạo trong:

```text
scripts/06_run_panda_pick_place_sim.py
```

Tôi thêm metadata vì PyBullet biết object là sphere/box/cylinder, còn VLM chỉ trả bbox. Khi bbox lệch, metadata giúp hệ thống chọn đúng theo ngữ nghĩa.

Ví dụ nếu command có:

```text
tròn
hình tròn
sphere
ball
quả cầu
```

Hệ thống ưu tiên:

```text
shape = sphere
```

Nếu command có:

```text
hình vuông
box
cube
hình hộp
```

Hệ thống ưu tiên:

```text
shape = box
```

Nếu command có màu:

```text
xanh lá
đỏ
xanh dương
vàng
trắng
```

Hệ thống cũng so với:

```text
color_name
```

## 12. VLM và GraspNet phối hợp như thế nào

Sau khi VLM chọn được object, GraspNet mới bắt đầu làm việc.

Luồng đầy đủ hiện tại:

```text
User command
  -> VLM chọn target_object và bbox
  -> bbox map sang selected_object_id
  -> render RGB-D của scene
  -> lấy point cloud riêng của selected_object_id bằng segmentation
  -> chạy GraspNet checkpoint-rs.tar
  -> GraspNet trả best grasp pose trong camera frame
  -> transform camera frame sang world frame
  -> Panda robot di chuyển tới pose đó
```

Quan trọng:

```text
VLM không tạo grasp pose.
VLM chỉ chọn vật cần gắp.
GraspNet mới là phần sinh grasp pose 6DoF.
```

## 13. Output được lưu ở đâu

Với UI text box, output thường nằm ở:

```text
data/outputs/vlm_panda_textbox/
```

Các file chính:

```text
01_render_rgb.png
01_render_depth.png
02_vlm_result.json
03_selected_bbox.png
04_graspnet/best_grasp.json
04_graspnet/target_cloud_for_graspnet.ply
vlm_panda_textbox_result.json
```

Ý nghĩa:

```text
01_render_rgb.png: ảnh gửi cho VLM
02_vlm_result.json: kết quả VLM trả về
03_selected_bbox.png: bbox object đã chọn sau khi map sang PyBullet
04_graspnet/best_grasp.json: grasp pose tốt nhất từ GraspNet
vlm_panda_textbox_result.json: tổng hợp toàn bộ pipeline
```

## 14. Cách kiểm tra VLM có hoạt động đúng không

Trong UI log, bạn nên nhìn các dòng:

```text
Calling VLM backend: qwen-local...
VLM target: ...
Selected PyBullet object_id: ...
Selected object: ...
Running GraspNet trained checkpoint...
```

Ví dụ tốt:

```text
Command: gắp vật thể hình tròn
VLM target: green sphere bbox=[...]
Selected object: {'shape': 'sphere', ...}
Running GraspNet trained checkpoint...
```

Nếu bạn nhập "gắp vật thể hình tròn" mà `Selected object` có:

```text
'shape': 'sphere'
```

thì bước VLM + object selection đã đúng.

## 15. Khi nào VLM có thể sai

VLM có thể sai trong các trường hợp:

```text
ảnh render quá xa hoặc object quá nhỏ
object bị robot che khuất
nhiều vật có màu/hình giống nhau
lệnh quá mơ hồ, ví dụ "gắp vật phù hợp nhất"
bbox trả về lệch khỏi object thật
```

Vì vậy tôi không dùng bbox VLM một cách mù quáng. Hệ thống có thêm:

```text
segmentation mask của PyBullet
object metadata
semantic fallback
JSON validation
bbox clamping
```

Các bước này làm pipeline ổn định hơn so với chỉ dùng bbox VLM.

## 16. Tóm tắt ngắn gọn

Trong project này, VLM được dùng như sau:

```text
1. Người dùng nhập lệnh tự nhiên trong text box.
2. PyBullet render ảnh RGB của scene.
3. Qwen2.5-VL local nhận ảnh + lệnh.
4. VLM trả về target_object, bbox, confidence, reason.
5. Code validate JSON và bbox.
6. bbox được map sang object_id bằng segmentation.
7. Nếu bbox lệch, semantic fallback dùng shape/color metadata để chọn đúng object.
8. Object đã chọn được đưa sang GraspNet để sinh grasp pose.
9. Robot Panda thực hiện gắp thả bằng pose từ GraspNet.
```

Vì vậy vai trò đúng của VLM là:

```text
hiểu câu lệnh + định vị vật thể mục tiêu trong ảnh
```

Vai trò đúng của GraspNet là:

```text
dự đoán tư thế gắp 6DoF từ point cloud của vật thể mục tiêu
```
