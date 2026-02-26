import os
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

yolo8n_model_path = "E:/Code/Python/YOLOv8-main/yolov8l_best.pt"

# 配置 YOLOv8 作为 SAHI 检测模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8', # or 'yolov8'
    model_path=yolo8n_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cuda:0'
)

# result = get_prediction("E:/Code/Python/YOLOv8-main/input/40_10_0264.png", detection_model)

# 进行切片推理
result = get_sliced_prediction(
    "E:/Code/Python/YOLOv8-main/input/80_4_0002.png",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# 输出检测结果
print("检测到的目标数量:", len(result.object_prediction_list))
for obj in result.object_prediction_list:
    print(f"类别: {obj.category.name}, 置信度: {obj.score.value}")
