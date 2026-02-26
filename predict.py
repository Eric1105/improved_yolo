import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
import os

# 参数设置
image_path = "/home/tmp/Documents/deeplearning/yolov8-main/202510210000.jpg"
output_path = "/home/tmp/Documents/deeplearning/yolov8-main/output/202510210000.jpg"
model_path = "/home/tmp/Documents/deeplearning/yolov8-main/runs/detect/train4/weights/best.pt"
tile_size = 640  # 切片尺寸
overlap = 320  # ✅ 50%重叠
conf_threshold = 0.3
iou_threshold = 0.5  # 对小目标用0.5
large_obj_iou_threshold = 0.3  # ✅ 对大框用更低的阈值

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 读取大图
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"无法读取图像: {image_path}")
H, W, _ = image.shape
print(f"图像尺寸: {W}x{H}")

# ✅ 定义一个函数来判断是否是大目标
def is_large_object(box, image_size, threshold=0.3):
    """判断检测框是否是大目标（占图像面积的比例）"""
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_size[0] * image_size[1]
    return (box_area / image_area) > threshold

# 进行切片并推理
all_detections = []
stride = tile_size - overlap

for y in range(0, H, stride):
    for x in range(0, W, stride):
        # 获取切片区域
        x_end = min(x + tile_size, W)
        y_end = min(y + tile_size, H)
        
        # ✅ 边界处理：调整起始位置确保切片大小一致
        if x_end - x < tile_size and x > 0:
            x = max(0, x_end - tile_size)
        if y_end - y < tile_size and y > 0:
            y = max(0, y_end - tile_size)
        
        tile = image[y:y_end, x:x_end]
        tile = np.ascontiguousarray(tile)
        
        # 进行推理
        results = model.predict(tile, conf=conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                x1, y1, x2, y2 = xyxy
                
                # ✅ 坐标转换回大图
                x1_global = x1 + x
                x2_global = x2 + x
                y1_global = y1 + y
                y2_global = y2 + y
                
                # ✅ 过滤掉边界上的大框（这些可能是不完整的检测）
                tile_h, tile_w = tile.shape[:2]
                margin = 10  # 边界阈值（像素）
                
                # 判断框是否在切片边界
                on_boundary = (
                    x1 < margin or y1 < margin or 
                    x2 > tile_w - margin or y2 > tile_h - margin
                )
                
                # 判断是否是大框
                box_area = (x2 - x1) * (y2 - y1)
                tile_area = tile_w * tile_h
                is_large = (box_area / tile_area) > 0.2  # 占切片20%以上认为是大框
                
                # ✅ 如果是边界上的大框，降低置信度或跳过
                if on_boundary and is_large:
                    # 降低边界大框的置信度，让中心区域的检测更优先
                    conf = conf * 0.5
                
                all_detections.append([x1_global, y1_global, x2_global, y2_global, conf, class_id])

print(f"总检测框数: {len(all_detections)}")

# ✅ 按类别分别进行NMS，并对不同类别使用不同的IOU阈值
filtered_detections = []
if len(all_detections) > 0:
    all_classes = set([det[5] for det in all_detections])
    
    for class_id in all_classes:
        class_detections = [det for det in all_detections if det[5] == class_id]
        
        if len(class_detections) > 0:
            boxes = torch.tensor([det[:4] for det in class_detections], dtype=torch.float32)
            scores = torch.tensor([det[4] for det in class_detections], dtype=torch.float32)
            
            # ✅ 根据框的大小选择IOU阈值
            # 计算每个框的面积
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            image_area = H * W
            
            # 判断是否大多数是大框
            large_boxes = areas > (image_area * 0.1)  # 占图像10%以上
            
            if large_boxes.sum() > len(boxes) * 0.3:  # 如果30%以上是大框
                # 使用更低的IOU阈值
                current_iou = large_obj_iou_threshold
                print(f"类别{class_id}: 检测到大目标，使用IOU={current_iou}")
            else:
                current_iou = iou_threshold
                print(f"类别{class_id}: 使用标准IOU={current_iou}")
            
            keep = nms(boxes, scores, current_iou)
            
            for i in keep:
                filtered_detections.append(class_detections[i])
    
    print(f"NMS后检测框数: {len(filtered_detections)}")
    print(f"各类别数量: {[(cls, len([d for d in filtered_detections if d[5] == cls])) for cls in all_classes]}")
else:
    print("未检测到任何目标")

# ✅ 定义清晰的颜色方案（BGR格式）
colors = {
    0: (0, 255, 0),    # 绿色 - 小目标/液滴
    1: (255, 0, 0),    # 蓝色 - 大框/容器
    2: (0, 0, 255),    # 红色 - 其他
    3: (0, 255, 255),  # 黄色 - 其他
}

# 绘制检测结果
for det in filtered_detections:
    x1, y1, x2, y2, conf, class_id = det
    color = colors.get(int(class_id), (255, 255, 255))
    
    # 绘制矩形框
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # 获取类别名称
    class_name = model.names[int(class_id)] if hasattr(model, 'names') else f"Class {int(class_id)}"
    label = f"{class_name}: {conf:.2f}"
    
    # ✅ 添加标签背景
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, 
                  (int(x1), int(y1) - label_h - 10), 
                  (int(x1) + label_w, int(y1)), 
                  color, -1)
    
    # 白色文字
    cv2.putText(image, label, (int(x1), int(y1) - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 保存结果
cv2.imwrite(output_path, image)
print(f"结果已保存到: {output_path}")

try:
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("无法显示图像窗口(可能在服务器环境),请查看保存的文件")