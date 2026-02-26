from ultralytics import YOLO
import cv2
import numpy as np

# 参数设置
image_path = "/home/tmp/Documents/deeplearning/yolov8-main/202510210160.jpg"
output_path = "/home/tmp/Documents/deeplearning/yolov8-main/output/202510210160_full.jpg"
model_path = "/home/tmp/Documents/deeplearning/yolov8-main/runs/detect/train4/weights/best.pt"

# 加载模型
model = YOLO(model_path)

# 读取图像
image = cv2.imread(image_path)
H, W, _ = image.shape
print(f"图像尺寸: {W}x{H}")

# 推理整图
results = model.predict(
    image,
    conf=0.3,
    imgsz=3840,
    verbose=True,
    device=0
)

# 获取检测结果
boxes = results[0].boxes
print(f"\n检测到 {len(boxes)} 个目标")

# 统计各类别数量
if len(boxes) > 0:
    classes = boxes.cls.cpu().numpy()
    for class_id in set(classes):
        count = (classes == class_id).sum()
        class_name = model.names[int(class_id)]
        print(f"  {class_name} (类别{int(class_id)}): {count} 个")

# ✅ 手动绘制结果（兼容旧版本）
colors = {
    0: (0, 255, 0),    # 绿色 - 大框/容器
    1: (255, 0, 0),    # 蓝色 - 小目标/液滴
    2: (0, 0, 255),    # 红色
    3: (0, 255, 255),  # 黄色
}

image_result = image.copy()

for box in boxes:
    # 获取坐标、置信度、类别
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0].cpu().numpy())
    class_id = int(box.cls[0].cpu().numpy())
    
    color = colors.get(class_id, (255, 255, 255))
    
    # 根据框大小调整线宽和字体
    box_area = (x2-x1) * (y2-y1)
    image_area = W * H
    is_large = (box_area / image_area) > 0.05  # 占图像5%以上算大框
    
    thickness = 4 if is_large else 2
    font_scale = 0.8 if is_large else 0.6
    font_thickness = 2 if is_large else 1
    
    # 画框
    cv2.rectangle(image_result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # 准备标签
    class_name = model.names[class_id]
    label = f"{class_name}: {conf:.2f}"
    
    # 计算文字大小
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    
    # 画标签背景
    y_text = int(y1) - 10
    if y_text < label_h + 10:  # 如果在图像顶部，标签放在框内
        y_text = int(y1) + label_h + 10
    
    cv2.rectangle(
        image_result, 
        (int(x1), y_text - label_h - 5), 
        (int(x1) + label_w + 5, y_text + 5), 
        color, -1
    )
    
    # 画文字
    cv2.putText(
        image_result, label, 
        (int(x1) + 2, y_text), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, (255, 255, 255), 
        font_thickness, cv2.LINE_AA
    )

# 保存结果
cv2.imwrite(output_path, image_result)
print(f"\n结果已保存到: {output_path}")

# 显示（可选）
try:
    # 由于图片较大，缩小显示
    display_scale = 0.5
    display_h = int(H * display_scale)
    display_w = int(W * display_scale)
    image_display = cv2.resize(image_result, (display_w, display_h))
    
    cv2.imshow("Result (50% scale)", image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("无法显示图像窗口")

# ✅ 保存详细的检测信息到txt
txt_path = output_path.replace('.jpg', '.txt')
with open(txt_path, 'w') as f:
    f.write(f"Image: {image_path}\n")
    f.write(f"Size: {W}x{H}\n")
    f.write(f"Total detections: {len(boxes)}\n\n")
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[class_id]
        
        f.write(f"Object {i+1}:\n")
        f.write(f"  Class: {class_name} (ID: {class_id})\n")
        f.write(f"  Confidence: {conf:.4f}\n")
        f.write(f"  BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]\n")
        f.write(f"  Size: {int(x2-x1)}x{int(y2-y1)}\n\n")

print(f"检测信息已保存到: {txt_path}")