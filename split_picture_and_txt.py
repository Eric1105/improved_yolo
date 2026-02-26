import cv2
import os
import numpy as np

# 输入/输出目录
input_image_dir = "E:/Code/Python/yolov11-microball/microball-original/test/images"
input_label_dir = "E:/Code/Python/yolov11-microball/microball-original/test/labels"
output_image_dir = "E:/Code/Python/yolov11-microball/microball/test/images"
output_label_dir = "E:/Code/Python/yolov11-microball/microball/test/labels"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 切割参数
tile_size = 1024  # 切片大小（推荐1024x1024）
overlap = 512      # 重叠区域（50%）

def read_yolo_label(label_path, img_w, img_h):
    """读取 YOLO 格式标签并转换为像素坐标"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            # 还原为像素坐标
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)
            boxes.append((class_id, x1, y1, x2, y2))
    return boxes

def save_yolo_label(output_label_path, boxes, tile_w, tile_h):
    """保存切图后的 YOLO 标签"""
    with open(output_label_path, "w") as f:
        for box in boxes:
            class_id, x1, y1, x2, y2 = box
            # 归一化回 YOLO 格式
            x_center = ((x1 + x2) / 2) / tile_w
            y_center = ((y1 + y2) / 2) / tile_h
            width = (x2 - x1) / tile_w
            height = (y2 - y1) / tile_h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def split_image_and_labels(image_path, label_path):
    """对图片进行切片，并转换目标框"""
    img = cv2.imread(image_path)
    img_h, img_w, _ = img.shape

    boxes = read_yolo_label(label_path, img_w, img_h)

    # 计算切片数量
    for i in range(0, img_h, tile_size - overlap):
        for j in range(0, img_w, tile_size - overlap):
            # 切片坐标
            x1, y1 = j, i
            x2, y2 = min(j + tile_size, img_w), min(i + tile_size, img_h)
            tile = img[y1:y2, x1:x2]

            # 计算新目标框
            new_boxes = []
            for box in boxes:
                class_id, bx1, by1, bx2, by2 = box
                # 目标框与切片重叠部分
                if bx2 > x1 and bx1 < x2 and by2 > y1 and by1 < y2:
                    nx1 = max(bx1 - x1, 0)
                    ny1 = max(by1 - y1, 0)
                    nx2 = min(bx2 - x1, tile_size)
                    ny2 = min(by2 - y1, tile_size)
                    new_boxes.append((class_id, nx1, ny1, nx2, ny2))

            # 保存切片图像
            tile_name = f"{os.path.basename(image_path).split('.')[0]}_{i}_{j}.jpg"
            cv2.imwrite(os.path.join(output_image_dir, tile_name), tile)

            # 保存新目标框
            label_name = tile_name.replace(".jpg", ".txt")
            save_yolo_label(os.path.join(output_label_dir, label_name), new_boxes, tile_size, tile_size)

# 处理所有图片
for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_image_dir, filename)
        label_path = os.path.join(input_label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        split_image_and_labels(image_path, label_path)

print("切图完成！")
