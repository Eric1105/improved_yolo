# #---------------------------------------------------------------------------------#
# #单张图片测试
# from ultralytics import YOLO
# import cv2
# import os
# import shutil

# def convert_to_yolo_format(image_path, model_path, output_dir, conf_threshold=0.3):
#     """
#     将推理结果转换为YOLO数据集格式
    
#     Args:
#         image_path: 输入图像路径
#         model_path: 模型路径
#         output_dir: 输出目录
#         conf_threshold: 置信度阈值
#     """
#     # 创建输出目录结构
#     images_dir = os.path.join(output_dir, 'images')
#     labels_dir = os.path.join(output_dir, 'labels')
#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(labels_dir, exist_ok=True)
    
#     # 加载模型
#     model = YOLO(model_path)
    
#     # 读取图像
#     image = cv2.imread(image_path)
#     H, W, _ = image.shape
    
#     # 推理
#     results = model.predict(
#         image,
#         conf=conf_threshold,
#         imgsz=2816,
#         verbose=False,
#         device=0
#     )
    
#     # 获取图像文件名（不含扩展名）
#     image_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # 复制图像到输出目录
#     output_image_path = os.path.join(images_dir, f"{image_name}.jpg")
#     shutil.copy(image_path, output_image_path)
    
#     # 创建标注文件
#     label_path = os.path.join(labels_dir, f"{image_name}.txt")
    
#     boxes = results[0].boxes
#     with open(label_path, 'w') as f:
#         for box in boxes:
#             # 获取xyxy格式坐标
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             class_id = int(box.cls[0].cpu().numpy())
            
#             # ✅ 转换为YOLO格式 (class_id x_center y_center width height)
#             # 归一化坐标到[0, 1]
#             x_center = ((x1 + x2) / 2) / W
#             y_center = ((y1 + y2) / 2) / H
#             width = (x2 - x1) / W
#             height = (y2 - y1) / H
            
#             # 写入文件
#             f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
#     print(f"已转换: {image_name}")
#     print(f"  图像: {output_image_path}")
#     print(f"  标注: {label_path}")
#     print(f"  目标数: {len(boxes)}")
    
#     return len(boxes)

# # 使用示例
# image_path = "/home/tmp/Documents/deeplearning/yolov8-main/202510210160.jpg"
# model_path = "/home/tmp/Documents/deeplearning/yolov8-main/runs/detect/train4/weights/best.pt"
# output_dir = "/home/tmp/Documents/deeplearning/yolov8-main/yolo_dataset"

# convert_to_yolo_format(image_path, model_path, output_dir, conf_threshold=0.3)

#---------------------------------------------------------------------------------#
#批量转换

from ultralytics import YOLO
import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def batch_convert_to_yolo(image_dir, model_path, output_dir, conf_threshold=0.3, imgsz=2816):
    """
    批量将推理结果转换为YOLO数据集格式
    
    Args:
        image_dir: 输入图像文件夹
        model_path: 模型路径
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        imgsz: 推理尺寸
    """
    # 创建输出目录结构
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = YOLO(model_path)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 张图像")
    
    total_objects = 0
    
    # 批量处理
    for image_path in tqdm(image_files, desc="转换中"):
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取: {image_path}")
            continue
        
        H, W, _ = image.shape
        
        # 推理
        results = model.predict(
            image,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
            device=0
        )
        
        # 获取文件名
        image_name = image_path.stem
        image_ext = image_path.suffix
        
        # 复制图像
        output_image_path = os.path.join(images_dir, f"{image_name}{image_ext}")
        shutil.copy(str(image_path), output_image_path)
        
        # 创建标注文件
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        
        boxes = results[0].boxes
        with open(label_path, 'w') as f:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # 转换为YOLO格式
                x_center = ((x1 + x2) / 2) / W
                y_center = ((y1 + y2) / 2) / H
                width = (x2 - x1) / W
                height = (y2 - y1) / H
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        total_objects += len(boxes)
    
    # 创建data.yaml文件
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# Auto-generated YOLO dataset\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"nc: {len(model.names)}\n")
        f.write(f"names:\n")
        for i, name in model.names.items():
            f.write(f"  {i}: {name}\n")
    
    print(f"\n转换完成!")
    print(f"  总图像数: {len(image_files)}")
    print(f"  总目标数: {total_objects}")
    print(f"  输出目录: {output_dir}")
    print(f"  data.yaml: {yaml_path}")

# 使用示例
image_dir = "/home/tmp/Documents/deeplearning/yolov8-main/test_images"
model_path = "/home/tmp/Documents/deeplearning/yolov8-main/runs/detect/train4/weights/best.pt"
output_dir = "/home/tmp/Documents/deeplearning/yolov8-main/yolo_dataset"

batch_convert_to_yolo(image_dir, model_path, output_dir, conf_threshold=0.3)