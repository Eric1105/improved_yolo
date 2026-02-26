# from ultralytics import YOLO
#
# # If you want to finetune the model with pretrained weights, you could load the
# # pretrained weights like below
# # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# # or
# # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLO('yolo11m.pt')
# model.train(data='droplet.yaml', epochs=500, batch=16, imgsz=640)


# Make sure to add the following lines at the bottom of your train.py script:

if __name__ == '__main__':
    import torch
    from ultralytics import YOLO

    # Initialize the YOLO model or training configuration
    model = YOLO("/home/tmp/Documents/deeplearning/yolov8-main/ultralytics/models/v8/yolov8m.yaml")

    # Call the train function
    model.train(data='/home/tmp/Documents/deeplearning/yolov8-main/dataset_1024/data.yaml', epochs=500, batch=16, imgsz=1024)
