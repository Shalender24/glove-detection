## Dataset

#### Name: glove Computer Vision Model -daatset

### Source:https://universe.roboflow.com/ztar-rcuht/glove-klt3a/dataset/1

## Structure: 
-Standard YOLO format 
- the data has 8 classes , convetrted that to 2 classe according to our needs.

   
## Classes:
0 : with_glove
1: without_glove

## Model Ued : YOLO

## Architecture: YOLOv8n (Ultralytics)

## Training script: train_yolo.py

## Hyperparameters:
-Epochs: 50
-Batch size: 16
-Image size: 640
-Optimizer: Adam
-Learning rate: 0.001


## Augmentations used : 
-HSV, 
-scale, 
-flip, 
-mosaic

## Preprocessing done :
-Converted 8 original classes → 2 classes (classes 0–6 → 0, class 7 → 1).


## ** How to Run

1. Run Inference + Save Results
python inference.py \
  --input Dataset/test/images \
  --output outputs \
  --model runs/detect/2_cls_det/weights/best.pt \
  --confidence 0.4 --batch-size 32


This will:

Save annotated images in results/

Save JSON logs in results/json_logs/
