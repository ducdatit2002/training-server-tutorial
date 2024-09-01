import os
import sys
import subprocess
import time
import numpy as np

from ultralytics import YOLO

HOME = os.getcwd() 
print(HOME)
os.chdir(HOME)

"""## Train model"""
subprocess.run(['yolo','task=classify', 'mode=train', 'model=yolov8m-cls.pt','data=home/ldtan/ldtan/PLANT50-ServerTrain/data/plant','epochs=300', 'batch=16', 'imgsz=640', 'dropout=0.2', 'save=True', 'save_period=10'])

"""## Validating model"""

os.chdir(HOME)

with open('valid.txt', 'w') as f:
    result = subprocess.run(['yolo', 'task=classify', 'mode=val', 'model=runs/classify/train/weights/best.pt', 'imgsz=640', 'data=home/ldtan/ldtan/PLANT50-ServerTrain/data/plant'], capture_output=True, text=True)
    f.write(result.stdout)
    f.write(result.stderr)
  
"""## Inference model"""
# Define the class names
class_names = ['aloevera', 'amla', 'amruta balli', 'apple', 'ashwagandha', 'azadirachta indica', 'banana', 'bellpepper', 'betel piper', 'bilimbi', 'broccoli', 'cabbage', 'cantaloupe', 'carrot','cassava', 'cauliflower',
               'citrus limon', 'coconut', 'corn', 'cucumber', 'curcuma', 'curry leaf', 'eggplant', 'galangal', 'ganike', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'mentha', 'ocimum sanctum', 'orange', 'paddy',
               'papaya', 'peper chili', 'pineapple', 'pomelo', 'potato', 'pumpkin', 'radish', 'rosa sinensis', 'shallot', 'soybeans', 'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon']


start_time = time.time()
model = YOLO('runs/classify/train/weights/best.pt')

class_inference_times = []
class_top1_accuracies = []

for class_name in class_names:
    class_start_time = time.time()
    class_path = os.path.join('home/ldtan/ldtan/PLANT50-ServerTrain/data/plant/valid/', class_name)
    class_results = model(source=class_path, imgsz=640)
    class_inference_time = time.time() - class_start_time
    class_inference_times.append(class_inference_time)

    class_top1_confidences = []
    for result in class_results:
        top1_conf = result.probs.top1conf.cpu().numpy().item()
        class_top1_confidences.append(top1_conf)
    class_top1_accuracy = np.mean(class_top1_confidences)
    class_top1_accuracies.append(class_top1_accuracy)

total_inference_time = time.time() - start_time
average_class_inference_time = np.mean(class_inference_times)
average_top1_accuracy = np.mean(class_top1_accuracies)

with open('inference.txt', 'w') as f:
    f.write("Inference Results:\n")
    f.write("+-----------------------+----------------------+----------------------+\n")
    f.write("| Metric                | Value                |                      |\n")
    f.write("+-----------------------+----------------------+----------------------+\n")
    f.write(f"| Total number of classes         | {len(class_names)}                  |             |\n")
    f.write("| Total Inference Time            | {:.2f} seconds      |             |\n".format(total_inference_time))
    f.write("| Average Class Inference Time    | {:.2f} seconds       |             |\n".format(average_class_inference_time))
    f.write("| Average Accuracy                | {:.3f}               |             |\n".format(average_top1_accuracy))
    f.write("+-----------------------+----------------------+----------------------+\n")
    f.write("|        Class          |   Inference Time (s) |       Accuracy       |\n")
    f.write("+-----------------------+----------------------+----------------------+\n")
    for i, class_name in enumerate(class_names):
        f.write("| {:<20}  | {:<20.2f} | {:<20.2f} |\n".format(class_name, class_inference_times[i], class_top1_accuracies[i]))
    f.write("+-----------------------+----------------------+----------------------+\n")
