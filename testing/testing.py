import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

#Test 1
"""model = YOLO("best.pt")  # Adjust the path to your best.pt

# Load the image
img = Image.open("player (507).png")

# Perform inference
results = model(img)

# Display results
print(results)""" # wrong classes

#Test 2
"""
# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
model = YOLO('best.pt')  # load a pretrained YOLOv8n detection model
model.train(data='config.yaml', epochs=2)  # train the model
model('player (507).png')  # predict on an image
"""

#Test 3: success

image_path = "player (1029).png"
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 450))
model = YOLO(r"C:\Users\diwak\OneDrive\Desktop\Football\detect\train6\weights\last.pt")
result = model.predict(resized_image,show=True)
# print(result)
cv2.waitKey(0)
cv2.destroyAllWindows()


