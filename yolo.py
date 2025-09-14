import cv2
import os
import numpy as np

file_dir = "C:/capstone/data/266.AI_기반_아동_미술심리_진단을_위한_그림_데이터_구축/01-1.정식개방데이터/Training/imageslabels"
file_name = "남자사람_7_남_00026"

image_file_path = os.path.join(file_dir, "images", file_name + ".jpg")
label_file = open(os.path.join(file_dir, "labels", file_name + ".txt"), mode='r', encoding='utf8')

img_array = np.fromfile(image_file_path, np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

labels = label_file.readlines()
for label in labels:
    label_value = label.split(' ')
    x1 = int((float(label_value[1]) - float(label_value[3])/2) * image.shape[1])
    y1 = int((float(label_value[2]) - float(label_value[4])/2) * image.shape[0])
    x2 = int((float(label_value[1]) + float(label_value[3])/2) * image.shape[1])
    y2 = int((float(label_value[2]) + float(label_value[4])/2) * image.shape[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 1)

cv2.imshow('image', image)
key = cv2.waitKey(0)
cv2.destroyAllWindows()