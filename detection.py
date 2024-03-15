import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob
import ultralytics
from ultralytics import YOLO

class ModelPrediction():
    def __init__(self, path, image_src):
        self.model = YOLO(path)
        self.image_src = image_src

    @staticmethod
    def getModel(path):
        # path = 'C:/Users/Asus/OneDrive/Desktop/Captcha/Dataset/runs/detect/train2/weights/best.pt'
        model = YOLO(path)
        return model

    @staticmethod
    def predict(image_src, path):
        model = ModelPrediction.getModel(path)
        output = model.predict(source=image_src,save=True)  # save ảnh: save=True, show ảnh: show=True
        return output
