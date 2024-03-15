import cv2
from ultralytics import YOLO
from detection import ModelPrediction

class Coords:
    def __init__(self, path, image_src):
        # self.model = ModelPrediction(path)
        self.path = path
        self.image_src = image_src

    def find_matching_coords(self, image_src, path):
        results = ModelPrediction.predict(image_src, path)
        allCords = []
        matchingCords = []

        for result in results:
            boxes = result.boxes.xyxy  # Bounding boxes in xyxy format
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls
            # [x1, y1, x2, y2], x1y1 is the top-left corner, x2y2 is the bottom-right corner
            for box, conf, cls in zip(boxes, confs, classes):
                allCords.append((box, conf, cls))

        for i in range(len(allCords)):
            for j in range(i + 1, len(allCords)):
                if allCords[i][-1] == allCords[j][-1]:
                    matchingCords.append(allCords[i][0])
                    matchingCords.append(allCords[j][0])

        extracted_coords = [tensor.tolist() for tensor in matchingCords]
        return extracted_coords

# Đường dẫn đến file weights
# path = 'C:/Users/Asus/OneDrive/Desktop/Captcha/Dataset/runs/detect/train2/weights/best.pt'

# # Tạo một đối tượng coords với đường dẫn đã cho
# coords_obj = Coords(path)

# # URL hoặc đường dẫn đến hình ảnh cần kiểm tra
# image_src = 'https://user-images.githubusercontent.com/50222899/128652952-6a8d19a6-de15-455b-a626-0f3903b47c7d.png'

# # Tìm các tọa độ giống nhau trong hình ảnh
# matching_coords = coords_obj.find_matching_coords(image_src)

# # In ra các tuple giống nhau
# print("Matching Tuples:")
# for tpl in matching_coords:
#     print(tpl)
