import cv2
from matplotlib import pyplot as plt
import math
# check tọa độ của ảnh
# Load ảnh
# Đường dẫn đến file ảnh của bạn
image_path = 'C:/Users/Asus/OneDrive/Desktop/Captcha/runs/detect/predict2/128652952-6a8d19a6-de15-455b-a626-0f3903b47c7d.png'
image = cv2.imread(image_path)

# Tọa độ của vùng cần cắt [x1, y1, x2, y2]
coordinates_1 = [431.7752990722656, 325.7951354980469, 499.2908020019531, 423.9853210449219]
coordinates_2 = [184.68255615234375, 186.78347778320312, 228.7422637939453, 246.72108459472656]

x1, y1, x2, y2 = [math.ceil(coord) for coord in coordinates_1]
x3, y3, x4, y4 = [math.ceil(coord) for coord in coordinates_2]
# Cắt vùng trong ảnh
cropped_image_1 = image[y1:y2, x1:x2]
cropped_image_2 = image[y3:y4, x3:x4]

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cropped_image_1, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image 1')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(cropped_image_2, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image 2')

plt.show()

# cv2.imshow('Cropped Image_1',cropped_image_1)
# cv2.imshow('Cropped Image_2', cropped_image_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
