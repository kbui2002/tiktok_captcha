import cords
import cv2
import os
import detection

if __name__ == "__main__":
    path = 'C:/Users/Asus/OneDrive/Desktop/Captcha/captcha_shape/Dataset/runs/detect/train2/weights/best.pt'  # link model
    # link image(local or online)
    # image_src = 'https://user-images.githubusercontent.com/50222899/128652952-6a8d19a6-de15-455b-a626-0f3903b47c7d.png'
    image_src = 'C:/Users/Asus/OneDrive/Desktop/Captcha/cach-mo-tiktok-shop-1-600x646.jpg'
    coords_obj = cords.Coords(path, image_src)
    matching_coords = coords_obj.find_matching_coords(image_src, path)
    for item in matching_coords:
        print(item)
    # output: [[273.0709533691406, 168.03746032714844, 311.9222412109375, 223.98512268066406], [307.1864929199219, 244.7312774658203, 332.4696044921875, 279.63916015625]]
