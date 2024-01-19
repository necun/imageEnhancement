import cv2
import numpy as np

def enhance_image(image_path):

    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    brightness = 30
    contrast = 55
    dummy = np.int16(enhanced_img)
    dummy = dummy * (contrast/127+1) - contrast + brightness
    dummy = np.clip(dummy, 0, 255)
    enhanced_img = np.uint8(dummy)
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20)
    enhanced_img = cv2.merge([h, s, v])
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_HSV2BGR)
    return enhanced_img


original_image_path = 'Inputs/13.jpeg'
enhanced_image = enhance_image(original_image_path)
cv2.imwrite('Outputs/13_enhanced_image.png', enhanced_image)
