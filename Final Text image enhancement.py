import cv2
import numpy as np


input_image = cv2.imread("D:\IITH/Necun\Opencv\INPUTS/13.jpeg", cv2.IMREAD_GRAYSCALE)

def Local_Adaptive_Thresholding(image):
    return  cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

def horizontal_noise(image):# (between text lines)
    horizontal_projection = cv2.reduce(image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    threshold_value = 0.2 * max(horizontal_projection)
    binary_image = np.where(horizontal_projection < threshold_value, 255, image)
    return binary_image

def vertical_noise(image): #between characters
    vertical_projection = cv2.reduce(image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    threshold_value = 0.2 * max(vertical_projection)
    binary_image = np.where(vertical_projection < threshold_value, 255, image)
    return binary_image

def median_filter(image,kernel_size=5):
    return cv2.medianBlur(image, kernel_size)
def inversion(img):
    return cv2.bitwise_not(img)
def font_thicker(image):
    image = inversion(image)
    image=cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)
    image = inversion(image)
    return image

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(image, -1, kernel)
    filled_image = cv2.inpaint(image, sharpened_img, 3,cv2.INPAINT_NS)
    return filled_image

binary_image=Local_Adaptive_Thresholding(input_image)
binary_image=horizontal_noise(binary_image)
binary_image=vertical_noise(binary_image)
binary_image=median_filter(binary_image,kernel_size=5)
binary_image= font_thicker(binary_image)
binary_image=sharpen(binary_image)
#binary_image=inversion(binary_image)

cv2.imshow('binary_image',binary_image)
cv2.waitKey(0)