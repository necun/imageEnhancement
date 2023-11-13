import cv2
import numpy as np
from PIL import Image,ImageEnhance
import pytesseract


def illumination_adjustment(image):
    enhanced_image=cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    return enhanced_image

def Local_Adaptive_Thresholding(image):
    return  cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

def horizontal_noise(image):# (between text lines)
    horizontal_projection = cv2.reduce(image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    threshold_value = 0.2 * max(horizontal_projection)
    binary_image = np.where(horizontal_projection < threshold_value, 255, image)
    return binary_image

def vertical_noise(image): #between characters
    vertical_projection = cv2.reduce(image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    threshold_value = 0.01 * max(vertical_projection)
    binary_image = np.where(vertical_projection < threshold_value, 255, image)
    return binary_image

def whole_noise(image):
    enhanced_image = cv2.GaussianBlur(image, (5, 5), 0)
    return enhanced_image
def white_balance(image):
   enhanced_image = cv2.xphoto.createSimpleWB().balanceWhite(image)
   return enhanced_image

def median_filter(image,kernel_size=5):
    return cv2.medianBlur(image, kernel_size)
def inversion(img):
    return cv2.bitwise_not(img)
def font_thicker(image):
    image = inversion(image)
    image=cv2.dilate(image, np.ones((1, 1), np.uint8), iterations=1)
    image = inversion(image)
    return image

def font_thinner(image):
    image =inversion(image)
    image = cv2.erode(image,np.ones((1,1),np.uint8),iterations=1)
    image=inversion(image)
    return image
def saturation_sharpness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5  # Increase saturation
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
    return enhanced_image

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge_img = cv2.filter2D(image, -1, kernel)
    edge_img=Image.fromarray(edge_img)
    pillow_enhancer = ImageEnhance.Sharpness(edge_img)
    sharpened_img = pillow_enhancer.enhance(2)
    sharpened_img=np.array(sharpened_img)
    filled_image = cv2.inpaint(image, sharpened_img, 0.5,cv2.INPAINT_NS)
    return filled_image

def deskew(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    median_angle = np.degrees(np.median(angles))
    rotation_angle = -(90 + median_angle) if median_angle < -45 else -median_angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image


input_image = cv2.imread("D:\IITH/Necun\imageEnhancement\Inputs/13.jpeg", cv2.IMREAD_GRAYSCALE)
binary_image = Local_Adaptive_Thresholding(input_image)
binary_image = deskew(binary_image)
binary_image = horizontal_noise(binary_image)
binary_image = vertical_noise(binary_image)
binary_image = median_filter(binary_image, kernel_size=5)
binary_image = font_thicker(binary_image)
binary_image = sharpen(binary_image)
# binary_image=inversion(binary_image)#inverted input for OCR
output_path = 'D:\IITH/Necun\imageEnhancement\Outputs/binary_image_deskew.png'
cv2.imwrite(output_path, binary_image)
cv2.waitKey(0)


