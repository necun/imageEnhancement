import cv2
import numpy as np
from PIL import Image,ImageEnhance
from io import BytesIO
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import FileResponse

app=FastAPI()

def Local_Adaptive_Thresholding(image):
    return  cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
def illumination_adjustment(image):
    enhanced_image=cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    return enhanced_image
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
    dilate = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5)), iterations=5)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -1.0 * angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image
@app.get('/')
async def root():
    return {'Title':'Image enchancement'}

@app.post('/upload/')
async def input(file:UploadFile):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")  # Convert to grayscale
    input_image = np.array(image)
    binary_image = Local_Adaptive_Thresholding(input_image)
    binary_image = deskew(binary_image)
    binary_image = horizontal_noise(binary_image)
    binary_image = vertical_noise(binary_image)
    binary_image = median_filter(binary_image, kernel_size=5)
    binary_image = font_thicker(binary_image)
    binary_image = sharpen(binary_image)
    # binary_image=inversion(binary_image)#inverted input for OCR
    output_path = 'D:\IITH/Necun\imageEnhancement\Outputs/binary_image.png'
    cv2.imwrite(output_path, binary_image)
    return FileResponse(output_path)