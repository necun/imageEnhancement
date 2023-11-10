import cv2
import numpy as np

i = cv2.imread("C:/Users/DELL/OneDrive/Desktop/input/1.jpeg")

def apply_sharp_black_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def apply_black_pop_filter(image, brightness_factor=1.2, contrast_factor=1.2):
    img = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
    return img

def apply_soft_tone_filter(image, gamma=1.5):
    soft_tone_image = np.power(image / 255.0, gamma) * 255.0
    soft_tone_image = soft_tone_image.astype(np.uint8)
    return soft_tone_image

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image
    
image = apply_sharp_black_filter(i)
# Basic Enhancements
# 1. Brightness and Contrast Adjustment
alpha = 1.5  # Contrast control (1.0 is the original)
beta = 30    # Brightness control (0 is the original)
enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 2. Saturation and Sharpness Enhancement
hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
hsv[..., 1] = hsv[..., 1] * 1.5  # Increase saturation
enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)  # Sharpening

# 3. Noise Reduction
enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)  # Gaussian blur

# 4. White Balance
enhanced_image = cv2.xphoto.createSimpleWB().balanceWhite(enhanced_image)
image1 = apply_black_pop_filter(enhanced_image)
image1 = apply_sharp_black_filter(image1)
image1 = grayscale(image1)
# Save the enhanced image
cv2.imwrite('C:/Users/DELL/Downloads/enhanced_image22.jpg', image1)
