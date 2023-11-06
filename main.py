import cv2
import numpy as np

image = cv2.imread("C:/Users/DELL/Downloads/sharpblack.jpg")

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
# Save the enhanced image
cv2.imwrite('C:/Users/DELL/Downloads/enhanced_image1.jpg', enhanced_image)
