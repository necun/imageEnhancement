import cv2
import numpy as np

img=cv2.imread("D:\IITH/Necun\Opencv\INPUTS/13.jpeg",0)

def inversion(img):
    return cv2.bitwise_not(img)

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def threshold_equalizer(img):
     thresh_val,thresh_img=cv2.threshold(img,120,220,cv2.THRESH_BINARY)
     return thresh_img

def noise_removal(img):
    kernel = np.ones((1,1),np.uint8)
    img=cv2.dilate(img,kernel,iterations=1)#text font thickness increases
    kernel = np.ones((1, 1), np.uint8)
    img=cv2.erode(img,kernel,iterations=1)
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    img=cv2.medianBlur(img,1)
    return img

def font_thicker(img):
    img = inversion(img)
    img=cv2.dilate(img,np.ones((1, 1), np.uint8),iterations=1)
    img = inversion(img)
    return img

def font_thinner(img):
    img=inversion(img)
    img=cv2.erode(img,np.ones((1, 1), np.uint8),iterations=1)
    img=inversion(img)
    return img

def laplacian(img):
    kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    LaplacianImage = cv2.filter2D(src=img ,ddepth=-1,kernel=kernel)
    c = -1
    g = img + c*LaplacianImage
    gClip = np.clip(g, 0, 255)
    return gClip

def CLAHE(img):
    (b, g, r) = cv2.split(img)
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    equalized_imageb = clahe.apply(b)
    equalized_imageg = clahe.apply(g)
    equalized_imager = clahe.apply(r)
    merged = cv2.merge([equalized_imageb, equalized_imageg, equalized_imager])
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return clahe_img

def contrast_stretching(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) * (255 / (image_max - image_min))

def smooth_image(image, method='gaussian', kernel_size=(5, 5), sigma_x=0):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, kernel_size, sigma_x)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size[0])
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size[0], sigma_x, sigma_x)

def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def inpaint_image(image, mask, method='telea', radius=3):
    if method == 'navier_stokes':
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    else:
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)

blurred = cv2.GaussianBlur(gray, (21, 21), 0)

# Calculate the difference between the original and blurred images
diff = cv2.absdiff(gray, blurred)

# Threshold the difference image to create a binary mask
threshold = 60
_, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

# Apply morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Denoise shadow regions using mean filter and median filter
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    shadow_region = gray[y:y+h, x:x+w]

    # Apply mean filter
    mean_denoised = cv2.blur(shadow_region, (5, 5))

    # Apply median filter
    median_denoised = cv2.medianBlur(shadow_region, 5)

    # Replace the shadow region with the denoised version
    gray[y:y+h, x:x+w] = median_denoised


#cv2.imshow('Shadow denoised Image', gray)
#inversion_img=inversion(img)
#cv2.imshow('inversion_img',inversion_img)
#gray_img=grayscale(img)
#cv2.imshow('gray_img',gray_img)

noise_img=noise_removal(gClip)
cv2.imshow('noise_img',noise_img)

cv2.imshow('thresh_img',threshold_array)

#thresh_img=threshold_equalizer(gClip)
#cv2.imshow('thresh_img',thresh_img)

#noise_img=noise_removal(thresh_img)
#cv2.imshow('noise_img',noise_img)

#thin_img=font_thinner(noise_img)
#cv2.imshow('thin_img',thin_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
