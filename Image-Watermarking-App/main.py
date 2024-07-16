import numpy as np
from PIL import Image
import cv2
import requests

image = Image.open("app\\static\\uploads\\image1.png")
image_logo_width = image.resize((500, 300))
image_text_width = image.resize((500, 300))
print(image_logo_width)

logo = Image.open("app\\static\\uploads\\logo.png")

image_logo_width = np.array(image_logo_width.convert('RGB'))
h_image, w_image, _ = image_logo_width.shape
logo = np.array(logo.convert('RGB'))
h_logo, w_logo, _ = logo.shape

center_y = int(h_image / 2)
center_x = int(w_image / 2)
top_y = center_y - int(h_logo / 2)
left_x = center_x - int(w_logo / 2)
bottom_y = top_y - h_logo
right_x = left_x - w_logo

# ROI - Region of Interest
roi = image_logo_width[top_y: bottom_y, left_x: right_x]
result = cv2.addWeighted(roi, 1, logo, 1, 0)

cv2.line(image_logo_width, (0, center_y), (left_x, center_y), (0, 0, 255), 1)
cv2.line(image_logo_width, (right_x, center_y), (w_image, center_y), (0, 0, 255), 1)
image_logo_width[top_y: bottom_y, left_x: right_x] = result

img = Image.fromarray(image_logo_width, 'RGB')
print(img)

# Text Watermark