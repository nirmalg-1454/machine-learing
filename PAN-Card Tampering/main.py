from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import requests

original = Image.open("C:\\Users\gnirm\Downloads\original_pan.png")
tampered = Image.open("C:\\Users\gnirm\Downloads\original_pan.png")

def getsize():
    print("Original Image format : ", original.format)
    print("Tampered Image format : ", tampered.format)
    print("Original Image Size : ", original.size)
    print("Tampered Image Size : ", tampered.size)

getsize()
original = original.resize((250, 160))
original.save("pan_card_tampering\image\original.png")
tampered = tampered.resize((250, 160))
tampered.save("pan_card_tampering\image\\tampered.png")
getsize()

print(original)
print(tampered)

original = cv2.imread("pan_card_tampering\image\original.png")
tampered = cv2.imread("pan_card_tampering\image\\tampered.png")

original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM : {}".format(score))

# Calculating threshold and Contours
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)

Image.fromarray(original)
Image.fromarray(tampered)

Image.fromarray(diff) # black shows the diff
Image.fromarray(thresh) # white portion shows the diff


