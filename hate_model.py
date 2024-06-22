from PIL import Image

from PIL import ImageFilter, Image, ImageDraw
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import easyocr

path = 'C:/Users/siddh/Desktop/Humour/Fllf-atXoAA4ZXR.jpg'
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
edges_arr = cv.Canny(img,100,200)
edges = Image.fromarray(edges_arr)
image = Image.open(path)
y_nonzero, x_nonzero = np.nonzero(edges)
img = image.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext(path, paragraph=True)	

mask = Image.new(mode="RGB", size=(edges_arr.shape[0], edges_arr.shape[1]))
draw = ImageDraw.Draw(mask)
text = ''

for i in result:
    point_0 = (i[0][0][0], i[0][0][1])
    point_1 = (i[0][2][0], i[0][2][1])
    draw.rectangle([point_0, point_1], fill = 'white')
    text += i[1] 

print(text)
mask.show()
