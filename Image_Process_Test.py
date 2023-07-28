from PIL import Image
import numpy as np
import matplotlib as mpl

im = Image.open("garden.jpg")
print(im.size)

RGB = im.getbands()



im = im.rotate(270)
X=600
Y=0
x=4000-X
y=3000
box = (X, Y, x, y)

im = im.crop(box)

threshold = (200,200)
threshold2 = (100,100)
threshold3 = (50,50)




color2 = (100,100,100)
color = (200,200,200)
color3 = (0,0,0)
colorelse = (225,225,225)

for i in range(im.width):
    for j in range(im.height):
        if im.getpixel((i,j)) < threshold2:
            im.putpixel((i,j),color2)
        elif im.getpixel((i,j)) < threshold:
            im.putpixel((i,j),color)
        elif im.getpixel((i,j)) < threshold3:
            im.putpixel((i,j),color3)
        else:
            im.putpixel((i,j),colorelse)



im.show()

