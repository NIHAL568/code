import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(""C:\Users\prana\OneDrive\Desktop\DL Lab\Cricket.jpg"")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
histeq=cv2.equalizeHist(gray)
thress,bin=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Threshold value is {thress}")
kernel=np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(bin,cv2.MORPH_OPEN,kernel,iterations=1)

plt.figure(figsize=(5,4))
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Orginal Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray,cmap='cool')
plt.title("Gray Image")
plt.axis("off")


plt.subplot(2,2,3)
plt.imshow(histeq)
plt.title("Histogram Equilization")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(opening,cmap='gray')
plt.title("Morphology Operation")
plt.axis("off")

plt.tight_layout()
plt.show()
