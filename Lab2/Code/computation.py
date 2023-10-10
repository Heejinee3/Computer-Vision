import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits import mplot3d

Filter = []
for i in range(3,17,2):
    Filter.append(np.random.randn(i, i))

Image = []
Image.append(cv2.imread('../questions/RISDance.jpg'))
for i in range(1,5):
    Image.append(cv2.resize(Image[0],dsize=(0, 0), fx=1-i*0.2, fy=1-i*0.2))

ImageSize = []
for i in range(5):
    ImageSize.append((Image[i].shape[0])*(Image[i].shape[1])/1000000)

Measure = [0,0,0,0,0,0,0]
for i in range(7):
    Measure[i] = [0,0,0,0,0]
for i in range(7):
    for j in range(5):
        start_time = time.time()
        output = cv2.filter2D(Image[j],-1,Filter[i])
        Measure[i][j] = time.time() - start_time


x = np.array(ImageSize)
y = np.linspace(3,15,7)
X,Y = np.meshgrid(x,y)
Z = np.array(Measure)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Computation Time')
ax.set_xlabel('Image Size (MPix)')
ax.set_ylabel('Filter Width (pixel)')
ax.set_zlabel('Computation Time')

plt.show()
