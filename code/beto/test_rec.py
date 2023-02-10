#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import scipy.signal as dsp
from scipy.signal.windows import hamming
from skimage import io as imgio

W = 5 # 7 is good together with q=0.1
q = 0.05
win1 = hamming(W)
k2 = np.outer(win1,win1)
k2 = np.reshape(k2,(W,W,1))
k1 = np.reshape(win1,(1,W))
kernel = np.tensordot(k2,k1,axes=1)

mrc = mrcfile.open('../../data/Position_72.rec')
data = mrc.data
print(data.shape)
im = data[:,150,:]
plt.figure()
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
#data_filt = dsp.medfilt(data,5)
data_filt = dsp.convolve(data,kernel)
thres = np.quantile(data_filt.ravel(),q)
data_filt = data_filt < thres
im_filt = data_filt[:,150,:]
plt.imshow(im_filt)
#mrcfile.write("Position_72_filtered_filt.rec",overwrite=True)
imgio.imsave("Position_72_row_150.pbm",im_filt)
plt.show()
x,y = np.nonzero(im_filt)
points = list(zip(x,y))
print(points)

x, y, z = np.nonzero(data_filt)
N = len(x)
points = list()
for i in range(N):
    points.append((x[i], y[i], z[i]))
print(points)

