#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import scipy.signal as dsp

mrc = mrcfile.open('../../data/Position_72.rec')
data = mrc.data
print(data.shape)
data_filt = dsp.medfilt(data,5)
im      = data[:,150,:]
im_filt = data_filt[:,150,:]
plt.figure()
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.imshow(im_filt)
plt.show()
