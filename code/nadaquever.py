#!/usr/bin/env python3

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3

n = 3
N = 1000
x = rand.uniform(low=0,high=1.2,size=(N,n))
#print('x',x)
xp = np.empty((N,n))

for i in range(N):
    xi = x[i,:]
    if np.sum(xi) < 1:
        xp[i,:] = x[i,:]
        continue
    ai = np.abs(xi)
    #print('abs(x)',ax)
    si = np.sign(xi)
    idx = np.argsort(-ai)
    y = ai[idx]
    #print('sorted(abs(x))',y)
    w = (np.cumsum(y) - 1)/np.arange(1,n+1)
    #print('w',w)
    good = False
    for j in range(len(w)):
        yj = np.maximum(y-w[j],0)
        Sj = np.sum(yj)
        if np.abs(Sj-1) < 1e-8:
            xp[i,:] = yj[idx]
            good = True
            break
    if not good:
        print('no good')
        break
        #print('y-w_i',yj)
        #print('sum(y-w_i)',np.sum(yi))
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2],color='black',label='orig',alpha=0.2)
ax.scatter(xp[:,0],xp[:,1],xp[:,2],color='blue',label='proj',alpha=0.2)
ax.legend()
plt.show()
