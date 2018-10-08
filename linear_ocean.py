#!/usr/bin/env python

"""
This program solves the Stommel and/or Munk model

d\psi /dx + \epsilon_s \nabla^2 \psi + \epsilon_m \nabla^4 \psi=  wsc

One can solve the Stommel problem with \epsilon_m = 0 and the munk
problem with \epsilon_s = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import spoisson

plt.ion()

# friction parameters for Stommel and Munk models. 
# (set to zero one of these parameter to select one model only)
epsilon_s = 1e-2 # Stommel
epsilon_m = 1e-5 # Munk

# construct square grid (NxN)
N = 100
dx = 1/N
x = np.linspace(dx,1-dx,N)
y = np.linspace(dx,1-dx,N)
xx,yy = np.meshgrid(x,y)

# wind forcing
rhs = np.sin(np.pi*yy)

# friction: linear (Stommel) and/or harmonic (Munk)
A = spoisson.poisson2d(N)
A = epsilon_s*A/dx**2 - epsilon_m*A*A/dx**4

# beta v
B = A.tocsr()
for j in range(0,N):
  for i in range(1,N-1):
    k = i + N*j
    B[k,k+1] += 0.5/dx
    B[k,k-1] -= 0.5/dx
    
  # points close to boundaries
  k = N*j
  B[k,k+1]   += 0.5/dx
  k = N-1 + N*j
  B[k,k-1] -= 0.5/dx

# solve linear system
sol = spoisson.sol(rhs, mat=B)

plt.figure()
plt.contour(xx,yy,sol, 10,colors='k',linewidths=1)
