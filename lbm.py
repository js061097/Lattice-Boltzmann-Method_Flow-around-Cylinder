import numpy as np
import csv

#Initialization of the parameters
nx          = 400   #number of points in x axis
ny          = 100   #number of points in y axis
rho_0        = 100  #initial density
tau         = 0.6   #relaxation parameter
nt          = 100   #timesteps

#Initialization of the Lattice weights and velocities
n = 9
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])
idxs = np.arange(n)
cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
X, Y = np.meshgrid(range(nx), range(ny))

#Initial Conditions
F = np.ones((ny,nx,n)) #Change it and try
F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/nx*4))
rho = np.sum(F,2)
for i in idxs:
  F[:,:,i] *= rho_0 / rho

#Boundary
cylinder = (X - nx/4)**2 + (Y - ny/2)**2 < (ny/4)**2


#Simulation with time
for it in range(nt):
  #Drift step
  for i, cx, cy in zip(idxs, cxs, cys):
    F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
    F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
  
  #Reflective Boundary Condition
  bndryF = F[cylinder,:]
  bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
  
  #Evaluation of velocities and densities
  rho = np.sum(F,2)
  ux  = np.sum(F*cxs,2) / rho
  uy  = np.sum(F*cys,2) / rho
  
  #Collision step
  Feq = np.zeros(F.shape)
  for i, cx, cy, w in zip(idxs, cxs, cys, weights):
    Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)

  F += -(1.0/tau) * (F - Feq)
  
  #Boundary Condition
  F[cylinder,:] = bndryF

#Writing the final velocities to csv files
#Add this to the simulation and display it for obtaining the dynamic output
with open('velocity_x.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(ux)
                   
with open('velocity_y.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(uy)
