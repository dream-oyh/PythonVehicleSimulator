import pandas as pd
import numpy as np
from vehicles.remus100 import remus100
import matplotlib.pyplot as plt
from lib.gnc import attitudeEuler
data = pd.read_csv("simData_tau.csv").to_numpy()
timestamps = data[:, 0]
eta = data[:, 1:7] # x,y,z,roll,pitch,yaw
nu = data[:, 7:13] # u,v,w,p,q,r
tau = data[:, 13:19]
N=1000
sampleTime=0.05
vehicle = remus100("stepInput", 30, 50, 1525, 0, 170)

eta_pred = np.zeros((len(eta),6))
nu_pred = np.zeros((len(nu),6))


eta_pred[0,:] = eta[0,:]
nu_pred[0,:] = nu[0,:]

for i in range(len(eta)-1):
    nu_pred[i+1] = vehicle.forward(eta_pred[i],nu_pred[i],tau[i],sampleTime)
    eta_pred[i+1] = attitudeEuler(eta_pred[i],nu_pred[i],sampleTime)


plt.figure(1)
plt.subplot(3, 3, 1)
plt.plot(nu_pred[:,0], label="u")
plt.plot(nu[:,0], label="u_true")
plt.legend()
plt.subplot(3, 3, 2)
plt.title("Velocity")
plt.plot(nu_pred[:,1], label="v")
plt.plot(nu[:,1], label="v_true")
plt.legend()
plt.subplot(3, 3, 3)
plt.plot(nu_pred[:,2], label="w")
plt.plot(nu[:,2], label="w_true")
plt.legend()
plt.subplot(3, 3, 4)
plt.plot(nu_pred[:,3], label="p")
plt.plot(nu[:,3], label="p_true")
plt.legend()
plt.subplot(3, 3, 5)
plt.plot(nu_pred[:,4], label="q")
plt.plot(nu[:,4], label="q_true")
plt.legend()
plt.subplot(3, 3, 6)
plt.plot(nu_pred[:,5], label="r")
plt.plot(nu[:,5], label="r_true")
plt.legend()

fig = plt.figure(3)
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory with a thinner line
ax.plot(
    eta_pred[:, 0],
    eta_pred[:, 1],
    eta_pred[:, 2],
    c="b",
    linewidth=0.5,
    label="Trajectory",
)
ax.scatter(
    eta_pred[0, 0], eta_pred[0, 1], eta_pred[0, 2], c="g", marker="o", s=50, label="Start"
)
ax.scatter(
    eta_pred[-1, 0], eta_pred[-1, 1], eta_pred[-1, 2], c="r", marker="o", s=50, label="End"
)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Position of Vehicle")

ax.legend()

plt.show()