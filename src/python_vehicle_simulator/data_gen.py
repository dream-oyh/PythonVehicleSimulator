import matplotlib.pyplot as plt
import numpy as np
from lib.mainLoop import simulate
from vehicles.remus100 import remus100

# circle_traj = np.loadtxt("src/traj/circle.txt")
# # ref_z = circle_traj[:, 3]
# # ref_psi = circle_traj[:, 6]
# ref_rpm = 1000

# # vehicle_0 = remus100("depthHeadingAutopilot", circle_traj[0,3], circle_traj[0,6], ref_rpm, 0.5, 170)
# # 初始化 data 数组时，使用一个足够大的初始大小
# N = 100
# data = np.zeros((50 * N, 19))  # 假设每个轨迹点最多产生 100 个样本
# eta_0 = circle_traj[0, 0:6]
# start_index = 0
# end_index = 0
# for i in range(1, 20):
#     if i > 0:
#         eta_0 = data[start_index - 1, 1:7]
#         nu_0 = data[start_index - 1, 7:13]  
#         u_actual_0 = data[start_index - 1, 13:16]
#     vehicle_i = remus100(
#         "depthHeadingAutopilot", circle_traj[i,2], circle_traj[i,5], ref_rpm, 0.5, 170, nu_0, u_actual_0
#     )
#     sampleTime = 0.05  # sample time [seconds]

#     [simTime, simData] = simulate(N, sampleTime, vehicle_i, eta_0)
#     end_index = start_index + simData.shape[0]
#     # 确保 data 数组有足够的空间
#     if end_index > data.shape[0]:
#         # 扩展 data 数组的大小
#         data = np.vstack((data, np.zeros((50 * N, 19))))
#     data[start_index:end_index, :] = np.concatenate((simTime, simData), axis=1)
#     start_index = end_index
# data = [t,x,y,z,roll,pitch,yaw,u,v,w,p,q,r,u_control(3,),u_actual(3,)]

# np.savetxt("simData.txt", data, delimiter=",")

N=1000
sampleTime=0.05
eta = np.array([0, 0, 0, 0, 0, 0])
vehicle = remus100("stepInput", 30, 50, 1525, 0, 170)
[simTime, simData] = simulate(N, sampleTime, vehicle, eta)
data = np.concatenate((simTime, simData), axis=1)
np.savetxt("simData_tau.csv", data, delimiter=",")

plt.figure(1)
plt.subplot(3, 3, 1)
plt.plot(simData[:,6], label="u")
plt.legend()
plt.subplot(3, 3, 2)
plt.title("Velocity")
plt.plot(simData[:,7], label="v")
plt.legend()
plt.subplot(3, 3, 3)
plt.plot(simData[:,8], label="w")
plt.legend()
plt.subplot(3, 3, 4)
plt.plot(simData[:, 9], label="p")
plt.legend()
plt.subplot(3, 3, 5)
plt.plot(simData[:, 10], label="q")
plt.legend()
plt.subplot(3, 3, 6)
plt.plot(simData[:, 11], label="r")
plt.legend()

# Create a 3D scatter plot for vehicle's 3D position
fig = plt.figure(3)
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory with a thinner line
ax.plot(
    simData[:, 0],
    simData[:, 1],
    simData[:, 2],
    c="b",
    linewidth=0.5,
    label="Trajectory",
)

# # Plot the reference trajectory
# ax.plot(
#     circle_traj[:, 0],  # X positions
#     circle_traj[:, 1],  # Y positions
#     circle_traj[:, 2],  # Z positions
#     c="orange",
#     linewidth=0.5,
#     linestyle="--",
#     label="Reference Trajectory",
# )

# Highlight the start and end points
ax.scatter(
    simData[0, 0], simData[0, 1], simData[0, 2], c="g", marker="o", s=50, label="Start"
)
ax.scatter(
    simData[-1, 0], simData[-1, 1], simData[-1, 2], c="r", marker="o", s=50, label="End"
)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Position of Vehicle")

ax.legend()

plt.show()
