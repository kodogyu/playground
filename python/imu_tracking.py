import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import cumtrapz
# # Sample rate and time
# sample_rate = 10  # Hz
# dt = 1.0 / sample_rate
# total_time = 10  # seconds
# print(f"total time: {total_time}")
# time = np.linspace(0, total_time, sample_rate * total_time)

# Read IMU values
time_list = []
gyro_data_list = []
accel_data_list = []

csv_file = "/home/kodogyu/Datasets/Euroc/V2_01_easy/V2_01_easy/mav0/imu0/data.csv"
# csv_file = "/home/kodogyu/Datasets/PAIR360/Calibration/Cam-Imu calib/trial1/gyro.csv"
print(f"csv file: {csv_file}")

with open(csv_file, 'r') as f:
    csv_reader = csv.reader(f)

    next(csv_reader)  # skip header
    limit = np.inf
    i = 0
    for row in csv_reader:
        time, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = row

        time_list.append(int(time)/1_000_000_000)
        gyro_data_list.append([float(gyro_x), float(gyro_y), float(gyro_z)])
        accel_data_list.append([float(acc_x), float(acc_y), float(acc_z)])

        i += 1
        if i > limit:
            break


# # Simulated IMU data
# # Replace these with your actual IMU data
# accel_data = np.random.randn(len(time), 3)  # Simulated accelerometer data (ax, ay, az)
# gyro_data = np.random.randn(len(time), 3)   # Simulated gyroscope data (gx, gy, gz)
time = np.array(time_list)
gyro_data = np.array(gyro_data_list)
accel_data = np.array(accel_data_list)

print(f"dt: {time[1] - time[0]}")

# Complementary filter parameters
alpha = 1.0

# Initial orientation estimate
orientation = np.zeros((len(time), 3))

# # Orientation estimation using complementary filter
# for t in range(1, len(time)):
#     dt = time[t] - time[t - 1]
#     accel_angle = np.arctan2(accel_data[t, 1], accel_data[t, 2])
#     gyro_angle = orientation[t - 1, 0] + gyro_data[t, 0] * dt

#     orientation[t, 0] = alpha * gyro_angle + (1 - alpha) * accel_angle

#     accel_angle = np.arctan2(accel_data[t, 0], accel_data[t, 2])
#     gyro_angle = orientation[t - 1, 1] + gyro_data[t, 1] * dt

#     orientation[t, 1] = alpha * gyro_angle + (1 - alpha) * accel_angle
for t in range(1, len(time)):
    dt = time[t] - time[t - 1]
    orientation[t] = orientation[t-1] + gyro_data[t-1] * dt

# Convert accelerometer data to global frame using estimated orientation
accel_global = np.zeros_like(accel_data)
for t in range(len(time)):
    ax, ay, az = accel_data[t]
    roll, pitch, yaw = orientation[t]
    
    # Rotation matrix from body to world frame
    R = np.array([
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
        [np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw), np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw), np.sin(roll) * np.cos(pitch)],
        [np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw), np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw), np.cos(roll) * np.cos(pitch)]
    ])
    accel_global[t] = R @ np.array([ax-9.81, ay, az])  # 중력
    # accel_global[t] = R @ np.array([ax, ay, az]) - np.array([9.81, 0., 0.])  # 중력

# Integrate to get velocity and position
velocity = cumtrapz(accel_global, dx=dt, initial=0, axis=0)
position = cumtrapz(velocity, dx=dt, initial=0, axis=0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(time, accel_data)
plt.title('Accelerometer Data')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend(['ax', 'ay', 'az'])

plt.subplot(3, 1, 2)
plt.plot(time, velocity)
plt.title('Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(['vx', 'vy', 'vz'])

plt.subplot(3, 1, 3)
plt.plot(time, position)
plt.title('Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend(['px', 'py', 'pz'])

plt.tight_layout()
# plt.show()

print(f"position shape: {position.shape}")

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(position.T[0], position.T[1], position.T[2], marker='x')
ax.scatter(*position[0], color='red')
plt.show()