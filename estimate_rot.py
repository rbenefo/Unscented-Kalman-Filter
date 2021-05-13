import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from convert_data import Calibrate
from convert_data import extract_euler_angles

from UKF import UKF
import os
import matplotlib.pyplot as plt
import sys
#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1, submission = True):
    #load data
    if submission:
        imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")) 
    else:
        imu = io.loadmat(os.path.join(os.path.dirname(__file__), "hw2_p2_data/imu/imuRaw" + str(data_num) + ".mat")) 
 
    time_vec = np.array(imu['ts'])
    accel = imu['vals'][0:3,:].astype(np.float64)
    gyro = imu['vals'][3:6,:].astype(np.float64)

    accel, gyro = Calibrate(accel, gyro).calibrate()

    # sys.exit()

    ukf = UKF()
    mu_data, cov_data = ukf.run(accel, gyro, time_vec, submission)
    #mu_data is phi, theta, psi--> roll, pitch, yaw

    #alpha --> psi (yaw); beta --> theta (pitch); gamma --> phi (roll)
    roll=mu_data[:,0] #0
    pitch=mu_data[:,1] #1

    yaw=mu_data[:,2]

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw, cov_data


if __name__ == "__main__":
    dataset =  3
    roll, pitch,yaw, cov_data= estimate_rot(dataset, submission=False)
    imu = io.loadmat(os.path.join(os.path.dirname(__file__), "hw2_p2_data/imu/imuRaw" + str(dataset) + ".mat")) 
    ###GYRO
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.scatter(imu["ts"],yaw, s = 4, label="Yaw")
    ax2.scatter(imu["ts"],roll, s = 4,label="Roll")
    ax3.scatter(imu["ts"],pitch, s=4,label="Pitch")


    ###VICON
    vicon = io.loadmat('hw2_p2_data/vicon/viconRot'+str(dataset)+'.mat')

    vicon_rotation_data = vicon["rots"]

    data = extract_euler_angles(vicon_rotation_data)
    ax1.scatter(vicon["ts"], data[0], color="blue", s=4,label="Vicon Yaw")
    ax2.scatter(vicon["ts"], data[2], color="red",s = 4, label="Vicon Roll")
    ax3.scatter(vicon["ts"], data[1], color="orange",s=4, label="Vicon Pitch")

    # plt.ylim(-1,1)
    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_ylabel("Radians")
    ax2.set_ylabel("Radians")
    ax3.set_ylabel("Radians")

    fig.tight_layout()
    plt.show()

    # plt.plot(imu["ts"], np.array(cov_data))
    # plt.show()