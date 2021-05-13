import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from quaternion import Quaternion
import sys

def extract_euler_angles(R):
    """Alpha is yaw angle, Beta is pitch,
    and Gamma is roll"""
    alpha = np.arctan2(R[1,0], R[0,0])
    beta = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+R[2,2]**2))
    gamma = np.arctan2(R[2,1],R[2,2])

    return alpha, beta, gamma

class Calibrate:
    def __init__(self, accel, gyro):
        self.accel = accel
        self.gyro = gyro
        self.Vref = 3300

    def calibrate_accel(self):
        accel = np.zeros((self.accel.shape))
        self.accel[0] =  -self.accel[0]
        self.accel[1] = -self.accel[1]
        self.accel[2] = self.accel[2]
        
        accel_sensitivity = 34.58
        accel_scale = self.Vref/1023/accel_sensitivity
        
        accel_bias = self.accel[:,0] - (np.array([0,0,9.81])/accel_scale)
        accel[0] = (self.accel[0]-accel_bias[0])*accel_scale
        accel[1] = (self.accel[1]-accel_bias[1])*accel_scale
        accel[2] = (self.accel[2]-accel_bias[2])*accel_scale
        return accel

    def calibrate_gyro(self, debug = False):
        gyro = np.zeros((self.gyro.shape))
        gyro[0]= self.gyro[1]
        gyro[1] = self.gyro[2]
        gyro[2] = self.gyro[0]

        gyro_sensitivity = 3.3*180/np.pi
        gyro_scale = self.Vref/1023/gyro_sensitivity
        
        gyro_bias = gyro[:,0]
        gyro[0] = (gyro[0]-gyro_bias[0])*gyro_scale
        gyro[1] = (gyro[1]-gyro_bias[1])*gyro_scale
        gyro[2] = (gyro[2]-gyro_bias[2])*gyro_scale

        # gyro = np.zeros((self.gyro.shape))
        if debug:
            gyro_bias = self.gyro[:,0]
            gyro[0]=(self.gyro[0]-gyro_bias[0])*gyro_scale #yaw #for dataset 8, yaw falls below -pi...
            gyro[1]=(self.gyro[1]-gyro_bias[1])*gyro_scale #roll
            gyro[2]=(self.gyro[2]-gyro_bias[2])*gyro_scale #pitch
        # else:
        #     gyro[0]=(self.gyro[1]-gyro_bias[1])*0.000150 #wx
        #     gyro[1]=(self.gyro[2]-gyro_bias[2])*0.000150 #wy
        #     gyro[2]=(self.gyro[0]-gyro_bias[0])*0.000150 #wz
        return gyro

    def calibrate(self, debug=False):
        accel = self.calibrate_accel()
        gyro = self.calibrate_gyro(debug)
        return accel, gyro

if __name__ == "__main__":
    data_num = 3
    imu = io.loadmat('hw2_p2_data/imu/imuRaw'+str(data_num)+'.mat') 
    accel = imu['vals'][0:3,:].astype(np.float64)
    gyro = imu['vals'][3:6,:].astype(np.float64) #REMEMBER: GYRO IS NOT ANGLES; IT'S ANGULAR VELOCITY.
    T = np.shape(imu['ts'])[1]
    vicon = io.loadmat('hw2_p2_data/vicon/viconRot'+str(data_num)+'.mat')

    vicon_rotation_data = vicon["rots"]
    accel, gyro = Calibrate(accel, gyro).calibrate(debug=True)

    gyro_angle = integrate.cumtrapz(gyro, np.arange(len(gyro[0])), initial=0, axis = 1)
    


    # # ###VICON
    data = np.degrees(extract_euler_angles(vicon_rotation_data))
    data_diff = np.diff(data)

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    # ax1.scatter(vicon["ts"], data[0], color="blue", s=4,label="Vicon Yaw")
    # ax3.scatter(vicon["ts"], data[1], color="orange",s=4, label="Vicon Pitch")
    # ax2.scatter(vicon["ts"], data[2], color="red",s = 4, label="Vicon Roll")

    # ax1.scatter(vicon["ts"][:,1:], data_diff[0], color="blue", s=4,alpha=0.4, label="Vicon Yaw")
    # ax3.scatter(vicon["ts"][:,1:], data_diff[1], color="orange",s=4,alpha=0.4, label="Vicon Pitch")
    # ax2.scatter(vicon["ts"][:,1:], data_diff[2], color="red",s = 4, alpha=0.4,label="Vicon Roll")

    ###ACCELEROMETER
    ax1.scatter(imu["ts"], accel[2], c = "green", s = 4, label="Accel Z")
    ax2.scatter(imu["ts"], accel[1], c = "red", s = 4, label="-Accel Y")
    ax3.scatter(imu["ts"], accel[0], c = "orange", s = 4, label="-Accel X")

    ###GYRO


    
    # ax1.scatter(imu["ts"], gyro_angle[0], c = "green", s = 4, label="IMU 0-> Yaw")
    # ax2.scatter(imu["ts"], gyro_angle[1], c = "purple", s = 4, label="IMU 1 -> Roll")
    # ax3.scatter(imu["ts"], gyro_angle[2], c = "cyan", s = 4, label="IMU 2 -> Pitch")
    # ax1.scatter(imu["ts"], gyro[0], c = "green", s = 4, label="IMU 0-> Yaw")
    # ax2.scatter(imu["ts"], gyro[1], c = "purple", s = 4, label="IMU 1 -> Roll")
    # ax3.scatter(imu["ts"], gyro[2], c = "cyan", s = 4, label="IMU 2 -> Pitch")



    # ax1.set_ylim(-0.025,0.025)
    # ax2.set_ylim(-0.025,0.025)
    # ax3.set_ylim(-0.025,0.025)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_ylabel("Acceleration (m/s^2)")
    ax2.set_ylabel("Acceleration (m/s^2)")
    ax3.set_ylabel("Acceleration (m/s^2)")

    fig.tight_layout()

    plt.show()
