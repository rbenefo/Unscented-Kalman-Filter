import numpy as np
from quaternion import Quaternion
from convert_data import Calibrate

from scipy import io
import sys
import time
from copy import deepcopy

class UKF:
    def __init__(self):
        self.mu = Quaternion(1, np.array([0,0,0]))
        self.P = np.eye(3)*0.2
        self.Q = np.eye(3)*3
        self.R = np.eye(3)*1.5

    def get_sigma_points(self, dt):
        p_and_q = self.P+self.Q*dt
        S = np.linalg.cholesky(p_and_q)
        n = S.shape[1]
        sigmas = S*np.linalg.cholesky(2*n*p_and_q)

        Wi = np.hstack((sigmas, -sigmas))
        X = np.zeros((4, Wi.shape[1]))
        for i in range(Wi.shape[1]):
            qw = vector_to_quaternion(Wi[0:3, i])
            net_q = self.mu*qw

            X[:, i] = net_q.q #quaternion
        return X

    def process_model(self, xk, m_gyro, dt):
        Y = np.zeros((xk.shape))
        q_delta = vector_to_quaternion(m_gyro*dt)
        norm = np.linalg.norm(m_gyro)
        alpha = norm*dt

        if norm != 0:
            axis_e = m_gyro/norm
        else:
            axis_e = np.array([1,0,0])

        q_delta = Quaternion(np.cos(alpha/2), axis_e*np.sin(alpha/2))
        for i in range(xk.shape[1]):
            new_q = quaternion_from_array(xk[:, i])*q_delta
            # new_q.normalize() #normalize maybe unecessary...
            Y[:,i] = new_q.q
        return Y

    def forward(self,m_gyro, dt):
        X = self.get_sigma_points(dt) #Xi should be 4x2n
        Y = self.process_model(X, m_gyro, dt)
        np.save("Yi.npy", Y)
        q_guess = deepcopy(self.mu) #is deepcopy needed here?
        self.mu, W = quaternion_mean(Y, q_guess)
        np.save("Wi.npy", W)

        P = np.zeros((3,3))
        for i in range(W.shape[0]):
            P+= np.outer(W[i], W[i])
        self.P = P/ X.shape[1] #check axis

        z = self.measurement_model(Y, 0)
        zk = np.mean(z, axis = 1)
        zk /= np.linalg.norm(zk)
        return z, zk, W
    
    def measurement_model(self, Y, wk):
        g_quat = Quaternion(0, [0,0,9.81])
        z = np.zeros((3, Y.shape[1]))
        for i in range(Y.shape[1]):
            if np.count_nonzero(Y[0:4, i]) == 0: #if all 0's
                qk = quaternion_from_array(np.array([1,0,0,0]))
            else:
                qk = quaternion_from_array(Y[0:4, i])
            gprime = qk.inv()*g_quat*qk
            z[:,i] = gprime.vec()
        return z

   
    def update(self, z, zk, w, m_acc, m_gyro, dt):
        diff_zz = z.T-zk
        Pzz = np.zeros((z.shape[0], z.shape[0]))
        for i in range(diff_zz.shape[0]):
            Pzz += np.outer(diff_zz[i,:], diff_zz[i,:])
        Pzz /= z.shape[1] #want to divide by 2n

        Pvv = Pzz + self.R

        Pxz = np.zeros((z.shape[0], z.shape[0]))
        for i in range(diff_zz.shape[0]):
            Pxz += np.outer(w[i,:],diff_zz[i,:])

        Pxz /= z.shape[1] #want to divide by 2n
        vk = m_acc - zk
        K = Pxz @ np.linalg.inv(Pvv)

        q_delta = vector_to_quaternion(K@vk)
        self.mu = q_delta*self.mu

        self.P = self.P - K @ Pvv @ K.T

        return self.mu, self.P
    
    def run(self, accel_obs, gyro_obs, time_vec, submission=False):
        mu_data = np.zeros((accel_obs.shape[1],3))
        mu_data[0]=self.mu.euler_angles()
        P_data = [self.P]
        for i in range(accel_obs.shape[1]):
            if not submission:
                print("Iter {}".format(i))
            if i == accel_obs.shape[1]-1:
                dt = np.mean(time_vec[:,-5:] - time_vec[:,-6:-1])
            else:
                dt = time_vec[:,i+1] - time_vec[:,i]

            z, zk, w = self.forward(gyro_obs[:,i], dt)
            mu, cov = self.update(z, zk, w, accel_obs[:,i], gyro_obs[:,i], dt)

            mu_data[i]=mu.euler_angles()
            # print("Euler Angles {}".format(mu[0].euler_angles()))
            P_data.append(cov)
        return mu_data, P_data


##Helper functions

def quaternion_mean(q, q_guess):
    lim = 1e-3
    e_vec = np.zeros((q.shape[1], 3))
    qt = q_guess
    
    while True:
        for i in range(q.shape[1]):
            if np.count_nonzero(q[:,i]) == 0:
                qi = quaternion_from_array(np.array([1,0,0,0]))
            else:
                qi = quaternion_from_array(q[:,i])
            e_quat = qi*qt.inv()
            e_quat.normalize()
            e_vec[i] = e_quat.vec()
        mean_e = np.mean(e_vec, axis = 0)
        norm_mean_e = np.linalg.norm(mean_e)
        if norm_mean_e < lim:
            return qt, e_vec
        else:
            mean_e_quat = vector_to_quaternion(mean_e, norm_mean_e)
            mean_e_quat.normalize()
            qt = mean_e_quat*qt


def vector_to_quaternion(v, norm = None):
    if norm is None:
        norm = np.linalg.norm(v)
    if norm != 0:
        axis_e = v/norm
    else:
        axis_e = np.array([1,0,0])
    q = Quaternion(np.cos(norm/2), axis_e*np.sin(norm/2))
    return q


def quaternion_from_array(v):
    q = Quaternion(v[0], v[1:4])
    q.normalize()

    return q


if __name__ == "__main__":
    ukf = UKF()
    data_num = 1
    imu = io.loadmat('hw2_p2_data/imu/imuRaw'+str(data_num)+'.mat') 
    accel = imu['vals'][0:3,:].astype(np.float64)
    gyro = imu['vals'][3:6,:].astype(np.float64) #REMEMBER: GYRO IS NOT ANGLES; IT'S ANGULAR VELOCITY.
    T = np.shape(imu['ts'])
    time_vec = imu["ts"]
    vicon = io.loadmat('hw2_p2_data/vicon/viconRot'+str(data_num)+'.mat')

    vicon_rotation_data = vicon["rots"]
    accel, gyro_angle = Calibrate(accel, gyro).calibrate()
    mu_data, cov_data = ukf.run(accel, gyro, time_vec)

    # x = np.array([[0.7, 0.1, 0.1, 0.1], [0.7, 0.1,0.1,0.1]])

    # qguess = Quaternion(1, np.array([0,0,0]))

    # q1 = averageQuaternions(x)
    # q2, evec = quaternion_mean(x, qguess)
    # print("q1 {}".format(q1))
    # print("q2 {}".format(q2))