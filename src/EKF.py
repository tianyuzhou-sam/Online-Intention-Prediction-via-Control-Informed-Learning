import numpy as np

class EKF:

    def __init__(self):
        pass

    def predict(self, theta_prev, P_prev, Q_prev):
        self.theta_ = theta_prev
        self.P_ = P_prev + Q_prev
        
    def update(self, dLdtheta, R, stageLoss):
        p = np.shape(dLdtheta)[1] #size of theta
        S = np.matmul(np.matmul(dLdtheta, self.P_), np.transpose(dLdtheta)) + R
        kalmanGain = np.matmul(np.matmul(self.P_, np.transpose(dLdtheta)), np.linalg.inv(S)) #Kalman Gain
        self.P = np.matmul((np.eye(p) - np.matmul(kalmanGain, dLdtheta)), self.P_) #Update P
        self.theta = self.theta_ - np.matmul(kalmanGain,stageLoss).flatten()
        return self.theta
    
if __name__ == '__main__':
    theta_prev = np.array((1, 2, 3))
    P_prev = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
    Q_prev = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]) 
    dLdtheta = np.array((1, 2, 3)) 
    R = 0.001
    stageLoss = 0.1

    updateTheta = EKF()
    updateTheta.predict(theta_prev, P_prev, Q_prev)
    updateTheta.update(dLdtheta, R, stageLoss)
    print(updateTheta.theta)