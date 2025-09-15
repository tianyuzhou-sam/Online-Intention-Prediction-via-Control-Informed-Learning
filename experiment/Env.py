from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
import math
import time


# robotic dog environment
class Dog:
    def __init__(self, project_name='my Dog'):
        self.project_name = 'my dog'

        # define the state of the quadrotor
        rx, ry = SX.sym('rx'), SX.sym('ry')
        self.r_I = vertcat(rx, ry)
        vx, vy = SX.sym('vx'), SX.sym('vy')
        self.v_I = vertcat(vx, vy)
        # yaw
        self.yaw = SX.sym('yaw')
        self.vyaw = SX.sym('vyaw')

    def initDyn(self):
        self.X = vertcat(self.r_I, self.yaw)
        self.U = vertcat(self.v_I, self.vyaw)
        f1 = self.U[0]*np.cos(self.X[2]) - self.U[1]*np.sin(self.X[2])
        f2 = self.U[0]*np.sin(self.X[2]) + self.U[1]*np.cos(self.X[2])
        f3 = self.U[2]
        self.f = vertcat(f1,f2,f3)

    def initDyn2(self):
        self.X = vertcat(self.r_I, self.yaw, self.v_I)
        self.ax, self.ay = SX.sym('ax'), SX.sym('ay')
        self.U = vertcat(self.ax, self.ay, self.vyaw)
        f1 = self.v_I[0]*np.cos(self.X[2]) - self.v_I[1]*np.sin(self.X[2])
        f2 = self.v_I[0]*np.sin(self.X[2]) + self.v_I[1]*np.cos(self.X[2])
        f3 = self.U[2]
        f4 = self.U[0]
        f5 = self.U[1]
        self.f = vertcat(f1,f2,f3,f4,f5)

    def initCost(self, wx=None, wd=None, wu1=None, wu2=None, wu3=None, goal=None):

        parameter = []
        if wx is None:
            self.wx = SX.sym('wx')
            parameter += [self.wx]
        else:
            self.wx = wx


        if wd is None:
            self.wd = SX.sym('wd')
            parameter += [self.wd]
        else:
            self.wd = wd

        if wu1 is None:
            self.wu1 = SX.sym('wu1')
            parameter += [self.wu1]
        else:
            self.wu1 = wu1
        
        if wu2 is None:
            self.wu2 = SX.sym('wu2')
            parameter += [self.wu2]
        else:
            self.wu2 = wu2

        if wu3 is None:
            self.wu3 = SX.sym('wu3')
            parameter += [self.wu3]
        else:
            self.wu3 = wu3

        if goal is None:
            xg, yg = SX.sym('xg'), SX.sym('yg')
            self.goal_r_I = vertcat(xg, yg)
            parameter += [xg]
            parameter += [yg]
        else:
            self.goal_r_I = goal



        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        self.cost_r_I = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)



        # the thrust cost
        self.cost_u1 = dot(self.U[0], self.U[0])
        self.cost_u2 = dot(self.U[1], self.U[1])
        self.cost_u3 = dot(self.U[2], self.U[2])

        # obstacle
        p = [0., -2.5]
        self.cost_d = dot(self.r_I - p, self.r_I - p)

        # self.path_cost = self.wr * self.cost_r_I + \
        #                  self.wv * self.cost_v_I + \
        #                  self.ww * self.cost_w_B + \
        #                  self.wq * self.cost_q + \
        #                  wthrust * self.cost_thrust
        self.path_cost = self.wu1 * self.cost_u1 + \
                         self.wu2 * self.cost_u2 + \
                         self.wu3 * self.cost_u3 - \
                         self.wd * self.cost_d
        self.final_cost = self.wx * self.cost_r_I 
                        #   self.wv * self.cost_v_I

    def initCost2(self, wx=None, wv=None, wd=None, wd2=None, wu1=None, wu2=None, wu3=None, goal=None):

        parameter = []
        if wx is None:
            self.wx = SX.sym('wx')
            parameter += [self.wx]
        else:
            self.wx = wx

        if wv is None:
            self.wv = SX.sym('wv')
            parameter += [self.wv]
        else:
            self.wv = wv

        if wd is None:
            self.wd = SX.sym('wd')
            parameter += [self.wd]
        else:
            self.wd = wd
        
        if wd2 is None:
            self.wd2 = SX.sym('wd2')
            parameter += [self.wd2]
        else:
            self.wd2 = wd2

        if wu1 is None:
            self.wu1 = SX.sym('wu1')
            parameter += [self.wu1]
        else:
            self.wu1 = wu1
        
        if wu2 is None:
            self.wu2 = SX.sym('wu2')
            parameter += [self.wu2]
        else:
            self.wu2 = wu2

        if wu3 is None:
            self.wu3 = SX.sym('wu3')
            parameter += [self.wu3]
        else:
            self.wu3 = wu3

        if goal is None:
            xg, yg = SX.sym('xg'), SX.sym('yg')
            self.goal_r_I = vertcat(xg, yg)
            parameter += [xg]
            parameter += [yg]
        else:
            self.goal_r_I = goal


        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        self.cost_r_I = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)



        # the thrust cost
        self.cost_u1 = dot(self.U[0], self.U[0])
        self.cost_u2 = dot(self.U[1], self.U[1])
        self.cost_u3 = dot(self.U[2], self.U[2])

        # obstacle
        p1 = [-1., -1]
        p2 = [1, 0.5]
        self.cost_d = dot(self.r_I - p1, self.r_I - p1)
        self.cost_d2 = dot(self.r_I - p2, self.r_I - p2)

        bound = 2

        self.cost_d = if_else(self.cost_d > bound, bound, self.cost_d)
        self.cost_d2 = if_else(self.cost_d2 > bound, bound, self.cost_d2)

        # self.path_cost = self.wr * self.cost_r_I + \
        #                  self.wv * self.cost_v_I + \
        #                  self.ww * self.cost_w_B + \
        #                  self.wq * self.cost_q + \
        #                  wthrust * self.cost_thrust
        self.path_cost = self.wu1 * self.cost_u1 + \
                         self.wu2 * self.cost_u2 + \
                         self.wu3 * self.cost_u3 - \
                         self.wd * self.cost_d - \
                         self.wd2 * self.cost_d2
        self.final_cost = self.wx * self.cost_r_I 
                        #   self.wv * self.cost_v_I

    def initCostNN(self, hidden_layers, goal=None):
        layers = hidden_layers + [1]

        a = vertcat(self.X, self.U)
        cost_auxvar = []
        Ak = SX.sym('Ak', layers[0], self.X.size()[0] + self.U.size()[0])
        bk = SX.sym('bk', layers[0])

        cost_auxvar += [Ak.reshape((-1,1))]
        cost_auxvar += [bk]
        a = mtimes(Ak, a) + bk

        for i in range(len(layers)-1):
            a = 0.5 * a * (1 + tanh(sqrt(2 / np.pi) * (a + 0.044715 * a ** 3)))
            Ak = SX.sym('Ak', layers[i+1], layers[i])
            bk = SX.sym('bk', layers[i+1])

            cost_auxvar += [Ak.reshape((-1,1))]
            cost_auxvar += [bk]
            a = mtimes(Ak, a) + bk

        if goal is None:
            xg, yg = SX.sym('xg'), SX.sym('yg')
            self.goal_r_I = vertcat(xg, yg)
            cost_auxvar += [xg]
            cost_auxvar += [yg]
        else:
            self.goal_r_I = goal

        self.cost_auxvar = vcat(cost_auxvar)
        self.n_cost_auxvar = self.cost_auxvar.shape[0]
        
        self.path_cost = norm_2(a)**2
        self.final_cost = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

    def initCostPoly(self, wgoal=None, w_xsq=None, w_x=None, w_ysq=None, w_y=None, wxy=None, wu=None, goal=None):
        parameter = []

        if wgoal is None:
            self.wgoal = SX.sym('wgoal')
            parameter += [self.wgoal]
        else:
            self.wgoal = wgoal

        # features for x
        if w_xsq is None:
            self.w_xsq = SX.sym('w_xsq')
            parameter += [self.w_xsq]
        else:
            self.w_xsq = w_xsq

        self.feature_xsq = 0.5 * self.r_I[0] * self.r_I[0]

        if w_x is None:
            self.w_x = SX.sym('w_x')
            parameter += [self.w_x]
        else:
            self.w_x = w_x

        self.feature_x = self.r_I[0]

        # features for y
        if w_ysq is None:
            self.w_ysq = SX.sym('w_ysq')
            parameter += [self.w_ysq]
        else:
            self.w_ysq = w_ysq

        self.feature_ysq = 0.5 * self.r_I[1] * self.r_I[1]

        if w_y is None:
            self.w_y = SX.sym('w_y')
            parameter += [self.w_y]
        else:
            self.w_y = w_y

        self.feature_y = self.r_I[1]

        if wxy is None:
            self.wxy = SX.sym('wxy')
            parameter += [self.wxy]
        else:
            self.wxy = wxy

        self.feature_xy = self.r_I[0] * self.r_I[1]

        if wu is None:
            self.wu = SX.sym('wu')
            parameter += [self.wu]
        else:
            self.wu = wu

        self.feature_u = dot(self.U, self.U)

        if goal is None:
            xg, yg = SX.sym('xg'), SX.sym('yg')
            self.goal_r_I = vertcat(xg, yg)
            parameter += [xg]
            parameter += [yg]
        else:
            self.goal_r_I = goal

        self.cost_auxvar = vcat(parameter)

        self.cost_r_I = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        self.path_cost = self.w_xsq * self.feature_xsq + self.w_x * self.feature_x + \
                         self.w_ysq * self.feature_ysq + self.w_y * self.feature_y + \
                         self.wxy * self.feature_xy + \
                         self.wu * self.feature_u + \
                         0*self.wgoal * self.cost_r_I

        # tune weights for final cost
        self.final_cost = self.wgoal * self.cost_r_I

# quadrotor (UAV) environment
class Quadrotor:
    def __init__(self, project_name='my UAV'):
        self.project_name = 'my uav'

        # define the state of the quadrotor
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0'), SX.sym('q1'), SX.sym('q2'), SX.sym('q3')
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1'), SX.sym('f2'), SX.sym('f3'), SX.sym('f4')
        self.T_B = vertcat(f1, f2, f3, f4)

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None, c=None):
        # global parameter
        g = 9.81

        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx')
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy')
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz')
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass')
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        if c is None:
            self.c = SX.sym('c')
            parameter += [self.c]
        else:
            self.c = c

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = vertcat(Mx, My, Mz)

        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(pinv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initNeuralDyn(self, hidden_layers):
        # Use neural network to represent the dynamic.
        # Note that here we use auxvar to denote the parameter of the neural dynamic
        layers = hidden_layers + [self.X.shape[0]]

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B

        # construct the neural policy with the argument inputs to specify the hidden layers of the neural policy
        a = vertcat(self.X, self.U)
        dyn_auxvar = []
        Ak = SX.sym('Ak', layers[0], self.X.size()[0]+self.U.size()[0])  # weights matrix
        bk = SX.sym('bk', layers[0])  # bias vector
        dyn_auxvar += [Ak.reshape((-1, 1))]
        dyn_auxvar += [bk]
        a = mtimes(Ak, a) + bk
        for i in range(len(layers)-1):
            # a = tanh(a)
            a = 1/(1+exp(-a))
            # a = a/(1+exp(-a))
            # a = exp(-a**2)
            # a = sin(a)
            # a = log(1+exp(a))
            # a = if_else(a > 0, a, 0)
            # a = if_else(a > 0, a, exp(a)-1)
            # a = 0.5*a*(1+tanh(sqrt(2/pi)*(a+0.044715*a**3)))
            Ak = SX.sym('Ak', layers[i+1], layers[i])  # weights matrix
            bk = SX.sym('bk', layers[i+1])  # bias vector
            dyn_auxvar += [Ak.reshape((-1, 1))]
            dyn_auxvar += [bk]
            a = mtimes(Ak, a) + bk
        self.dyn_auxvar=vcat(dyn_auxvar)
        self.n_auxvar = self.dyn_auxvar.shape[0]
        self.f = a

    def initCost(self, wr=None, wv=None, wq=None, ww=None, wd=None, goal=None, wthrust=0.1):

        parameter = []
        if wr is None:
            self.wr = SX.sym('wr')
            parameter += [self.wr]
        else:
            self.wr = wr

        if wv is None:
            self.wv = SX.sym('wv')
            parameter += [self.wv]
        else:
            self.wv = wv

        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq

        if ww is None:
            self.ww = SX.sym('ww')
            parameter += [self.ww]
        else:
            self.ww = ww

        if wd is None:
            self.wd = SX.sym('wd')
            parameter += [self.wd]
        else:
            self.wd = wd
        
        if goal is None:
            xg, yg, zg = SX.sym('xg'), SX.sym('yg'), SX.sym('zg')
            self.goal_r_I = vertcat(xg, yg, zg)
            parameter += [xg]
            parameter += [yg]
            parameter += [zg]
        else:
            self.goal_r_I = goal

        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        self.cost_r_I = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        # goal velocity
        goal_v_I = np.array([0, 0, 0])
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # final attitude error
        goal_q = toQuaternion(0, [0, 0, 1])
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        goal_w_B = np.array([0, 0, 0])
        self.cost_w_B = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

        p = [0., 5]
        self.cost_d = dot(self.r_I[:2] - p, self.r_I[:2] - p)

        height = 0.6
        self.cost_z = (self.r_I[2] - height) ** 2

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        # self.path_cost = self.wr * self.cost_r_I + \
        #                  self.wv * self.cost_v_I + \
        #                  self.ww * self.cost_w_B + \
        #                  self.wq * self.cost_q + \
        #                  wthrust * self.cost_thrust - \
        #                  self.wd * self.cost_d
        self.path_cost = wthrust * self.cost_thrust - \
                         self.wd * self.cost_d + \
                         100 * self.cost_z
        self.final_cost = self.wr * self.cost_r_I + \
                          self.wv * self.cost_v_I + \
                          self.ww * self.cost_w_B + \
                          self.wq * self.cost_q

    # def initCost(self, QuadDesiredStates: QuadStates, wr=None, wv=None, wq=None, ww=None, wthrust=0.1):

    #     # load the goal states
    #     goal_r_I = np.array(QuadDesiredStates.position)
    #     goal_v_I = np.array(QuadDesiredStates.velocity)
    #     goal_q = QuadDesiredStates.attitude_quaternion
    #     goal_w_B = QuadDesiredStates.angular_velocity

    #     parameter = []
    #     if wr is None:
    #         self.wr = SX.sym('wr')
    #         parameter += [self.wr]
    #     else:
    #         self.wr = wr

    #     if wv is None:
    #         self.wv = SX.sym('wv')
    #         parameter += [self.wv]
    #     else:
    #         self.wv = wv

    #     if wq is None:
    #         self.wq = SX.sym('wq')
    #         parameter += [self.wq]
    #     else:
    #         self.wq = wq

    #     if ww is None:
    #         self.ww = SX.sym('ww')
    #         parameter += [self.ww]
    #     else:
    #         self.ww = ww

    #     self.cost_auxvar = vcat(parameter)

    #     # goal position in the world frame
    #     self.cost_r_I = dot(self.r_I - goal_r_I, self.r_I - goal_r_I)

    #     # goal velocity
    #     self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

    #     # final attitude error
    #     goal_R_B_I = self.dir_cosine(goal_q)
    #     R_B_I = self.dir_cosine(self.q)
    #     self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

    #     # auglar velocity cost
    #     self.cost_w_B = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

    #     # the thrust cost
    #     self.cost_thrust = dot(self.T_B, self.T_B)

    #     self.path_cost = self.wr * self.cost_r_I + \
    #                      self.wv * self.cost_v_I + \
    #                      self.ww * self.cost_w_B + \
    #                      self.wq * self.cost_q + \
    #                      wthrust * self.cost_thrust
    #     self.final_cost = self.wr * self.cost_r_I + \
    #                       self.wv * self.cost_v_I + \
    #                       self.ww * self.cost_w_B + \
    #                       self.wq * self.cost_q

    # def initCost2(self, QuadDesiredStates: QuadStates, wthrust=0.1):

    #     # load the goal states
    #     goal_r_I = np.array(QuadDesiredStates.position)
    #     goal_v_I = np.array(QuadDesiredStates.velocity)
    #     goal_q = QuadDesiredStates.attitude_quaternion
    #     goal_w_B = QuadDesiredStates.angular_velocity

    #     parameter = []

    #     self.wrx = SX.sym('wrx')
    #     parameter += [self.wrx]
    #     self.wry = SX.sym('wry')
    #     parameter += [self.wry]
    #     self.wrz = SX.sym('wrz')
    #     parameter += [self.wrz]

    #     self.wvx = SX.sym('wvx')
    #     parameter += [self.wvx]
    #     self.wvy = SX.sym('wvy')
    #     parameter += [self.wvy]
    #     self.wvz = SX.sym('wvz')
    #     parameter += [self.wvz]

    #     self.wwx = SX.sym('wwx')
    #     parameter += [self.wwx]
    #     self.wwy = SX.sym('wwy')
    #     parameter += [self.wwy]
    #     self.wwz = SX.sym('wwz')
    #     parameter += [self.wwz]

    #     self.wq = SX.sym('wq')
    #     parameter += [self.wq]

    #     self.cost_auxvar = vcat(parameter)

    #     # goal position in the world frame
    #     self.cost_r_I_x = (self.r_I[0] - goal_r_I[0]) ** 2
    #     self.cost_r_I_y = (self.r_I[1] - goal_r_I[1]) ** 2
    #     self.cost_r_I_z = (self.r_I[2] - goal_r_I[2]) ** 2

    #     # goal velocity
    #     self.cost_v_I_x = (self.v_I[0] - goal_v_I[0]) ** 2
    #     self.cost_v_I_y = (self.v_I[1] - goal_v_I[1]) ** 2
    #     self.cost_v_I_z = (self.v_I[2] - goal_v_I[2]) ** 2

    #     # final attitude error
    #     goal_R_B_I = self.dir_cosine(goal_q)
    #     R_B_I = self.dir_cosine(self.q)
    #     self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

    #     # auglar velocity cost
    #     self.cost_w_B_x = (self.w_B[0] - goal_w_B[0]) ** 2
    #     self.cost_w_B_y = (self.w_B[1] - goal_w_B[1]) ** 2
    #     self.cost_w_B_z = (self.w_B[2] - goal_w_B[2]) ** 2

    #     # the thrust cost
    #     self.cost_thrust = dot(self.T_B, self.T_B)

    #     self.path_cost = self.wrx * self.cost_r_I_x + self.wry * self.cost_r_I_y + self.wrz * self.cost_r_I_z + \
    #                      self.wvx * self.cost_v_I_x + self.wvy * self.cost_v_I_y + self.wvz * self.cost_v_I_z + \
    #                      self.wwx * self.cost_w_B_x + self.wwy * self.cost_w_B_y + self.wwz * self.cost_w_B_z + \
    #                      self.wq * self.cost_q + \
    #                      wthrust * self.cost_thrust
    #     self.final_cost = self.wrx * self.cost_r_I_x + self.wry * self.cost_r_I_y + self.wrz * self.cost_r_I_z + \
    #                       self.wvx * self.cost_v_I_x + self.wvy * self.cost_v_I_y + self.wvz * self.cost_v_I_z + \
    #                       self.wwx * self.cost_w_B_x + self.wwy * self.cost_w_B_y + self.wwz * self.cost_w_B_z + \
    #                       self.wq * self.cost_q

    def initCost_Polynomial(self, w_xsq=None, w_x=None, w_ysq=None, w_y=None, w_zsq=None, w_z=None, goal_r_I=None, w_thrust=0.1):

        # load the goal states
        goal_v_I = [0,0,0]
        goal_q = [0,0,0,1]
        goal_w_B = [0,0,0]

        parameter = []
        # goal aspect
        
        # velocity aspect
        self.cost_goal_v = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # orientation aspect
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_goal_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # angular aspect
        self.cost_goal_w = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

        # thrust aspect
        self.cost_thrust = dot(self.T_B, self.T_B)

        # features for x
        if w_xsq is None:
            self.w_xsq = SX.sym('w_xsq')
            parameter += [self.w_xsq]
        else:
            self.w_xsq = w_xsq

        self.feature_xsq = 0.5 * self.r_I[0] * self.r_I[0]
        if w_x is None:
            self.w_x = SX.sym('w_x')
            parameter += [self.w_x]
        else:
            self.w_x = w_x
        self.feature_x = self.r_I[0]

        # features for y
        if w_ysq is None:
            self.w_ysq = SX.sym('w_ysq')
            parameter += [self.w_ysq]
        else:
            self.w_ysq = w_ysq
        self.feature_ysq = 0.5 * self.r_I[1] * self.r_I[1]

        if w_y is None:
            self.w_y = SX.sym('w_y')
            parameter += [self.w_y]
        else:
            self.w_y = w_y
        self.feature_y = self.r_I[1]

        # features for z
        if w_zsq is None:   
            self.w_zsq = SX.sym('w_zsq')
            parameter += [self.w_zsq]
        else:
            self.w_zsq = w_zsq
        self.feature_zsq = 0.5 * self.r_I[2] * self.r_I[2]

        if w_z is None:
            self.w_z = SX.sym('w_z')
            parameter += [self.w_z]
        else:
            self.w_z = w_z
        self.feature_z = self.r_I[2]

        if goal_r_I is None:
            self.goal_x = SX.sym('goal_x')
            parameter += [self.goal_x]
            self.goal_y = SX.sym('goal_y')
            parameter += [self.goal_y]
            self.goal_z = SX.sym('goal_z')
            parameter += [self.goal_z]
            self.goal_r_I = vertcat(self.goal_x, self.goal_y, self.goal_z)
        else:
            self.goal_r_I = goal_r_I

        self.cost_goal_r = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        self.cost_auxvar = vcat(parameter)

        # # feature for xy
        # self.w_xy = SX.sym('w_xy')
        # self.feature_xy = self.r_I[0] * self.r_I[1]
        # parameter += [self.w_xy]

        self.path_cost = self.w_xsq * self.feature_xsq + self.w_x * self.feature_x + \
                         self.w_ysq * self.feature_ysq + self.w_y * self.feature_y + \
                         self.w_zsq * self.feature_zsq + self.w_z * self.feature_z + \
                         w_thrust * self.cost_thrust

        # tune weights for final cost
        self.final_cost = 1 * self.cost_goal_r + \
                          11 * self.cost_goal_v + \
                          100 * self.cost_goal_q + \
                          10 * self.cost_goal_w


        self.cost_auxvar = vcat(parameter)

    def get_quadrotor_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len / 2, 0, 0)
        r2 = vertcat(0, -wing_len / 2, 0)
        r3 = vertcat(-wing_len / 2, 0, 0)
        r4 = vertcat(0, wing_len / 2, 0)

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 15))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]

            # q here is a 1D list, no matter which type state_traj is (numpy 2d array or 2d list)
            if abs(np.linalg.norm(q)) > 1e-6:
                q = np.array(q) / np.linalg.norm(q)

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = r1_pos
            position[t, 6:9] = r2_pos
            position[t, 9:12] = r3_pos
            position[t, 12:15] = r4_pos

        return position

    def play_animation(self, wing_len, state_traj, file_name_prefix: str, space_limits: list, save_option: bool, state_traj_ref=None, dt=0.1, title='UAV Maneuvering',
                       horizon=1, waypoints=None):

        # plot
        # params = {'axes.labelsize': 25,
        #           'axes.titlesize': 25,
        #           'xtick.labelsize': 20,
        #           'ytick.labelsize': 20,
        #           'legend.fontsize': 16}
        # plt.rcParams.update(params)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=15, labelpad=15)
        ax.set_ylabel('Y (m)', fontsize=15, labelpad=15)
        ax.set_zlabel('Z (m)', fontsize=15, labelpad=15)
        #ax.set_zlim(0, 12)
        #ax.set_ylim(-9, 9)
        #ax.set_xlim(-9, 9)
        self.set_axes_equal_all(ax, space_limits)
        ax.set_title('UAV manuvering', pad=15, fontsize=20)
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.55, 0.50, "time", transform=ax.transAxes, fontsize=15)

        # fig = plt.figure(figsize=(7,6))
        # ax = fig.add_subplot(1, 1, 1, projection='3d', )
        # ax.set_xlabel('X', fontsize=40, labelpad=10)
        # ax.set_ylabel('Y', fontsize=40, labelpad=10)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_ylim(-9, 9)
        # ax.set_xlim(-9, 9)
        # ax.set_xticks(np.arange(-8, 9, 4))
        # ax.set_yticks(np.arange(-8, 9, 8))
        # ax.tick_params(labelbottom=False, labelright=False, labelleft=False)
        # ax.view_init(elev=88, azim=-90)
        # ax.set_title('Top view', fontsize=40, pad=-25)
        # ax.set_position([-0.17, -0.12, 1.30, 1.15])
        # time_template = 'time = %.1fs'
        # time_text = ax.text2D(0.55, 0.20, "time", transform=ax.transAxes, fontsize=0)


        # fig = plt.figure(figsize=(7,6))
        # ax = fig.add_subplot(1, 1, 1, projection='3d', )
        # ax.set_xlabel('X', fontsize=40, labelpad=10)
        # ax.set_zlabel('Z', fontsize=40, labelpad=10)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_ylim(-9, 9)
        # ax.set_xlim(-9, 9)
        # ax.set_zlim(0, 8)
        # ax.set_zticks(np.arange(0, 8, 2))
        # ax.set_xticks(np.arange(-8, 9, 4))
        # # ax.set_yticks(np.arange(-8, 9, 8))
        # ax.tick_params(labelbottom=False, labelright=False, labelleft=False)
        # # ax.view_init(elev=0, azim=-90)
        # ax.set_title('Front view', fontsize=40, pad=-25)
        # ax.set_position([-0.17, -0.12, 1.30, 1.15])
        # time_template = 'time = %.1fs'
        # time_text = ax.text2D(0.55, 0.20, "time", transform=ax.transAxes, fontsize=0)

        # draw the obstacles
        # bar2_back = ax.bar3d([-1], [-3], [0], dx=[0.5], dy=[0.5], dz=[4.5], color='#D95319')
        # bar2_top = ax.bar3d([-1], [-3], [4], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319')
        # bar2_front = ax.bar3d([-1], [-6.5], [4.5], dx=[0.5], dy=[0.5], dz=[-4.5], color='#D95319')
        #
        # bar1_front = ax.bar3d([2.5], [2], [1.5], dx=[0.5], dy=[0.5], dz=[4.5], color='#D95319')
        # bar1_bottom = ax.bar3d([2.5], [2], [1.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
        # bar1_top = ax.bar3d([2.5], [2.5], [5.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
        # bar1_back = ax.bar3d([2.5], [6.5], [6.0], dx=[0.5], dy=[0.5], dz=[-4.5], color='#D95319')

        if waypoints is not None:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=80, zorder=10000, color='red', alpha=1,
                       marker='^')

        # data
        position = self.get_quadrotor_position(wing_len, state_traj)
        sim_horizon = np.size(position, 0)
        time_interval = float(horizon / sim_horizon)

        if state_traj_ref is None:
            position_ref = self.get_quadrotor_position(0, numpy.zeros_like(position))
        else:
            position_ref = self.get_quadrotor_position(wing_len, state_traj_ref)

        # animation
        line_traj, = ax.plot(position[:1, 0], position[:1, 1], position[:1, 2])
        c_x, c_y, c_z = position[0, 0:3]
        r1_x, r1_y, r1_z = position[0, 3:6]
        r2_x, r2_y, r2_z = position[0, 6:9]
        r3_x, r3_y, r3_z = position[0, 9:12]
        r4_x, r4_y, r4_z = position[0, 12:15]
        line_arm1, = ax.plot(np.array([c_x, r1_x]), np.array([c_y, r1_y]), np.array([c_z, r1_z]),
            linewidth=4, color='blue', marker='o', markersize=4, markerfacecolor='black')
        line_arm2, = ax.plot(np.array([c_x, r2_x]), np.array([c_y, r2_y]), np.array([c_z, r2_z]),
            linewidth=4, color='red', marker='o', markersize=4, markerfacecolor='black')
        line_arm3, = ax.plot(np.array([c_x, r3_x]), np.array([c_y, r3_y]), np.array([c_z, r3_z]),
            linewidth=4, color='blue', marker='o', markersize=4, markerfacecolor='black')
        line_arm4, = ax.plot(np.array([c_x, r4_x]), np.array([c_y, r4_y]), np.array([c_z, r4_z]),
            linewidth=4, color='red', marker='o', markersize=4, markerfacecolor='black')

        line_traj_ref, = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
        c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
        r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
        r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
        r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
        r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
        line_arm1_ref, = ax.plot(np.array([c_x_ref, r1_x_ref]), np.array([c_y_ref, r1_y_ref]), np.array([c_z_ref, r1_z_ref]),
            linewidth=2, color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm2_ref, = ax.plot(np.array([c_x_ref, r2_x_ref]), np.array([c_y_ref, r2_y_ref]), np.array([c_z_ref, r2_z_ref]),
            linewidth=2, color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm3_ref, = ax.plot(np.array([c_x_ref, r3_x_ref]), np.array([c_y_ref, r3_y_ref]), np.array([c_z_ref, r3_z_ref]),
            linewidth=2, color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm4_ref, = ax.plot(np.array([c_x_ref, r4_x_ref]), np.array([c_y_ref, r4_y_ref]), np.array([c_z_ref, r4_z_ref]),
            linewidth=2, color='gray', marker='o', markersize=3, alpha=0.7)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def update_traj(num):

            # customize
            time_text.set_text(time_template % (num * time_interval))

            # trajectory
            line_traj.set_data(position[:num, 0], position[:num, 1])
            line_traj.set_3d_properties(position[:num, 2])

            # uav
            c_x, c_y, c_z = position[num, 0:3]
            r1_x, r1_y, r1_z = position[num, 3:6]
            r2_x, r2_y, r2_z = position[num, 6:9]
            r3_x, r3_y, r3_z = position[num, 9:12]
            r4_x, r4_y, r4_z = position[num, 12:15]

            line_arm1.set_data(np.array([c_x, r1_x]), np.array([c_y, r1_y]))
            line_arm1.set_3d_properties([c_z, r1_z])

            line_arm2.set_data(np.array([c_x, r2_x]), np.array([c_y, r2_y]))
            line_arm2.set_3d_properties([c_z, r2_z])

            line_arm3.set_data(np.array([c_x, r3_x]), np.array([c_y, r3_y]))
            line_arm3.set_3d_properties([c_z, r3_z])

            line_arm4.set_data(np.array([c_x, r4_x]), np.array([c_y, r4_y]))
            line_arm4.set_3d_properties([c_z, r4_z])

            # trajectory ref
            num = sim_horizon - 1
            line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            line_traj_ref.set_3d_properties(position_ref[:num, 2])

            # uav ref
            c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]

            line_arm1_ref.set_data(np.array([c_x_ref, r1_x_ref]), np.array([c_y_ref, r1_y_ref]))
            line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])

            line_arm2_ref.set_data(np.array([c_x_ref, r2_x_ref]), np.array([c_y_ref, r2_y_ref]))
            line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])

            line_arm3_ref.set_data(np.array([c_x_ref, r3_x_ref]), np.array([c_y_ref, r3_y_ref]))
            line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])

            line_arm4_ref.set_data(np.array([c_x_ref, r4_x_ref]), np.array([c_y_ref, r4_y_ref]))
            line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            return line_traj, line_arm1, line_arm2, line_arm3, line_arm4, \
                   line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=80, blit=True, cache_frame_data=False)

        if save_option == True:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save(file_name_prefix + '.gif', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )


    def set_axes_equal_all(self, ax, space_limits: list):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        Reference: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = space_limits[0]
        y_limits = space_limits[1]
        z_limits = space_limits[2]

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



        
# converter to quaternion from (angle, direction)
def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()


# normalized verctor
def normalizeVec(vec):
    if type(vec) == list:
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    return vec


def quaternion_conj(q):
    conj_q = q
    conj_q[1] = -q[1]
    conj_q[2] = -q[2]
    conj_q[3] = -q[3]
    return conj_q