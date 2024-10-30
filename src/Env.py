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

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None, c=0.01):
        # global parameter
        g = 10

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
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initNeuralDyn(self, hidden_layers):
        # Use neural network to represent the dynamic.
        # Note that here we use auxvar to denote the parameter of the neural dynamic
        layers = hidden_layers + [self.X.shape[0]]

        g = 10
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
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

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
            a = tanh(a)
            # a = if_else(a > 0, a, 0)
            # a = 0.5*a*(1+tanh(sqrt(2/pi)*(a+0.044715*a**3)))
            Ak = SX.sym('Ak', layers[i+1], layers[i])  # weights matrix
            bk = SX.sym('bk', layers[i+1])  # bias vector
            dyn_auxvar += [Ak.reshape((-1, 1))]
            dyn_auxvar += [bk]
            a = mtimes(Ak, a) + bk
        self.dyn_auxvar=vcat(dyn_auxvar)
        self.n_auxvar = self.dyn_auxvar.shape[0]
        self.f = a


    def initCost(self, wr=None, wv=None, wq=None, ww=None, goal=None, goal_v_I=None, goal_q=None, goal_w_B=None, wthrust=0.1):

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

        if goal is None:
            xg, yg, zg = SX.sym('xg'), SX.sym('yg'), SX.sym('zg')
            self.goal_r_I = vertcat(xg, yg, zg)
            parameter += [xg]
            parameter += [yg]
            parameter += [zg]
        else:
            self.goal_r_I = goal

        if goal_v_I is None:
            xvg, yvg, zvg = SX.sym('xvg'), SX.sym('yvg'), SX.sym('zvg')
            self.goal_v_I = vertcat(xvg, yvg, zvg)
            parameter += [xvg]
            parameter += [yvg]
            parameter += [zvg]
        else:
            self.goal_v_I = goal_v_I

        if goal_q is None:
            q1, q2, q3, q4 = SX.sym('q1'), SX.sym('q2'), SX.sym('q3'), SX.sym('q4')
            self.goal_q = vertcat(q1, q2, q3, q4)
            parameter += [q1]
            parameter += [q2]
            parameter += [q3]
            parameter += [q4]
        else:
            self.goal_q = goal_q

        if goal_w_B is None:
            w1, w2, w3 = SX.sym('w1'), SX.sym('w2'), SX.sym('w3')
            self.goal_w_B = vertcat(w1, w2, w3)
            parameter += [w1]
            parameter += [w2]
            parameter += [w3]
        else:
            self.goal_w_B = goal_w_B

        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        self.cost_r_I = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        # goal velocity
        self.cost_v_I = dot(self.v_I - self.goal_v_I, self.v_I - self.goal_v_I)

        # final attitude error
        goal_R_B_I = self.dir_cosine(self.goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        self.cost_w_B = dot(self.w_B - self.goal_w_B, self.w_B - self.goal_w_B)

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        # self.path_cost = self.wr * self.cost_r_I + \
        #                  self.wv * self.cost_v_I + \
        #                  self.ww * self.cost_w_B + \
        #                  self.wq * self.cost_q + \
        #                  wthrust * self.cost_thrust
        self.path_cost = wthrust * self.cost_thrust
        self.final_cost = self.wr * self.cost_r_I + \
                          self.wv * self.cost_v_I + \
                          self.ww * self.cost_w_B + \
                          self.wq * self.cost_q
        
    def initCostNeural(self, wr=None, wv=None, wq=None, ww=None, goal=None, goal_v_I=None, goal_q=None, goal_w_B=None, wthrust=0.1):

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

        if goal is None:
            xg, yg, zg = SX.sym('xg'), SX.sym('yg'), SX.sym('zg')
            self.goal_r_I = vertcat(xg, yg, zg)
            parameter += [xg]
            parameter += [yg]
            parameter += [zg]
        else:
            self.goal_r_I = goal

        if goal_v_I is None:
            xvg, yvg, zvg = SX.sym('xvg'), SX.sym('yvg'), SX.sym('zvg')
            self.goal_v_I = vertcat(xvg, yvg, zvg)
            parameter += [xvg]
            parameter += [yvg]
            parameter += [zvg]
        else:
            self.goal_v_I = goal_v_I

        if goal_q is None:
            q1, q2, q3, q4 = SX.sym('q1'), SX.sym('q2'), SX.sym('q3'), SX.sym('q4')
            self.goal_q = vertcat(q1, q2, q3, q4)
            parameter += [q1]
            parameter += [q2]
            parameter += [q3]
            parameter += [q4]
        else:
            self.goal_q = goal_q

        if goal_w_B is None:
            w1, w2, w3 = SX.sym('w1'), SX.sym('w2'), SX.sym('w3')
            self.goal_w_B = vertcat(w1, w2, w3)
            parameter += [w1]
            parameter += [w2]
            parameter += [w3]
        else:
            self.goal_w_B = goal_w_B

        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        self.cost_r_I = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        # goal velocity
        self.cost_v_I = dot(self.v_I - self.goal_v_I, self.v_I - self.goal_v_I)

        # final attitude error
        goal_R_B_I = self.dir_cosine(self.goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        self.cost_w_B = dot(self.w_B - self.goal_w_B, self.w_B - self.goal_w_B)

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.wv * self.cost_v_I + \
                         self.ww * self.cost_w_B + \
                         self.wq * self.cost_q + \
                         wthrust * self.cost_thrust

        self.final_cost = self.wr * self.cost_r_I + \
                          self.wv * self.cost_v_I + \
                          self.ww * self.cost_w_B + \
                          self.wq * self.cost_q

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

    def play_animation(self, wing_len, state_traj, state_traj_ref=None, dt=0.1, save_option=0, title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(0, 10)
        ax.set_ylim(-8, 8)
        ax.set_xlim(-8, 8)
        ax.set_title(title, pad=20, fontsize=15)

        # target landing point
        # ax.scatter3D([0.0], [0.0], [0.0], c="r", marker="x")

        # data
        position = self.get_quadrotor_position(wing_len, state_traj)
        sim_horizon = np.size(position, 0)

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
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color='red', marker='o', markersize=3)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color='blue', marker='o', markersize=3)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color='red', marker='o', markersize=3)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color='blue', marker='o', markersize=3)

        line_traj_ref, = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
        c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
        r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
        r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
        r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
        r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
        line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            line_traj.set_data(position[:num, 0], position[:num, 1])
            line_traj.set_3d_properties(position[:num, 2])

            # uav
            c_x, c_y, c_z = position[num, 0:3]
            r1_x, r1_y, r1_z = position[num, 3:6]
            r2_x, r2_y, r2_z = position[num, 6:9]
            r3_x, r3_y, r3_z = position[num, 9:12]
            r4_x, r4_y, r4_z = position[num, 12:15]

            line_arm1.set_data(np.array([[c_x, r1_x], [c_y, r1_y]]))
            line_arm1.set_3d_properties([c_z, r1_z])

            line_arm2.set_data(np.array([[c_x, r2_x], [c_y, r2_y]]))
            line_arm2.set_3d_properties([c_z, r2_z])

            line_arm3.set_data(np.array([[c_x, r3_x], [c_y, r3_y]]))
            line_arm3.set_3d_properties([c_z, r3_z])

            line_arm4.set_data(np.array([[c_x, r4_x], [c_y, r4_y]]))
            line_arm4.set_3d_properties([c_z, r4_z])


            # trajectory ref
            num=sim_horizon-1
            line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            line_traj_ref.set_3d_properties(position_ref[:num, 2])

            # uav ref
            c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]

            line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])

            line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])

            line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])

            line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            return line_traj, line_arm1, line_arm2, line_arm3, line_arm4, \
                   line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2'+title + '.mp4', writer=writer, dpi=300)
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