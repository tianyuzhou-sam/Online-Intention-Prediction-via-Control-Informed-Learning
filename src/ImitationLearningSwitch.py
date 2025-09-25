import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os, shutil
import sys
import time
import math
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
sys.path.append(os.getcwd() + '/src/')
import Env
from EKF import EKF
from Loss_function import Loss
import copy


class ImitationLearning:
    def __init__(self, project="", init_state=None, true_theta=None, dynsys=None, trueSys=None, dt=None, horizon=None, H=None, MemoryTime=None, window=None, noise=None, pred_init=None , saveFlag=False):

        self.saveFlag = saveFlag
        self.plotTrajFlag = False
        self.printFlag = True
        if saveFlag:
            if not os.path.exists("results/"):
                os.mkdir("results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.dynsys = dynsys
        self.trueSys = trueSys
        self.num_dyn_auxvar = dynsys.dyn_auxvar.shape[0]
        self.num_cost_auxvar = dynsys.cost_auxvar.shape[0]
        self.dt = dt
        self.demo_horizon = horizon
        self.init_state = init_state
        self.noise = noise
        self.pred_init = pred_init
        self.window = window
        # ------------------------------ get demos data ------------------------------
        self.true_theta = true_theta
        self.demoSys = PDP.OCSys()
        self.demoSys.setAuxvarVariable(vertcat(self.trueSys.dyn_auxvar, self.trueSys.cost_auxvar))
        self.demoSys.setControlVariable(self.trueSys.U)
        self.demoSys.setStateVariable(self.trueSys.X)
        self.truedyn = self.trueSys.X + self.dt * self.trueSys.f
        self.demoSys.setDyn(self.truedyn)
        self.demoSys.setPathCost(self.trueSys.path_cost)
        self.demoSys.setFinalCost(self.trueSys.final_cost)

        self.demo_traj = self.demoSys.ocSolver(ini_state=init_state, horizon=horizon, auxvar_value = [])

        if self.printFlag:
            print('True theta', self.true_theta)

        # ------------------------------ initialize Classes ------------------------------
        self.sysoc = PDP.OCSys()
        self.sysoc.setAuxvarVariable(vertcat(self.dynsys.dyn_auxvar, self.dynsys.cost_auxvar))
        self.sysoc.setControlVariable(self.dynsys.U)
        self.sysoc.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.sysoc.setDyn(self.dyn)
        self.sysoc.setPathCost(self.dynsys.path_cost)
        self.sysoc.setFinalCost(self.dynsys.final_cost)
        self.sysoc.diffPMP()
        self.lqr_solver = PDP.LQR()

        # ------------------------------ initilize tunable parameter ------------------------------
        self.sigma = 0.
        self.theta = np.zeros(self.sysoc.n_auxvar)
        self.theta[-len(self.pred_init):] = self.pred_init
        self.H = H
        self.MemoryTime = MemoryTime

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj_original = self.demo_traj['state_traj_opt']
        self.demo_control_traj = self.demo_traj['control_traj_opt']
        # if project == 'uniform':
        # self.demo_state_traj = self.demo_state_traj_original + (np.random.random((self.demo_horizon+1,len(init_state)))-0.5)*noise*2
        # if project == 'normal':
        self.demo_state_traj = self.demo_state_traj_original + np.random.normal(0, noise, (self.demo_horizon+1,len(init_state)))
        self.ref_traj = list()

        # ------------------------------ other setup ------------------------------
        self.iteration = 1
        self.Loss_his = []
        self.theta_error = []
        self.goal_error = []
        self.data_time = []
        self.gradient_time = []
        self.ekf_time = []
        self.x_his = []
        self.u_his = []

        self.dt = dt
        self.switch_time = [window]
        self.switch_flag = 0
        while self.switch_time[-1]+window < horizon:
            self.switch_time.append(self.switch_time[-1] + window)
        self.switch_goal = []


    # def set_sigma(self, sigma):
    #     self.sigma = sigma
    #     self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2

    def set_iteration(self, iteration):
        self.iteration = iteration

    def set_sigma(self, sigma):
        self.sigma = sigma

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def initialize_parameter(self):
        # self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2
        for idx in range(len(self.true_theta)-len(self.pred_init)):
            self.theta[idx] = self.true_theta[idx] + self.sigma * (np.random.random() - 0.5)*self.true_theta[idx]
        # print(self.theta)

    def predict_horizon(self, H):
        self.H = H

    def set_goal_range(self, goal_range):
        self.goal_range = goal_range

    def switch_target(self, switch_time, switch_goal):
        self.switch_time = switch_time
        self.switch_goal = switch_goal
        self.switch_flag = 0

    def solve(self):
        self.theta_his = [self.theta]
        # if self.printFlag:
        #     print('theta = ', self.theta)
        init_state = self.init_state
        for iter in range(self.iteration):
            for idx in range(self.demo_horizon):

                if idx == self.switch_time[self.switch_flag]:
                    
                    # goal_position = self.switch_goal[self.switch_flag]
                    angle = (np.random.random()-0.5) * 2 * np.pi
                    goal_position = [self.goal_range*np.cos(angle)+self.demo_state_traj[self.switch_time[self.switch_flag]][0], self.goal_range*np.sin(angle)+self.demo_state_traj[self.switch_time[self.switch_flag]][1], 1]
                    self.switch_goal += [goal_position]
                    goal_v_I = np.array([0,0,0])
                    goal_q = Env.toQuaternion(0, [0,0,1])
                    goal_w_B = np.array([0,0,0])

                    # if self.switch_flag == 0:
                    #     self.true_theta[:-len(self.pred_init)] = [1, 1, 1, 1, 0.4, 0.01, 0.1, 12, 0.8, 5, 0.8]
                    # else:
                    #     self.true_theta[:-len(self.pred_init)] = [1, 1, 1, 1, 0.4, 0.01, 0.1, 8, 1.2, 5, 1.2]

                    self.true_theta[-len(self.pred_init):] = [goal_position[0], goal_position[1], goal_position[2], goal_v_I[0], goal_v_I[1], goal_v_I[2], goal_q[0], goal_q[1], goal_q[2], goal_q[3], goal_w_B[0], goal_w_B[1], goal_w_B[2]]

                    self.trueSys = Env.Quadrotor()
                    self.trueSys.initDyn(self.true_theta[0], self.true_theta[1], self.true_theta[2], self.true_theta[3], self.true_theta[4], self.true_theta[5])
                    self.trueSys.initCost(self.true_theta[6], self.true_theta[7], self.true_theta[8], self.true_theta[9], self.true_theta[10], goal_position, goal_v_I, goal_q, goal_w_B)

                    self.demoSys = PDP.OCSys()
                    self.demoSys.setAuxvarVariable(vertcat(self.trueSys.dyn_auxvar, self.trueSys.cost_auxvar))
                    self.demoSys.setControlVariable(self.trueSys.U)
                    self.demoSys.setStateVariable(self.trueSys.X)
                    self.truedyn = self.trueSys.X + self.dt * self.trueSys.f
                    self.demoSys.setDyn(self.truedyn)
                    self.demoSys.setPathCost(self.trueSys.path_cost)
                    self.demoSys.setFinalCost(self.trueSys.final_cost)

                    self.new_traj = self.demoSys.ocSolver(ini_state=self.demo_state_traj[self.switch_time[self.switch_flag]], horizon=self.demo_horizon-self.switch_time[self.switch_flag], auxvar_value = [])

                    self.demo_state_traj_original[self.switch_time[self.switch_flag]:] = self.new_traj['state_traj_opt']
                    self.demo_control_traj[self.switch_time[self.switch_flag]:] = self.new_traj['control_traj_opt']
                    self.demo_state_traj = self.demo_state_traj_original + np.random.normal(0, self.noise, (self.demo_horizon+1,len(init_state)))

                    if self.switch_flag < len(self.switch_time)-1:
                        self.switch_flag = self.switch_flag + 1

                    # print(self.new_traj['state_traj_opt'])
                    # print(self.true_theta)


                    


                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if idx == 0:
                    traj = self.sysoc.ocSolver(ini_state=init_state, horizon=self.H, auxvar_value = self.theta)
                    self.ref_traj = traj
                    continue
                else:
                    traj = self.sysoc.ocSolverWithRef(ini_state=init_state, horizon=self.H, auxvar_value = self.theta, ref = self.ref_traj)
                
                self.ref_traj = traj

                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                aux_sys = self.sysoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                                control_traj_opt=traj['control_traj_opt'],
                                                costate_traj_opt=traj['costate_traj_opt'],
                                                auxvar_value = self.theta)
                self.lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
                self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                            Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
                self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
                aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.H)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']


                if idx < self.MemoryTime:
                    dxdtheta_t = dxdtheta_traj[idx]
                else:
                    dxdtheta_t = dxdtheta_traj[self.MemoryTime]

                

                # dudtheta_t = dudtheta_traj[idx]
                dxidtheta_t = dxdtheta_t
                # dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

                # --------------------------- Loss function, dLdXi ---------------------------------------- 
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']

                # xi = SX.sym("xi", self.dynsys.X.shape[0]+self.dynsys.U.shape[0])
                # demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
                # current_traj = np.hstack((state_traj[idx], control_traj[idx]))

                xi = SX.sym("xi", self.dynsys.X.shape[0])
                demo_traj = (self.demo_state_traj[idx])
                if idx < self.MemoryTime:    
                    current_traj = (state_traj[idx])
                else:
                    current_traj = (state_traj[self.MemoryTime])


                loss = demo_traj - xi
                dLdXi = jacobian(loss, xi)
                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()
                if idx < self.MemoryTime:    
                    self.evaluateLoss(state_traj, control_traj, 0)
                else:
                    self.evaluateLoss(state_traj, control_traj, idx)

                self.evaluateGoalError()

                
                # if self.plotTrajFlag:
                #     self.plotTraj(state_traj, control_traj)

                # evaluate the loss
                # dldx_traj = state_traj - self.demo_state_traj
                # dldu_traj = control_traj - self.demo_control_traj
                
                # --------------------------- Chain rule ----------------------------------------
                dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)
                dp = dLdtheta

                # --------------------------- EKF ----------------------------------------
                ekf_start_time = time.time()
                updateTheta = EKF()
                updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
                updateTheta.update(dp, self.R, lossNow)
                self.ekf_time += [time.time()-ekf_start_time]
                if self.printFlag:
                    if self.iteration < 1000:
                        # print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                        # print('theta = ', updateTheta.theta)
                        # print('Loss = ', self.Loss_his[-1])
                        print('L goal = ', np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0])

                # print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                # print('theta = ', updateTheta.theta)
                # print('L goal = ', np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0])

                self.P_prev = updateTheta.P
                self.theta = updateTheta.theta
                for ndata in range(len(self.theta)-len(self.pred_init)):
                    if self.theta[ndata] < 1e-8:
                        self.theta[ndata] = 1e-8

                if idx > self.MemoryTime:
                    init_state = self.demo_state_traj[idx-self.MemoryTime]
                
                
                self.data_time += [time.time()-data_start_time]
                self.x_his += [state_traj]
                self.u_his += [control_traj]
                self.theta_his += [self.theta]
                if self.printFlag:
                    print(time.time()-data_start_time)
                # fig, axs = plt.subplots()
                # axs.plot(self.demo_state_traj_original[idx,0],self.demo_state_traj_original[idx,1],'bo')
                # axs.plot(self.demo_state_traj_original[-1,0],self.demo_state_traj_original[-1,1],'ro')
                # axs.plot(self.theta[-3],self.theta[-2],'r*')
                # axs.set_xlim([-10,10])
                # axs.set_ylim([-10,10])
                # plt.show()
                # self.plotTraj(state_traj, control_traj)

                self.H = self.demo_horizon-np.max([idx-self.MemoryTime, 0])

                target_idx = idx + self.H - self.MemoryTime
                if target_idx < self.MemoryTime:
                    target_idx = self.MemoryTime

                if target_idx > self.demo_horizon:
                    target_idx = self.demo_horizon
                # self.plot_2D_traj(state_traj, self.demo_state_traj[idx], target_idx)
                # print(self.theta)

                

        # --------------------------- learned full iter ---------------------------
        # traj = self.sysoc.ocSolver(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta)
        # state_traj = traj['state_traj_opt']
        # control_traj = traj['control_traj_opt']
        # self.evaluateLoss(state_traj, control_traj)
        # self.x_his += [state_traj]
        # self.u_his += [control_traj]
        # print(self.theta)
        print('Case ' + str(self.project) + ' Loss goal: ' + str(self.goal_error[-1]))

        # --------------------------- save all Loss ---------------------------
        if self.saveFlag:
            self.saveAll()
        if self.plotTrajFlag:
            self.plotLoss()
            # self.plotTraj(state_traj, control_traj)

        # self.animateTraj()
            

    def evaluateLoss(self, state_traj, control_traj, idx):
        Loss = 0
        loss_his = []
        for jdx in range(self.demo_horizon-idx):
            each_traj_t = state_traj[jdx]
            demo_traj_t = self.demo_state_traj_original[jdx+idx]
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        self.Loss_his += [np.asarray(Loss)[0,0]]

        # self.theta_error += [np.asarray(norm_2(self.theta[:-len(self.pred_init)]-self.true_theta[:-len(self.pred_init)])**2)[0,0]]
    def evaluateGoalError(self):
        self.goal_error += [np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0]]


    def saveAll(self):
        sio.savemat("results/result_" + self.project + ".mat", {'Loss': self.Loss_his,
                                                    'theta_error': self.theta_error, 'goal_error': self.goal_error,
                                                    'true_theta': self.true_theta, 'theta': self.theta_his,
                                                    'demo_state': self.demo_state_traj_original,
                                                    'demo_state_noise': self.demo_state_traj, 'demo_control': self.demo_control_traj,
                                                    'state': self.x_his, 'control': self.u_his,
                                                    'data_time': self.data_time, 'gradient_time': self.gradient_time,
                                                    'ekf_time': self.ekf_time, 'switch_time': self.switch_time,
                                                    'switch_goal': self.switch_goal})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.goal_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Goal Error")
        axs.set_title(self.project)
        plt.show()

        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.project)

        plt.show()

        

    def plotTraj(self, state_traj, control_traj):
        linewidth = 3
        plt.rcParams.update({'font.size': 28})
        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        for idx in range(3):
            axs[idx].plot(iter, state_traj[:,idx],'b', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj[:,idx],'g', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx],'r--', linewidth=linewidth)
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$y$")
        axs[2].set_ylabel("$z$")
        axs[0].set_xlabel("")
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[1].set_xlabel("")
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])
        axs[-1].set_xlabel("$t$")
        # axs[0].set_title("State Trajectory")
        # axs[0].legend(['Predicted State','Observed State','True State'])

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        for idx in range(3):
            axs[idx].plot(iter, state_traj[:,idx+3],'b', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj[:,idx+3],'g', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx+3],'r--', linewidth=linewidth)
        axs[0].set_ylabel("$v_x$")
        axs[1].set_ylabel("$v_y$")
        axs[2].set_ylabel("$v_z$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_xlabel("")
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[1].set_xlabel("")
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])
        # axs[0].set_title("State Trajectory")

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(4,1, figsize=(10,8))
        for idx in range(4):
            axs[idx].plot(iter, state_traj[:,idx+6],'b', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj[:,idx+6],'g', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx+6],'r--', linewidth=linewidth)
            if idx > 0:
                axs[idx].set_ylim([-1,1])
            else:
                axs[idx].set_ylim([0,2])
        axs[0].set_ylabel("$q_1$")
        axs[1].set_ylabel("$q_2$")
        axs[2].set_ylabel("$q_3$")
        axs[3].set_ylabel("$q_4$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_xlabel("")
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[1].set_xlabel("")
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])
        # axs[0].set_title("State Trajectory")

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        for idx in range(3):
            axs[idx].plot(iter, state_traj[:,idx+10],'b', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj[:,idx+10],'g', linewidth=linewidth)
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx+10],'r--', linewidth=linewidth)
            axs[idx].set_ylim([-1,1])
        axs[0].set_ylabel("$\omega_x$")
        axs[1].set_ylabel("$\omega_y$")
        axs[2].set_ylabel("$\omega_z$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_xlabel("")
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[1].set_xlabel("")
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])
        # axs[0].set_title("State Trajectory")

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        axs[1].plot(iter, state_traj[:,6],'b', linewidth=linewidth)
        axs[1].plot(iter, self.demo_state_traj[:,6],'g', linewidth=linewidth)
        axs[1].plot(iter, self.demo_state_traj_original[:,6],'r--', linewidth=linewidth)
        axs[1].set_ylim([-1,3])
        axs[1].set_ylabel("$q_1$")
        axs[2].plot(iter, state_traj[:,10],'b', linewidth=linewidth)
        axs[2].plot(iter, self.demo_state_traj[:,10],'g', linewidth=linewidth)
        axs[2].plot(iter, self.demo_state_traj_original[:,10],'r--', linewidth=linewidth)
        axs[2].set_ylim([-2,2])
        axs[2].set_ylabel("$\omega_x$")
        axs[-1].set_xlabel("$t$")
        axs[0].axis('off')  # Hide the axes
        line1, = axs[0].plot([], [], 'b', label='Predicted State', linewidth=linewidth)
        line2, = axs[0].plot([], [], 'g', label='Observed State', linewidth=linewidth)
        line3, = axs[0].plot([], [], 'r--', label='True State', linewidth=linewidth)
        axs[0].legend(loc='center')
        axs[0].set_xlabel("")
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[1].set_xlabel("")
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj,'b', linewidth=linewidth)
            axs.plot(iter, self.demo_control_traj,'r', linewidth=linewidth)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx],'b', linewidth=linewidth)
                axs[idx].plot(iter, self.demo_control_traj[:,idx],'r', linewidth=linewidth)
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()

    def plot_2D_traj(self, state_traj, current_state, target_idx):
        fig, axs = plt.subplots()
        linewidth = 3
        axs.plot(self.demo_state_traj_original[:,0], self.demo_state_traj_original[:,1],'r', linewidth=linewidth)
        axs.plot(current_state[0], current_state[1],'ro')
        axs.plot(self.demo_state_traj_original[-1,0], self.demo_state_traj_original[-1,1],'r*')
        axs.plot(state_traj[:,0], state_traj[:,1],'b', linewidth=linewidth)
        # axs.plot(self.demo_state_traj_original[target_idx,0], self.demo_state_traj_original[target_idx,1],'b*')
        plt.show()

    def animateTraj(self):
        goal_error = self.goal_error
        demo_state_noise = self.demo_state_traj
        state = self.x_his
        theta = self.theta_his
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        # ax2.set_title('2D Trajectory Animation (Top View)')
        # ax2.grid(True, alpha=0.3)

        # Set axis limits based on trajectory bounds (considering both actual and predicted)
        margin = 0.2
        x_min = min(demo_state_noise[:,0])
        x_max = max(demo_state_noise[:,0])
        y_min = min(demo_state_noise[:,1])
        y_max = max(demo_state_noise[:,1])
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax2.set_xlim([-10,10])
        ax2.set_ylim([-10,10])

        # Initialize animation elements
        trajectory_line, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Trajectory')
        predicted_line, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Prediction')
        current_point, = ax2.plot([], [], 'ro', markersize=8, label='Current State')
        goal_point, = ax2.plot([], [], 'r*', markersize=15, label='Goal')
        predicted_goal_point, = ax2.plot([], [], 'b*', markersize=12, label='Predicted Goal')

        # Add time text in top left corner
        time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=16, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add legend at right-bottom
        ax2.legend(loc='lower right')

        # Animation function
        def animate(frame):
            # Update trajectory lines (show full trajectory up to current frame)
            trajectory_line.set_data(demo_state_noise[:frame+1,0], demo_state_noise[:frame+1,1])
            
            # Update predicted trajectory (show full predicted trajectory up to current frame)
            if frame < len(state):
                predicted_traj = state[frame]
                x_pred_frame = predicted_traj[:, 0]  # x positions for this frame
                y_pred_frame = predicted_traj[:, 1]  # y positions for this frame
                predicted_line.set_data(x_pred_frame, y_pred_frame)
                
                # Extract predicted goal from theta history (elements 11-12 of the state)
                # theta contains the goal position in elements 11-12
                predicted_goal_x = theta[frame][11]  # x coordinate of predicted goal
                predicted_goal_y = theta[frame][12]  # y coordinate of predicted goal
                predicted_goal_point.set_data([predicted_goal_x], [predicted_goal_y])
            
            # Update current position point
            if frame < len(demo_state_noise):
                current_point.set_data([demo_state_noise[frame,0]], [demo_state_noise[frame,1]])
            
            # Determine current goal based on switch times
            current_goal = theta[0][-len(self.pred_init):]
            for i, switch_t in enumerate(self.switch_time):
                if frame >= switch_t:
                    current_goal = self.switch_goal[i]
            
            # Update goal position
            goal_point.set_data([current_goal[0]], [current_goal[1]])
            
            # Update time text (show time step from 0 to 120)
            time_text.set_text(f'Time: {frame}')
            
            return trajectory_line, predicted_line, current_point, goal_point, predicted_goal_point, time_text

        # Create animation
        anim = animation.FuncAnimation(fig2, animate, frames=len(demo_state_noise), 
                                    interval=150, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()

