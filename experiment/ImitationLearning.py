import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt 
import os, shutil
import sys
import time
import math
import csv
import pandas as pd
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
sys.path.append(os.getcwd() + '/src/')
import Env
from EKF import EKF
from Loss_function import Loss
import copy


class ImitationLearning:
    def __init__(self, project="dog", init_state=None, true_theta=None, dynsys=None, dt=None, horizon=None, pred_init=None , saveFlag=False):

        if project == 'dog':
            self.start_time = 9
        elif project == 'quad':
            self.start_time = 11
        self.saveFlag = saveFlag
        self.plotTrajFlag = True
        self.printFlag = True
        if saveFlag:
            if not os.path.exists("results/"):
                os.mkdir("results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.dynsys = dynsys
        self.dt = dt
        self.demo_horizon = horizon
        self.init_state = init_state
        self.pred_init = pred_init
        
        # ------------------------------ get demos data ------------------------------
        

        if project == 'dog':
            # load csv file
            with open('experiment/dog_data.csv', 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                data = []
                for row in csv_reader:
                    # Convert string values to float
                    data.append([float(val) for val in row])
                self.demo_traj = np.array(data)
            self.demo_traj[:,0] = self.demo_traj[:,0] - self.demo_traj[0, 0]
            
            # Filter data for specific time intervals
            mask = (self.demo_traj[:,0] >= self.start_time) & (self.demo_traj[:,0] <= self.start_time + self.dt*self.demo_horizon)
            filtered_indices = np.where(mask)[0]
            # Get samples at dt=0.1 intervals

            time_points = np.arange(self.start_time, self.start_time + self.demo_horizon*self.dt + 0.1, self.dt)
            sampled_indices = []
            for t in time_points:
                # Find the closest time point in the data
                idx = np.abs(self.demo_traj[filtered_indices, 0] - t).argmin()
                sampled_indices.append(filtered_indices[idx])
            
            self.demo_traj = self.demo_traj[sampled_indices]
            self.demo_traj = self.demo_traj[:(self.demo_horizon+1),:]
            
            self.demo_state_traj = np.column_stack((self.demo_traj[:,1], self.demo_traj[:,2], self.demo_traj[:,6]))

        # if project == 'quad':
        #     data = sio.loadmat('experiment/quad_demos.mat')
        #     self.demo_state_traj_original = data['trajectories']['state_traj_opt'][0][0]
        #     self.demo_state_traj = data['trajectories']['state_traj_opt'][0][0]
        if project == 'quad':
        #     # load csv file
            with open('experiment/quad_data.csv', 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                data = []
                for row in csv_reader:
                    # Convert string values to float
                    data.append([float(val) for val in row])
                self.demo_traj = np.array(data)
            self.demo_traj[:,0] = self.demo_traj[:,0] - self.demo_traj[0, 0]
            
            # Filter data for specific time intervals
            mask = (self.demo_traj[:,0] >= self.start_time) & (self.demo_traj[:,0] <= self.start_time + self.dt*self.demo_horizon)
            filtered_indices = np.where(mask)[0]
            # Get samples at dt=0.1 intervals

            time_points = np.arange(self.start_time, self.start_time + self.demo_horizon*self.dt + 0.1, self.dt)
            sampled_indices = []
            for t in time_points:
                # Find the closest time point in the data
                idx = np.abs(self.demo_traj[filtered_indices, 0] - t).argmin()
                sampled_indices.append(filtered_indices[idx])
            
            self.demo_traj = self.demo_traj[sampled_indices]
            self.demo_traj = self.demo_traj[:(self.demo_horizon+1),:]
            
            self.demo_state_traj = np.column_stack((self.demo_traj[:,1], self.demo_traj[:,2], self.demo_traj[:,6]))
            self.init_state[0] = self.demo_state_traj[0,0]
            self.init_state[1] = self.demo_state_traj[0,1]

        # ------------------------------ initialize Classes ------------------------------
        self.sysoc = PDP.OCSys()
        if project == 'dog':
            self.sysoc.setAuxvarVariable(self.dynsys.cost_auxvar)
        if project == 'quad':
            self.sysoc.setAuxvarVariable(vertcat(self.dynsys.dyn_auxvar, self.dynsys.cost_auxvar))
        self.sysoc.setControlVariable(self.dynsys.U)
        self.sysoc.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.sysoc.setDyn(self.dyn)
        self.sysoc.setPathCost(self.dynsys.path_cost)
        self.sysoc.setFinalCost(self.dynsys.final_cost)
        self.sysoc.diffPMP()
        self.lqr_solver = PDP.LQR()

        # if project == 'dog':
        self.true_theta = true_theta
        self.theta = np.zeros(len(self.true_theta))
        if project == 'dog':
            self.true_theta = np.hstack((self.true_theta, self.demo_state_traj[-1, :2]))
        if project == 'quad':
            self.true_theta = np.hstack((self.true_theta, self.demo_state_traj[-1, :3]))


        self.sigma = 0.0
        if project == 'dog':
            self.theta = np.hstack((self.theta, self.pred_init))
        if project == 'quad':
            self.theta = np.hstack((self.theta, self.demo_state_traj[0, :3]))

        for idx in range(len(self.true_theta)-len(self.pred_init)):
            self.theta[idx] = self.true_theta[idx] + self.sigma * (np.random.random() - 0.5)*self.true_theta[idx]
        print('true theta', self.true_theta)
        print('Initial theta', self.theta)
            
        # if project == '3':
        #     self.true_theta = true_theta
        #     self.theta = np.array(true_theta)
        #     self.theta[-2] = self.pred_init[0]
        #     self.theta[-1] = self.pred_init[1]
        
        print(self.theta)
        # print(self.demo_state_traj)
        # print('True theta', self.true_theta)

        # ------------------------------ initilize tunable parameter ------------------------------


        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        
        
        
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

    # def set_sigma(self, sigma):
    #     self.sigma = sigma
    #     self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2

    def set_iteration(self, iteration):
        self.iteration = iteration

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R


    def solve(self):
        self.theta_his = [self.theta]
        # if self.printFlag:
        #     print('theta = ', self.theta)
        for iter in range(self.iteration):
            for idx in range(self.demo_horizon):
                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if idx == 0:
                    traj = self.sysoc.ocSolver(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                else:
                    traj = self.sysoc.ocSolverWithRef(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref = self.ref_traj)
                
                self.ref_traj = traj

                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                aux_sys = self.sysoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                                control_traj_opt=traj['control_traj_opt'],
                                                costate_traj_opt=traj['costate_traj_opt'],
                                                auxvar_value = self.theta)
                self.lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
                # Add small regularization to each element of Huu
                Huu_reg = [h + 1e-6*np.eye(h.shape[0]) if isinstance(h, np.ndarray) else h for h in aux_sys['Huu']]
                self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=Huu_reg, Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                            Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
                self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
                aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.demo_horizon)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']

                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]
                if self.project == 'dog':
                    dxidtheta_t = dxdtheta_t[:2]
                if self.project == 'quad':
                    dxidtheta_t = dxdtheta_t[:2]
                # dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

                # --------------------------- Loss function, dLdXi ---------------------------------------- 
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']

                if self.project == 'dog':
                    xi = SX.sym("xi", self.dynsys.X.shape[0])[:2]
                    demo_traj = self.demo_state_traj[idx][:2]
                    current_traj = state_traj[idx][:2]
                if self.project == 'quad':
                    xi = SX.sym("xi", self.dynsys.X.shape[0])[:2]
                    demo_traj = self.demo_state_traj[idx][:2]
                    current_traj = state_traj[idx][:2]

                loss = demo_traj - xi
                dLdXi = jacobian(loss, xi)
                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()
                self.evaluateLoss(state_traj, control_traj)
                
                # if self.plotTrajFlag:
                #     self.plotTraj(state_traj, control_traj)

                
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
                        print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                        # print('theta = ', updateTheta.theta)
                        # print('Loss = ', self.Loss_his[-1])
                        print('L goal = ', np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0])
                    else:
                        if(iter*self.demo_horizon+idx) % 100 == 0:
                            print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                # print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                # print('theta = ', updateTheta.theta)
                # print('L goal = ', np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0])

                self.P_prev = updateTheta.P
                self.theta = updateTheta.theta
                # if self.project == '1':
                for ndata in range(len(self.theta)-len(self.pred_init)):
                    if self.theta[ndata] < 1e-8:
                        self.theta[ndata] = 1e-8

                self.data_time += [time.time()-data_start_time]
                self.x_his += [state_traj]
                self.u_his += [control_traj]
                self.theta_his += [self.theta]
                if self.printFlag:
                    print(time.time()-data_start_time)

                # self.plotTraj(state_traj, control_traj)
                # self.plot_demo(idx)

                

        # --------------------------- learned full iter ---------------------------
        traj = self.sysoc.ocSolverWithRef(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref = self.ref_traj)
        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']
        self.evaluateLoss(state_traj, control_traj)
        self.x_his += [state_traj]
        self.u_his += [control_traj]
        print(self.theta)
        print('Case ' + str(self.project) + ' Loss goal: ' + str(self.goal_error[-1]))

        # --------------------------- save all Loss ---------------------------
        if self.saveFlag:
            self.saveAll()
        if self.plotTrajFlag:
            self.plotLoss()
            self.plotTraj(state_traj, control_traj)
            

    def evaluateLoss(self, state_traj, control_traj):
        Loss = 0
        loss_his = []
        for jdx in range(self.demo_horizon):
            each_traj_t = np.hstack((state_traj[jdx,:2]))
            demo_traj_t = np.hstack((self.demo_state_traj[jdx,:2]))
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        self.Loss_his += [np.asarray(Loss)[0,0]]

        # self.theta_error += [np.asarray(norm_2(self.theta[:-len(self.pred_init)]-self.true_theta[:-len(self.pred_init)])**2)[0,0]]
        self.goal_error += [np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0]]


    def saveAll(self):
        sio.savemat("results/result_" + self.project + ".mat", {'Loss': self.Loss_his,
                                                    'theta_error': self.theta_error, 'goal_error': self.goal_error,
                                                    'true_theta': self.true_theta, 'theta': self.theta_his,
                                                    'demo_state_noise': self.demo_state_traj,
                                                    'state': self.x_his, 'control': self.u_his,
                                                    'data_time': self.data_time, 'gradient_time': self.gradient_time,
                                                    'ekf_time': self.ekf_time})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.goal_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Goal Error")
        axs.set_title(self.project)

        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.project)

        fig, axs = plt.subplots()
        axs.plot(self.theta_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Theta Error")
        axs.set_title(self.project)
        plt.show()

        

    def plotTraj(self, state_traj, control_traj):

        # iter = [*range(len(state_traj))]
        # fig, axs = plt.subplots(len(state_traj[0]),1)
        # for idx in range(len(state_traj[0])):
        #     axs[idx].plot(iter, state_traj[:,idx],'b')
        #     axs[idx].plot(iter, self.demo_state_traj[:,idx],'g')
        #     axs[idx].plot(iter, self.demo_state_traj_original[:,idx],'r')
        #     axs[idx].set_ylabel("x"+str(idx+1))
        # axs[-1].set_xlabel("Iteration")
        # axs[0].set_title("State Trajectory")

        
        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(2,1)
        for idx in range(2):
            axs[idx].plot(iter, state_traj[:,idx],'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx],'r--')
            # if self.project == 'quad':
            #     axs[idx].plot(iter, self.demo_state_traj_original[:,idx],'y--')
            # axs[idx].set_ylabel("x"+str(idx+1))
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$y$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_title("State Trajectory")
        axs[0].legend(['Predicted State','Observed State'])

        # iter = [*range(len(state_traj))]
        # fig, axs = plt.subplots(3,1)
        # for idx in range(3):
        #     axs[idx].plot(iter, state_traj[:,idx+3],'b')
        #     axs[idx].plot(iter, self.demo_state_traj[:,idx+3],'g')
        #     axs[idx].plot(iter, self.demo_state_traj[:,idx+3],'r--')
        #     # axs[idx].set_ylabel("x"+str(idx+4))
        # axs[0].set_ylabel("$v_x$")
        # axs[1].set_ylabel("$v_y$")
        # axs[2].set_ylabel("$v_yaw$")
        # axs[-1].set_xlabel("$t$")
        # axs[0].set_title("State Trajectory")

        # fig, axs = plt.subplots(1,1)
        # axs.plot(self.demo_state_traj[:,0], self.demo_state_traj[:,1], 'r--')
        # axs.set_xlabel("$x$")
        # axs.set_ylabel("$y$")
        # axs.set_title("State Trajectory")

        plt.show()

    def plot_demo(self, iter):
        fig, axs = plt.subplots(1,1)
        axs.plot(self.demo_state_traj[:iter,0], self.demo_state_traj[:iter,1], 'b')
        axs.plot(self.theta[-2], self.theta[-1], 'r*')
        axs.set_title("State Trajectory")
        axs.set_xlim(-4, 4)
        axs.set_ylim(-3, 3)
        plt.show()


class ImitationLearningNN:
    def __init__(self, project="dog", init_state=None, true_theta=None, dynsys=None, dt=None, horizon=None, pred_init=None , saveFlag=False):

        if project == 'dog':
            self.start_time = 9
        elif project == 'quad':
            self.start_time = 11
        self.saveFlag = saveFlag
        self.plotTrajFlag = True
        self.printFlag = True
        if saveFlag:
            if not os.path.exists("results/"):
                os.mkdir("results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.dynsys = dynsys
        self.dt = dt
        self.demo_horizon = horizon
        self.init_state = init_state
        self.pred_init = pred_init
        
        # ------------------------------ get demos data ------------------------------
        

        if project == 'dog':
            # load csv file
            with open('experiment/dog_data.csv', 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                data = []
                for row in csv_reader:
                    # Convert string values to float
                    data.append([float(val) for val in row])
                self.demo_traj = np.array(data)
            self.demo_traj[:,0] = self.demo_traj[:,0] - self.demo_traj[0, 0]
            
            # Filter data for specific time intervals
            mask = (self.demo_traj[:,0] >= self.start_time) & (self.demo_traj[:,0] <= self.start_time + self.dt*self.demo_horizon)
            filtered_indices = np.where(mask)[0]
            # Get samples at dt=0.1 intervals

            time_points = np.arange(self.start_time, self.start_time + self.demo_horizon*self.dt + 0.1, self.dt)
            sampled_indices = []
            for t in time_points:
                # Find the closest time point in the data
                idx = np.abs(self.demo_traj[filtered_indices, 0] - t).argmin()
                sampled_indices.append(filtered_indices[idx])
            
            self.demo_traj = self.demo_traj[sampled_indices]
            self.demo_traj = self.demo_traj[:(self.demo_horizon+1),:]
            
            self.demo_state_traj = np.column_stack((self.demo_traj[:,1], self.demo_traj[:,2], self.demo_traj[:,6]))

        # if project == 'quad':
        #     data = sio.loadmat('experiment/quad_demos.mat')
        #     self.demo_state_traj_original = data['trajectories']['state_traj_opt'][0][0]
        #     self.demo_state_traj = data['trajectories']['state_traj_opt'][0][0]
        if project == 'quad':
        #     # load csv file
            with open('experiment/quad_data.csv', 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                data = []
                for row in csv_reader:
                    # Convert string values to float
                    data.append([float(val) for val in row])
                self.demo_traj = np.array(data)
            self.demo_traj[:,0] = self.demo_traj[:,0] - self.demo_traj[0, 0]
            
            # Filter data for specific time intervals
            mask = (self.demo_traj[:,0] >= self.start_time) & (self.demo_traj[:,0] <= self.start_time + self.dt*self.demo_horizon)
            filtered_indices = np.where(mask)[0]
            # Get samples at dt=0.1 intervals

            time_points = np.arange(self.start_time, self.start_time + self.demo_horizon*self.dt + 0.1, self.dt)
            sampled_indices = []
            for t in time_points:
                # Find the closest time point in the data
                idx = np.abs(self.demo_traj[filtered_indices, 0] - t).argmin()
                sampled_indices.append(filtered_indices[idx])
            
            self.demo_traj = self.demo_traj[sampled_indices]
            self.demo_traj = self.demo_traj[:(self.demo_horizon+1),:]
            
            self.demo_state_traj = np.column_stack((self.demo_traj[:,1], self.demo_traj[:,2], self.demo_traj[:,6]))
            self.init_state[0] = self.demo_state_traj[0,0]
            self.init_state[1] = self.demo_state_traj[0,1]

        # ------------------------------ initialize Classes ------------------------------
        self.sysoc = PDP.OCSys()
        if project == 'dog':
            self.sysoc.setAuxvarVariable(self.dynsys.cost_auxvar)
        if project == 'quad':
            self.sysoc.setAuxvarVariable(vertcat(self.dynsys.dyn_auxvar, self.dynsys.cost_auxvar))
        self.sysoc.setControlVariable(self.dynsys.U)
        self.sysoc.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.sysoc.setDyn(self.dyn)
        self.sysoc.setPathCost(self.dynsys.path_cost)
        self.sysoc.setFinalCost(self.dynsys.final_cost)
        self.sysoc.diffPMP()
        self.lqr_solver = PDP.LQR()

        # if project == 'dog':
        self.true_theta = true_theta
        self.theta = np.zeros(len(self.true_theta))
        if project == 'dog':
            self.true_theta = np.hstack((self.true_theta, self.demo_state_traj[-1, :2]))
        if project == 'quad':
            self.true_theta = np.hstack((self.true_theta, self.demo_state_traj[-1, :3]))


        self.sigma = 0.4
        if project == 'dog':
            self.theta = np.hstack((self.theta, self.pred_init))
        if project == 'quad':
            self.theta = np.hstack((self.theta, self.demo_state_traj[0, :3]))

        for idx in range(len(self.true_theta)-len(self.pred_init)):
            self.theta[idx] = self.true_theta[idx] + self.sigma * (np.random.random() - 0.5)*self.true_theta[idx]
        print('true theta', self.true_theta)
        print('Initial theta', self.theta)
            
        # if project == '3':
        #     self.true_theta = true_theta
        #     self.theta = np.array(true_theta)
        #     self.theta[-2] = self.pred_init[0]
        #     self.theta[-1] = self.pred_init[1]
        
        print(self.theta)
        # print(self.demo_state_traj)
        # print('True theta', self.true_theta)

        # ------------------------------ initilize tunable parameter ------------------------------


        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        
        
        
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

    # def set_sigma(self, sigma):
    #     self.sigma = sigma
    #     self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2

    def set_iteration(self, iteration):
        self.iteration = iteration

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def initialize_nn_parameter(self, theta):
        self.theta = theta
        obj_theta = self.true_theta[-len(self.pred_init)-5:-len(self.pred_init)] 
        for idx in range(len(self.true_theta)-len(self.pred_init)-5,len(self.true_theta)-len(self.pred_init)):
            obj_theta[idx-(len(self.true_theta)-len(self.pred_init)-5)] = self.true_theta[idx] + self.sigma * (np.random.random() - 0.5)
        self.theta = np.hstack((self.theta, obj_theta, self.pred_init))

        print(self.theta)

        self.dp = np.zeros(self.theta.shape)

    def solve(self):
        print(self.theta)
        self.theta_his = [self.theta]
        # if self.printFlag:
        #     print('theta = ', self.theta)
        for iter in range(self.iteration):
            for idx in range(self.demo_horizon):
                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if idx == 0:
                    traj = self.sysoc.ocSolver(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                else:
                    traj = self.sysoc.ocSolverWithRef(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref = self.ref_traj)
                
                self.ref_traj = traj

                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                aux_sys = self.sysoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                                control_traj_opt=traj['control_traj_opt'],
                                                costate_traj_opt=traj['costate_traj_opt'],
                                                auxvar_value = self.theta)
                self.lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
                # Add small regularization to each element of Huu
                Huu_reg = [h + 1e-6*np.eye(h.shape[0]) if isinstance(h, np.ndarray) else h for h in aux_sys['Huu']]
                self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=Huu_reg, Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                            Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
                self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
                aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.demo_horizon)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']

                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]
                if self.project == 'dog':
                    dxidtheta_t = dxdtheta_t[:2]
                if self.project == 'quad':
                    dxidtheta_t = dxdtheta_t[:2]
                # dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

                # --------------------------- Loss function, dLdXi ---------------------------------------- 
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']

                if self.project == 'dog':
                    xi = SX.sym("xi", self.dynsys.X.shape[0])[:2]
                    demo_traj = self.demo_state_traj[idx][:2]
                    current_traj = state_traj[idx][:2]
                if self.project == 'quad':
                    xi = SX.sym("xi", self.dynsys.X.shape[0])[:2]
                    demo_traj = self.demo_state_traj[idx][:2]
                    current_traj = state_traj[idx][:2]

                loss = demo_traj - xi
                dLdXi = jacobian(loss, xi)
                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()
                self.evaluateLoss(state_traj, control_traj)
                
                # if self.plotTrajFlag:
                #     self.plotTraj(state_traj, control_traj)

                
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
                        print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                        # print('theta = ', updateTheta.theta)
                        # print('Loss = ', self.Loss_his[-1])
                        print('L goal = ', np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0])
                    else:
                        if(iter*self.demo_horizon+idx) % 100 == 0:
                            print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                # print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                # print('theta = ', updateTheta.theta)
                # print('L goal = ', np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0])

                self.P_prev = updateTheta.P
                self.theta = updateTheta.theta
                # if self.project == '1':
                for ndata in range(len(self.theta)-len(self.pred_init)):
                    if self.theta[ndata] < 1e-8:
                        self.theta[ndata] = 1e-8

                self.data_time += [time.time()-data_start_time]
                self.x_his += [state_traj]
                self.u_his += [control_traj]
                self.theta_his += [self.theta]
                if self.printFlag:
                    print(time.time()-data_start_time)

                # self.plotTraj(state_traj, control_traj)
                # self.plot_demo(idx)

                

        # --------------------------- learned full iter ---------------------------
        traj = self.sysoc.ocSolverWithRef(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref = self.ref_traj)
        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']
        self.evaluateLoss(state_traj, control_traj)
        self.x_his += [state_traj]
        self.u_his += [control_traj]
        print(self.theta)
        print('Case ' + str(self.project) + ' Loss goal: ' + str(self.goal_error[-1]))

        # --------------------------- save all Loss ---------------------------
        if self.saveFlag:
            self.saveAll()
        if self.plotTrajFlag:
            self.plotLoss()
            self.plotTraj(state_traj, control_traj)
            

    def evaluateLoss(self, state_traj, control_traj):
        Loss = 0
        loss_his = []
        for jdx in range(self.demo_horizon):
            each_traj_t = np.hstack((state_traj[jdx,:2]))
            demo_traj_t = np.hstack((self.demo_state_traj[jdx,:2]))
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        self.Loss_his += [np.asarray(Loss)[0,0]]

        # self.theta_error += [np.asarray(norm_2(self.theta[:-len(self.pred_init)]-self.true_theta[:-len(self.pred_init)])**2)[0,0]]
        self.goal_error += [np.asarray(norm_2(self.theta[-len(self.pred_init):]-self.true_theta[-len(self.pred_init):])**2)[0,0]]


    def saveAll(self):
        sio.savemat("results/result_" + self.project + ".mat", {'Loss': self.Loss_his,
                                                    'theta_error': self.theta_error, 'goal_error': self.goal_error,
                                                    'true_theta': self.true_theta, 'theta': self.theta_his,
                                                    'demo_state_noise': self.demo_state_traj,
                                                    'state': self.x_his, 'control': self.u_his,
                                                    'data_time': self.data_time, 'gradient_time': self.gradient_time,
                                                    'ekf_time': self.ekf_time})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.goal_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Goal Error")
        axs.set_title(self.project)

        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.project)

        fig, axs = plt.subplots()
        axs.plot(self.theta_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Theta Error")
        axs.set_title(self.project)
        plt.show()

        

    def plotTraj(self, state_traj, control_traj):

        # iter = [*range(len(state_traj))]
        # fig, axs = plt.subplots(len(state_traj[0]),1)
        # for idx in range(len(state_traj[0])):
        #     axs[idx].plot(iter, state_traj[:,idx],'b')
        #     axs[idx].plot(iter, self.demo_state_traj[:,idx],'g')
        #     axs[idx].plot(iter, self.demo_state_traj_original[:,idx],'r')
        #     axs[idx].set_ylabel("x"+str(idx+1))
        # axs[-1].set_xlabel("Iteration")
        # axs[0].set_title("State Trajectory")

        
        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(2,1)
        for idx in range(2):
            axs[idx].plot(iter, state_traj[:,idx],'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx],'r--')
            # if self.project == 'quad':
            #     axs[idx].plot(iter, self.demo_state_traj_original[:,idx],'y--')
            # axs[idx].set_ylabel("x"+str(idx+1))
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$y$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_title("State Trajectory")
        axs[0].legend(['Predicted State','Observed State'])

        # iter = [*range(len(state_traj))]
        # fig, axs = plt.subplots(3,1)
        # for idx in range(3):
        #     axs[idx].plot(iter, state_traj[:,idx+3],'b')
        #     axs[idx].plot(iter, self.demo_state_traj[:,idx+3],'g')
        #     axs[idx].plot(iter, self.demo_state_traj[:,idx+3],'r--')
        #     # axs[idx].set_ylabel("x"+str(idx+4))
        # axs[0].set_ylabel("$v_x$")
        # axs[1].set_ylabel("$v_y$")
        # axs[2].set_ylabel("$v_yaw$")
        # axs[-1].set_xlabel("$t$")
        # axs[0].set_title("State Trajectory")

        # fig, axs = plt.subplots(1,1)
        # axs.plot(self.demo_state_traj[:,0], self.demo_state_traj[:,1], 'r--')
        # axs.set_xlabel("$x$")
        # axs.set_ylabel("$y$")
        # axs.set_title("State Trajectory")

        plt.show()

    def plot_demo(self, iter):
        fig, axs = plt.subplots(1,1)
        axs.plot(self.demo_state_traj[:iter,0], self.demo_state_traj[:iter,1], 'b')
        axs.plot(self.theta[-2], self.theta[-1], 'r*')
        axs.set_title("State Trajectory")
        axs.set_xlim(-4, 4)
        axs.set_ylim(-3, 3)
        plt.show()