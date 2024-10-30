import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt 
import os, shutil
import sys
import time
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
sys.path.append(os.getcwd() + '/src/')
import Env
from EKF import EKF
from Loss_function import Loss


class ImitationLearning:
    def __init__(self, project="", init_state=None, true_theta=None, dynsys=None, trueSys=None, dt=None, horizon=None, noise=None, pred_init=None , saveFlag=False):

        self.saveFlag = saveFlag
        self.plotTrajFlag = True
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
        # self.theta= np.hstack((np.random.random((1,602)).flatten(),self.true_theta[-len(self.pred_init):]))
        # self.theta= np.random.random((1,3447)).flatten()
        data = sio.loadmat('theta.mat')

        # self.theta= np.hstack((data['theta'][0],self.true_theta[-len(self.pred_init):]))
        self.theta = data['theta'][0]

        print(self.theta)
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
        self.sigma = 0.5
        # self.theta = np.zeros(self.sysoc.n_auxvar)
        # self.theta[-len(self.pred_init):] = self.pred_init

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj_original = self.demo_traj['state_traj_opt']
        self.demo_control_traj = self.demo_traj['control_traj_opt']
        self.demo_state_traj = self.demo_state_traj_original + (np.random.random((self.demo_horizon+1,len(init_state)))-0.5)*noise
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

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2

    def set_iteration(self, iteration):
        self.iteration = iteration

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def initialize_parameter(self):
        self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2
    
    def initialize_nn_parameter(self):
        # self.theta = np.random.random(self.sysoc.n_auxvar-len(self.pred_init)-4)
        # self.theta = np.random.random(2257)
        # self.theta = self.true_theta
        # for idx in range(2257):
        #     self.theta[idx] = self.theta[idx] + (0.5-np.random.random())*0.01
        obj_theta = self.true_theta[-len(self.pred_init)-4:-len(self.pred_init)] + (np.random.random((1,4))-0.5)*self.sigma
        self.theta = np.hstack((self.theta, obj_theta[0], self.pred_init))
        print(self.theta)

        self.dp = np.zeros(self.theta.shape)

    def solve(self):
        self.theta_his = [self.theta]
        if self.printFlag:
            print('theta = ', self.theta)
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
                self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                            Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
                self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
                aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.demo_horizon)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']

                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]
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
                current_traj = (state_traj[idx])

                loss = demo_traj - xi
                dLdXi = jacobian(loss, xi)
                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()
                self.evaluateLoss(state_traj, control_traj)
                
                # if self.plotTrajFlag:
                #     self.plotTraj(state_traj, control_traj)

                # evaluate the loss
                dldx_traj = state_traj - self.demo_state_traj
                dldu_traj = control_traj - self.demo_control_traj
                
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

                

        # --------------------------- learned full iter ---------------------------
        traj = self.sysoc.ocSolver(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta)
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
            each_traj_t = np.hstack((state_traj[jdx], control_traj[jdx]))
            demo_traj_t = np.hstack((self.demo_state_traj_original[jdx], self.demo_control_traj[jdx]))
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
                                                    'demo_state': self.demo_state_traj_original,
                                                    'demo_state_noise': self.demo_state_traj, 'demo_control': self.demo_control_traj,
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
        plt.show()

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
        fig, axs = plt.subplots(3,1)
        for idx in range(3):
            axs[idx].plot(iter, state_traj[:,idx],'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx],'g')
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx],'r--')
            # axs[idx].set_ylabel("x"+str(idx+1))
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$y$")
        axs[2].set_ylabel("$z$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_title("State Trajectory")
        axs[0].legend(['Predicted State','Observed State','True State'])

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(3,1)
        for idx in range(3):
            axs[idx].plot(iter, state_traj[:,idx+3],'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx+3],'g')
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx+3],'r--')
            # axs[idx].set_ylabel("x"+str(idx+4))
        axs[0].set_ylabel("$v_x$")
        axs[1].set_ylabel("$v_y$")
        axs[2].set_ylabel("$v_z$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(4,1)
        for idx in range(4):
            axs[idx].plot(iter, state_traj[:,idx+6],'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx+6],'g')
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx+6],'r--')
            # axs[idx].set_ylabel("x"+str(idx+7))
            if idx > 0:
                axs[idx].set_ylim([-1,1])
            else:
                axs[idx].set_ylim([0,2])
        axs[0].set_ylabel("$q_1$")
        axs[1].set_ylabel("$q_2$")
        axs[2].set_ylabel("$q_3$")
        axs[3].set_ylabel("$q_4$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_title("State Trajectory")


        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(3,1)
        for idx in range(3):
            axs[idx].plot(iter, state_traj[:,idx+10],'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx+10],'g')
            axs[idx].plot(iter, self.demo_state_traj_original[:,idx+10],'r--')
            # axs[idx].set_ylabel("x"+str(idx+11))
            axs[idx].set_ylim([-1,1])
        axs[0].set_ylabel("$\omega_x$")
        axs[1].set_ylabel("$\omega_y$")
        axs[2].set_ylabel("$\omega_z$")
        axs[-1].set_xlabel("$t$")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj,'b')
            axs.plot(iter, self.demo_control_traj,'r')
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx],'b')
                axs[idx].plot(iter, self.demo_control_traj[:,idx],'r')
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()


