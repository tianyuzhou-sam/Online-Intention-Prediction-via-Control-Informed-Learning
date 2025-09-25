import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os, shutil
import sys
import time
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
sys.path.append(os.getcwd() + '/src/')
import Env
from EKF import EKF
from Loss_function import Loss
import copy


class ImitationLearning:
    def __init__(self, project="", init_state=None, true_theta=None, dynsys=None, trueSys=None, dt=None, horizon=None, H=None, MemoryTime=None, noise=None, pred_init=None , saveFlag=False):

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
        self.sigma = 0.5
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

    def initialize_parameter(self):
        # self.theta[:-len(self.pred_init)] = self.true_theta[:-len(self.pred_init)] + self.sigma * np.random.random(len(self.true_theta)-len(self.pred_init)) - self.sigma / 2
        for idx in range(len(self.true_theta)-len(self.pred_init)):
            self.theta[idx] = self.true_theta[idx] + self.sigma * (np.random.random() - 0.5)*self.true_theta[idx]
        # print(self.theta)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_iteration(self, iteration):
        self.iteration = iteration

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def switch_target(self, switch_time, switch_goal):
        self.switch_time = switch_time
        self.switch_goal = switch_goal
        self.switch_flag = 0

    def solve(self):
        self.theta_his = [self.theta]
        init_state = self.init_state
        for iter in range(self.iteration):
            for idx in range(self.demo_horizon):
                data_start_time = time.time()

                if idx == self.switch_time[self.switch_flag]:
                    
                    goal_position = self.switch_goal[self.switch_flag]
                    goal_v_I = np.array([0,0,0])
                    goal_q = Env.toQuaternion(0, [0,0,1])
                    goal_w_B = np.array([0,0,0])

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


                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if idx == 0:
                    traj = self.sysoc.ocSolver(ini_state=self.init_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                    self.ref_traj = traj
                    continue
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
                aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.H)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']


                # dudtheta_t = dudtheta_traj[idx]
                dxidtheta_t = dxdtheta_traj[idx]
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
                # self.evaluateLoss(state_traj, control_traj)
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

                # self.plot_2D_traj(state_traj, self.demo_state_traj[idx])



                

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
                self.plotTraj(state_traj, control_traj)

            # self.animateTraj()

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

        plt.show()

    def plotTraj(self, state_traj, control_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].plot(iter, self.demo_state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj)
            axs.plot(iter, self.demo_control_traj)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx])
                axs[idx].plot(iter, self.demo_control_traj[:,idx])
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()

    def plot_2D_traj(self, state_traj, current_state):
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
        ax2.set_xlim([-6, 6])
        ax2.set_ylim([-1, 6])

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


class SysID:
    def __init__(self, project="", mode="", dynsys=None, dt=0.05, dir="", demoFile="", saveFlag=False):

        self.dir = dir
        self.saveFlag = saveFlag
        self.plotTrajFlag = False
        if saveFlag:
            if not os.path.exists(self.dir+"results/"):
                os.mkdir(self.dir+"results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.mode = mode
        self.dynsys = dynsys
        self.num_dyn_auxvar = dynsys.dyn_auxvar.shape[0]
        
        # ------------------------------ load demos data ------------------------------
        data = sio.loadmat(dir+demoFile)
        data = data[demoFile[:len(demoFile)-4]][0,0]
        # self.trajectories = data['trajectories']
        self.dt = dt
        self.true_theta = data['true_parameter'].flatten()
        self.true_theta = self.true_theta[:self.num_dyn_auxvar]
        print(data['true_parameter'].flatten())
        print(self.true_theta)

        self.n_batch = len(data['batch_inputs'])
        self.batch_inputs = []
        self.batch_states = []
        for idx in range(self.n_batch):
            self.batch_inputs += [data['batch_inputs'][idx]]
            self.batch_states += [data['batch_states'][idx]]

        # ------------------------------ initialize Classes ------------------------------
        self.sysid = PDP.SysID()
        self.sysid.setAuxvarVariable(self.dynsys.dyn_auxvar)
        self.sysid.setControlVariable(self.dynsys.U)
        self.sysid.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.sysid.setDyn(self.dyn)

        # ------------------------------ initilize tunable parameter ------------------------------
        self.sigma = 0.9
        self.initial_theta = self.true_theta + self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2
        self.theta = self.initial_theta
        print('theta = ', self.theta)

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)

        # ------------------------------ other setup ------------------------------
        self.iteration = 1
        self.Loss_his = []
        self.theta_error = []
        self.data_time = []
        self.gradient_time = []
        self.ekf_time = []

        self.theta_his = [self.theta]

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.initial_theta = self.true_theta + self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2
        self.theta = self.initial_theta

    def set_iteration(self, iteration):
        self.iteration = iteration

    def initialize_nn_parameter(self):
        self.theta = np.random.random(self.sysid.n_auxvar)
        self.dp = np.zeros(self.theta.shape)

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def solve(self):
        n_data = 0
        for iter in range(self.iteration):
            for batches in range(self.n_batch):
                input_traj = self.batch_inputs[batches]
                ini_state = self.batch_states[batches][0, :]
                horizon = np.size(self.batch_inputs[batches], 0)
                ob_state_traj = self.batch_states[batches]
                for idx in range(horizon):
                    data_start_time = time.time()
                    # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                    state_traj = self.sysid.integrateDyn(ini_state=ini_state, inputs=input_traj, auxvar_value=self.theta)
                    # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                    gradient_start_time = time.time()
                    aux_sys = self.sysid.getAuxSys(state_traj=state_traj, control_traj=input_traj, auxvar_value=self.theta)
                    aux_sol = self.sysid.integrateAuxSys(dynF=aux_sys['dynF'],
                                                dynE=aux_sys['dynE'],
                                                ini_condition=np.zeros((self.sysid.n_state, self.sysid.n_auxvar)))
                    self.gradient_time += [time.time()-gradient_start_time]
                    # --------------------------- take solution of the auxiliary control system ---------------------------
                    dxdtheta_traj = aux_sol['state_traj']
                    dxdtheta_t = dxdtheta_traj[idx]
                    dxidtheta_t = dxdtheta_t

                    # --------------------------- Loss function, dLdXi ---------------------------------------- 
                    xi = SX.sym("xi", self.dynsys.X.shape[0])
                    demo_traj = ob_state_traj[idx]
                    current_traj = state_traj[idx]

                    loss = demo_traj - xi
                    dLdXi = jacobian(loss, xi)
                    lossFun = Function("lossFun", [xi], [loss])
                    dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                    lossNow = lossFun(current_traj).full()
                    dLdXiNow = dLdXiFun(current_traj).full()

                    loss = 0
                    for jdx in range(self.n_batch):
                        input_traj_j = self.batch_inputs[jdx]
                        ini_state_j = self.batch_states[jdx][0, :]
                        horizon_j = np.size(self.batch_inputs[jdx], 0)
                        ob_state_traj_j = self.batch_states[jdx]

                        state_traj_j = self.sysid.integrateDyn(ini_state=ini_state_j, inputs=input_traj_j, auxvar_value=self.theta)
                        loss += self.evaluateLoss(state_traj_j, ob_state_traj_j, horizon_j)

                    self.Loss_his += [loss/self.n_batch]
                    # self.theta_error += [np.asarray(norm_2(self.theta-self.true_theta)**2)[0,0]]

                    self.evaluateLoss(state_traj, ob_state_traj, horizon)

                    if self.plotTrajFlag:
                        self.plotTraj(state_traj, ob_state_traj)

                    # --------------------------- Chain rule ----------------------------------------
                    dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)
                    dp = dLdtheta

                    # --------------------------- EKF ----------------------------------------
                    ekf_start_time = time.time()
                    updateTheta = EKF()
                    updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
                    updateTheta.update(dp, self.R, lossNow)
                    self.ekf_time += [time.time()-ekf_start_time]
                    if n_data < 100:
                        print('Data = ', n_data, 'Loss = ', self.Loss_his[-1])
                    else:
                        if(n_data) % 100 == 0:
                            print('Data = ', n_data, 'Loss = ', self.Loss_his[-1])
                    self.P_prev = updateTheta.P
                    self.theta = updateTheta.theta
                    self.data_time += [time.time()-data_start_time]

                    self.theta_his += [self.theta]
                    # print('Theta = ', self.theta)
                    n_data += 1

        # --------------------------- learned full iter ---------------------------
        loss = 0
        for jdx in range(self.n_batch):
            input_traj_j = self.batch_inputs[jdx]
            ini_state_j = self.batch_states[jdx][0, :]
            horizon_j = np.size(self.batch_inputs[jdx], 0)
            ob_state_traj_j = self.batch_states[jdx]

            state_traj_j = self.sysid.integrateDyn(ini_state=ini_state_j, inputs=input_traj_j, auxvar_value=self.theta)
            loss += self.evaluateLoss(state_traj_j, ob_state_traj_j, horizon_j)

        self.Loss_his += [loss/self.n_batch]

        # --------------------------- save all Loss ---------------------------
        self.plotTraj(state_traj, ob_state_traj)
        if self.saveFlag:
            self.saveAll()
        
        self.plotLoss()

    def evaluateLoss(self, state_traj, ob_state_traj, horizon):
        Loss = 0
        loss_his = []
        for jdx in range(horizon):
            each_traj_t = state_traj[jdx]
            demo_traj_t = ob_state_traj[jdx]
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        
        return np.asarray(Loss)[0,0]

    def saveEach(self, idx, traj, loss_his):
        sio.savemat(self.dir+"results/iter_"+str(idx)+".mat", {'trajectories': traj,
                                                                'losses': loss_his,
                                                                'dt': self.dt,
                                                                'theta': self.theta})

    def saveAll(self):
        horizon = (len(self.Loss_his)-1)/self.iteration
        sio.savemat(self.dir+"results/Loss_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'Loss': self.Loss_his,
                                                  'horizon': horizon})
        sio.savemat(self.dir+"results/time.mat", {'Data': self.data_time, 'Gradient': self.gradient_time,
                                                    'EKF': self.ekf_time})
        # sio.savemat(self.dir+"results/theta_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'true_theta': self.true_theta, 'theta': self.theta_his})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.mode + ": " + self.project)

        # fig, axs = plt.subplots()
        # axs.plot(self.theta_error)
        # axs.set_xlabel("Data")
        # axs.set_ylabel("Theta Error")
        # axs.set_title(self.mode + ": " + self.project)
        plt.show()

    def plotTraj(self, state_traj, ob_state_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].plot(iter, ob_state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        plt.show()


class PolicyTuning:
    def __init__(self, project="", mode="", case="", dynsys=None, nnFactor=None, dir="", demoFile="", saveFlag=False):

        self.dir = dir
        self.saveFlag = saveFlag
        self.plotTrajFlag = False
        if saveFlag:
            if not os.path.exists(self.dir+"results/"):
                os.mkdir(self.dir+"results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.mode = mode
        self.case = case
        self.dynsys = dynsys
        self.n_state = dynsys.X.size()[0]
        self.n_control = dynsys.U.size()[0]
        
        # ------------------------------ load demos data ------------------------------
        data = sio.loadmat(dir+demoFile)
        self.trajectories = data['trajectories']
        self.dt = data['dt']

        # ------------------------------ initialize Classes ------------------------------
        self.system = PDP.ControlPlanning()
        self.system.setControlVariable(self.dynsys.U)
        self.system.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.system.setDyn(self.dyn)
        self.system.setPathCost(self.dynsys.path_cost)
        self.system.setFinalCost(self.dynsys.final_cost)

        self.system.init_step_neural_policy(hidden_layers=[nnFactor*self.system.n_state,nnFactor*self.system.n_state])
        self.theta = np.random.randn(self.system.n_auxvar)
        self.nnFactor = nnFactor

        # ------------------------------ initilize tunable parameter ------------------------------
        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj = self.trajectories[0, 0]['state_traj_opt'][0, 0]
        self.demo_control_traj = self.trajectories[0, 0]['control_traj_opt'][0, 0]
        self.ini_state = self.demo_state_traj[0, :]
        self.horizon = self.demo_control_traj.shape[0]

        # ------------------------------ other setup ------------------------------
        self.iteration = 1
        self.Loss_his = []
        self.theta_error = []
        self.data_time = []
        self.gradient_time = []
        self.ekf_time = []

        self.x_his = []
        self.u_his = []

    def set_iteration(self, iteration):
        self.iteration = iteration

    def generate_traj(self, dt, horizon, ini_state):
        self.horizon = horizon
        self.dt = dt
        # ------------------------------ initialize Classes ------------------------------
        self.system = PDP.ControlPlanning()
        self.system.setControlVariable(self.dynsys.U)
        self.system.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.system.setDyn(self.dyn)
        self.system.setPathCost(self.dynsys.path_cost)
        self.system.setFinalCost(self.dynsys.final_cost)

        self.system.init_step_neural_policy(hidden_layers=[self.nnFactor*self.system.n_state,self.nnFactor*self.system.n_state])
        self.theta = np.random.randn(self.system.n_auxvar)

        # ------------------------------ initilize tunable parameter ------------------------------
        self.loss = 0
        self.dp = np.zeros(self.theta.shape)

        # ------------------------------ other setup ------------------------------
        self.Loss_his = []
        self.theta_error = []

        self.true_system = PDP.OCSys()
        self.true_system.setStateVariable(self.dynsys.X)
        self.true_system.setControlVariable(self.dynsys.U)
        self.true_system.setDyn(self.dyn)
        self.true_system.setPathCost(self.dynsys.path_cost)
        self.true_system.setFinalCost(self.dynsys.final_cost)
        self.true_sol = self.true_system.ocSolver(ini_state=ini_state, horizon=horizon)
 
        self.demo_state_traj = self.true_sol['state_traj_opt']
        self.demo_control_traj = self.true_sol['control_traj_opt']
        self.ini_state = self.demo_state_traj[0, :]
        print("Optimal cost:", self.true_sol['cost'])

        if self.saveFlag:
            sio.savemat(self.dir+"results/demo.mat", {'state_traj': self.demo_state_traj,
                                                      'control_traj': self.demo_control_traj,
                                                      'horizon': horizon,
                                                      'cost': self.true_sol['cost']})

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def solve(self):
        for iter in range(self.iteration):
            for idx in range(self.horizon):
                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                sol = self.system.integrateSys(ini_state=self.ini_state, horizon=self.horizon, auxvar_value=self.theta)
                state_traj = sol['state_traj']
                control_traj = sol['control_traj']
                cost = sol['cost']

                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                aux_sys = self.system.getAuxSys(state_traj=state_traj, control_traj=control_traj, auxvar_value=self.theta)
                # --------------------------- take solution of the auxiliary control system ---------------------------
                aux_sol = self.system.integrateAuxSys(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'],
                                            dUx=aux_sys['dUx'], dUe=aux_sys['dUe'],
                                            ini_condition=numpy.zeros((self.system.n_state, self.system.n_auxvar)))
                self.gradient_time += [time.time()-gradient_start_time]
                # --------------------------- take solution of the auxiliary control system ---------------------------
                dxdtheta_traj = aux_sol['state_traj']
                dudtheta_traj = aux_sol['control_traj']
                
                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]
                dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

                # --------------------------- take solution of the auxiliary control system ---------------------------
                dxdtheta_traj = aux_sol['state_traj']
                dudtheta_traj = aux_sol['control_traj']
                
                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]
                
                if self.case == "traj" or "Objective":
                    dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))
                    # --------------------------- Loss function, dLdXi ---------------------------------------- 
                    xi = SX.sym("xi", self.dynsys.X.shape[0]+self.dynsys.U.shape[0])
                    demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
                    current_traj = np.hstack((state_traj[idx], control_traj[idx]))
                elif self.case == "state":
                    dxidtheta_t = dxdtheta_t
                    xi = SX.sym("xi", self.dynsys.X.shape[0])
                    demo_traj = self.demo_state_traj[idx]
                    current_traj = state_traj[idx]
                else:
                    print("Case not defined!")
                    sys.exit()

                loss = demo_traj - xi
                dLdXi = jacobian(loss, xi)
                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()

                if self.case == 'Objective':
                    self.Loss_his += [cost]
                else:
                    self.evaluateLoss(state_traj, control_traj)

                if self.plotTrajFlag:
                    self.plotTraj(state_traj, control_traj)

                # --------------------------- Chain rule ----------------------------------------
                dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)
                dp = dLdtheta

                # --------------------------- EKF ----------------------------------------
                ekf_start_time = time.time()
                updateTheta = EKF()
                updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
                updateTheta.update(dp, self.R, lossNow)
                self.ekf_time += [time.time()-ekf_start_time]
                if self.iteration*self.horizon < 100:
                    print('Data = ', iter*self.horizon+idx, 'Loss = ', self.Loss_his[-1])
                else:
                    if(iter*self.horizon+idx) % 100 == 0:
                        print('Data = ', iter*self.horizon+idx, 'Loss = ', self.Loss_his[-1])
                self.P_prev = updateTheta.P
                self.theta = updateTheta.theta
                self.data_time += [time.time()-data_start_time]

                self.x_his += [state_traj]
                self.u_his += [control_traj]
                # print(time.time()-data_start_time)

        # --------------------------- save all Loss ---------------------------
        sol = self.system.integrateSys(ini_state=self.ini_state, horizon=self.horizon, auxvar_value=self.theta)
        state_traj = sol['state_traj']
        control_traj = sol['control_traj']
        cost = sol['cost']
        self.x_his += [state_traj]
        self.u_his += [control_traj]
        self.plotTraj(state_traj, control_traj)
        if self.case == 'Objective':
            self.Loss_his += [cost]
        else:
            self.evaluateLoss(state_traj, control_traj)
        if self.saveFlag:
            self.saveAll()
        
        self.plotLoss()
        return self.Loss_his

    def evaluateLoss(self, state_traj, control_traj):
        Loss = 0
        loss_his = []
        for jdx in range(self.horizon):
            each_traj_t = np.hstack((state_traj[jdx], control_traj[jdx]))
            demo_traj_t = np.hstack((self.demo_state_traj[jdx], self.demo_control_traj[jdx]))
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        
        self.Loss_his += [np.asarray(Loss)[0,0]]

    def saveEach(self, idx, traj, loss_his):
        sio.savemat(self.dir+"results/iter_"+str(idx)+".mat", {'trajectories': traj,
                                                                'losses': loss_his,
                                                                'dt': self.dt,
                                                                'theta': self.theta})

    def saveAll(self):
        sio.savemat(self.dir+"results/Loss_" + time.strftime("%Y%m%d%H%M%S") + ".mat", 
                                    {'Loss': self.Loss_his, 'horizon': self.horizon})
        sio.savemat(self.dir+"results/time.mat", {'Data': self.data_time, 'Gradient': self.gradient_time,
                                                    'EKF': self.ekf_time})

        sio.savemat(self.dir+"results/theta_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'demo_state': self.demo_state_traj, 
                                    'demo_control': self.demo_control_traj, 'state': self.x_his, 'control': self.u_his})


    def load(self, dir):
        data = sio.loadmat(dir)


    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.mode + ": " + self.project)

        plt.show()

    def plotTraj(self, state_traj, control_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].plot(iter, self.demo_state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj)
            axs.plot(iter, self.demo_control_traj)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx])
                axs[idx].plot(iter, self.demo_control_traj[:,idx])
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()
