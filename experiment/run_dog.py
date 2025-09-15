#!/usr/bin/python

import sys
import time
import math
from scipy.io import loadmat

sys.path.append('../lib/python/amd64')
import robot_interface as sdk


if __name__ == '__main__':
    # load trajetory
    data = loadmat('experiment/dog_demos.mat')
    control = data['trajectories']['control_traj_opt'][0][0]
    n_control = len(control)

    dt = data['dt'][0][0]

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    timestep = 0
    while True:
        time.sleep(dt)
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)

        cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0

        if motiontime > 50:
            if timestep < n_control:
                vx = control[timestep][0]
                vy = control[timestep][1]
                cmd.velocity = [vx, vy]
                cmd.yawSpeed = control[timestep][2]
                timestep = timestep + 1
            

        udp.SetSend(cmd)
        udp.Send()
