import numpy as np
import casadi

class Loss:
    def __init__(self, n, m):
        self.xi = casadi.SX.sym("xi", n+m)

    def getLossValue(self, traj_t, traj_oc_t):

        self.loss = casadi.norm_2(self.xi-traj_oc_t)**2
        self.dLdXi = casadi.jacobian(self.loss, self.xi)
        self.lossFun = casadi.Function("lossFun", [self.xi], [self.loss])
        self.dLdXiFun = casadi.Function("dLdXiFun", [self.xi], [self.dLdXi])

        lossNow = self.lossFun(traj_t).full()[0, 0]
        dLdXiNow = self.dLdXiFun(traj_t).full()[0,0]

        return lossNow, dLdXiNow

