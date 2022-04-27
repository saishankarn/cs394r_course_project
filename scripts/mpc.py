from pickle import TRUE
from tabnanny import verbose
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

class MPCLinear():
    """
    ### Inputs
        x: State
        u: Control input
        A, B: Defined by x[k+1] = Ax[k] + Bu[k]
        
    """
    def __init__(self, A, B, Q, R, T):
        self.nx = A.shape[1]
        self.nu = B.shape[1]
        
        self.A  = A
        self.B  = B
        self.Q  = Q
        self.R  = R
        self.T  = T

        self.SAFETY_DIST    = 5
        self.THROTTLE_LIMIT = 1

    def action(self, x0, xref):
        X = cp.Variable((self.nx, self.T))
        U = cp.Variable((self.nu, self.T))

        cost = 0.0
        constraints = []

    
        for t in range(self.T):
            if max(self.R.shape) == 1:
                cost += self.R*cp.square(U[:,t])
            else:
                cost += cp.quad_form(U[:,t], self.R)

            if t != 0:
                cost += cp.quad_form(X[:, t] - xref, self.Q)

            if t < (self.T - 1):
                constraints += [X[:, t + 1] == self.A @ X[:, t] + self.B @ U[:, t]] #TODO: Add disturbance
                if max(self.R.shape) == 1:
                    cost += self.R*cp.square(U[:, t + 1] - U[:, t]) #To prevent chattering
                else: 
                    cost += cp.quad_form(U[:, t + 1] - U[:, t], self.R)

        constraints += [X[:, 0] == x0]
        constraints += [X[0, :] >= self.SAFETY_DIST]
        constraints += [cp.abs(U[0, :]) <= self.THROTTLE_LIMIT]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        # assert prob.is_dpp()

        result = prob.solve(verbose=True)

        return U.value, X.value

# Ts = 1

# A = np.array([[1, Ts],
#               [0,  1]])

# B = np.array([[-1/2*Ts*Ts, 0],
#               [-Ts, 0]])
# B = np.array([[-1/2*Ts*Ts],
#               [-Ts]])

# T = 100
# t = np.arange(T)

# Q = np.diag([10, 10])
# R = np.array([5])
# # R = np.diag([5, 5])

# modelMPC = MPCLinear(A, B, Q, R, T)

# x0 = [145, 0]
# xref = [5, 0]

# u, y = modelMPC.action(x0, xref)

# # Plot the results
# plt.subplot(3, 1, 1)
# plt.plot(t, y[0])
# plt.plot([0,T], [5,5])
# plt.xlabel("t [sec]")
# plt.ylabel("s [m]")

# plt.subplot(3, 1, 2)
# plt.plot(t, y[1])
# plt.xlabel("t [sec]")
# plt.ylabel("v [m/s]")

# plt.subplot(3, 1, 3)
# plt.plot(t, u[0])
# plt.xlabel("t [sec]")
# plt.ylabel("u")

# plt.show()