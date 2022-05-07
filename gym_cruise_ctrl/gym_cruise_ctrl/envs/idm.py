from math import sqrt

"""
### Equation from paper:
    Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning
"""

class IDM():
    def __init__(self) -> None:
        self.d0    = 5      # Minimum gap distance
        self.T     = 0.0    # Safe time headway
        self.a     = 0.7    # Maximal acceleration
        self.b     = 1.5    # Desired deceleration
        self.delta = 4      # Acceleration exponent
        self.v0    = 20     # Desired velocity, also the max velocity of the ego vehicle

    def action(self, d, delta_v, v):
        d_star = self.d0 + v*self.T + v*delta_v/(2*sqrt(self.a*self.b))
        acc    = self.a*(1 - (v/self.v0)**self.delta - 2*(d_star/d)**2)
        return acc