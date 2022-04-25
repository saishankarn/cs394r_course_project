
class OneDTimeOptimalControl():
    def __init__(self, max_vel, max_acc, max_dec, del_t):
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_dec = max_dec
        self.del_t   = del_t   
        self.safety_margin = 1

    def action(self, v0, distance_remaining):
        if distance_remaining <= 0.0:
            v1 = 0.0
            return -1
        else :
            ### Acceleration
            v1 = v0 + self.max_acc*self.del_t
            del_s1 = v1*self.del_t + 0.5*v1*v1/self.max_dec
            del_s2 = 0.5*v0*v0/self.max_dec
            if v1 <= self.max_vel and del_s1 < distance_remaining - self.safety_margin:
                return 1
            
            ### Cruise
            elif v0 <= self.max_vel and del_s2 < distance_remaining  - self.safety_margin:
                v1 = v0
                return 0.0
            
            ### Deceleration
            else :
                v1 = v0 - self.max_dec*self.del_t
                v1 = max(v1, 0)
                return -1



