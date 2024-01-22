import torch

class PIDController:
    def __init__(self, Ki):
        self.Ki = Ki
        self.reset()

    def reset(self):
        self.integral = 0

    def step(self, pos, pos_target):
        self.integral = self.integral + (pos_target-pos)
        return pos_target + self.Ki * self.integral 