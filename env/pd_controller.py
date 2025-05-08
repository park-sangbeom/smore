import numpy as np

class PD_Controller():
    def __init__(self, dt=0.1, p=0.01*60, d=0.001*60, sample_time=0.01):
        self.start_point = np.zeros(shape=2) 
        self.dt           = dt 
        self.kp           = p 
        self.kd           = d 
        self.prev_torque  = 0 
        self.current_time = 0.0
        self.prev_time    = self.current_time
        self.prev_error   = np.zeros(shape=2)
        self.sample_time  = sample_time
        self.PTerm        = 0.0 
        self.DTerm        = 0.0 
        self.output       = 0.0 

    def clear(self):
        self.start_point  = np.zeros(shape=2)
        self.prev_error   = np.zeros(shape=2)
        self.Pterm        = 0.0 
        self.Dterm        = 0.0 
        self.output       = np.zeros(shape=2)
        self.current_time = 0.0 
        self.prev_time    = 0.0 

    def control(self, diff_value, current_time):
        error             = self.start_point-diff_value
        self.current_time = current_time
        delta_time        = self.current_time-self.prev_time + 1e-6
        delta_err         = error-self.prev_error
        if (delta_time>=self.sample_time):
            self.PTerm = self.kp*error 
            self.DTerm = 0.0 
            if delta_time >0: 
                self.DTerm = delta_err/delta_time
            # Remember last time and last error for next calculation
            self.prev_time = self.current_time
            self.prev_error = error
            # Clipping with min, max stall torque
            self.output = self.PTerm + (self.kd * self.DTerm)



