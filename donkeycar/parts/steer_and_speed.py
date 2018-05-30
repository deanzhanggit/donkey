
import pigpio 
import numpy as np


class PWMSteering:
    """
    Wrapper over a PWM motor cotnroller to convert angles to PWM pulses.
    """
    LEFT_ANGLE = -1 
    RIGHT_ANGLE = 1
    MID_DUTY = 23
    STEER_PIN_NUM = 13

    def __init__(self,left_max=26,right_max=19,pid_controller=None):

        self.pid_controller = pid_controller
        self.left_max = left_max
        self.right_max = right_max
                   
        self.p=pigpio.pi()   
        self.p.set_mode(self.STEER_PIN_NUM,pigpio.OUTPUT)
        self.p.set_PWM_frequency(self.STEER_PIN_NUM,50) 

    def run(self, angle):
        if angle < 0 :
            output_duty = self.MID_DUTY + (self.left_max - self.MID_DUTY)* angle * self.LEFT_ANGLE
        else:
            output_duty = self.MID_DUTY + (self.right_max - self.MID_DUTY)* angle * self.RIGHT_ANGLE
        
        self.p.set_PWM_dutycycle(self.STEER_PIN_NUM, np.round(output_duty))

    def shutdown(self):
        self.run(self.MID_DUTY) #set steering straight


class PWMMotor:
    """
    Wrapper over a PWM motor cotnroller to convert angles to PWM pulses.
    """
    MOTOR_PIN_NUM = 19
    MOTOR_STOP = 78
    MAX_SPEED = 90
    def __init__(self,pid_controller=None):

        self.pid_controller = pid_controller

        self.p = pigpio.pi()   
        self.p.set_mode(self.MOTOR_PIN_NUM,pigpio.OUTPUT)
        self.p.set_PWM_frequency(self.MOTOR_PIN_NUM,1000) 
        self.p.set_PWM_dutycycle(self.MOTOR_PIN_NUM,self.MOTOR_STOP)

    def run(self, speed):
        if speed < 0 :
            self.p.set_PWM_dutycycle(self.MOTOR_PIN_NUM, self.MOTOR_STOP)
        else:
        
            output_duty = self.MOTOR_STOP + (self.MAX_SPEED - self.MOTOR_STOP)* speed
            self.p.set_PWM_dutycycle(self.MOTOR_PIN_NUM, int(output_duty))

    def shutdown(self):
        self.run(self.MOTOR_STOP) #set steering straight