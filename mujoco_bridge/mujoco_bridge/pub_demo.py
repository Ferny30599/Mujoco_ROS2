import os
import csv
import math
import numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

#:::::::::::Ganancias::::::::::::::
lambda1 = 5.0
lambda2 = 4.0
rho1    = 5.0
rho2    = 4.0
kp1     = 3.0
kp2     = 3.0
kl1     = 3.0
kl2     = 3.0

class CsvLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f'joints_{ts}.csv')
        self._f = open(self.path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow([
            't',
            'r_hip_d','r_knee_d','l_hip_d','l_knee_d',
            'r_hip','r_knee','l_hip','l_knee',
            'r_dhip','r_dknee','l_dhip','l_dknee',
            'u_r_hip','u_r_knee','u_l_hip','u_l_knee',
            'u_norm_r_hip','u_norm_r_knee','u_norm_l_hip','u_norm_l_knee'
        ])
        self._f.flush()

    def log(self, t, thd_R, thd_L, th_R, th_L, dth_R, dth_L, u_R, u_L, u_R_norm, u_L_norm,):
        self._w.writerow([
            f"{t:.6f}",
            f"{thd_R[0]:.9f}", f"{thd_R[1]:.9f}", f"{thd_L[0]:.9f}", f"{thd_L[1]:.9f}",
            f"{th_R[0]:.9f}",  f"{th_R[1]:.9f}",  f"{th_L[0]:.9f}",  f"{th_L[1]:.9f}",
            f"{dth_R[0]:.9f}", f"{dth_R[1]:.9f}", f"{dth_L[0]:.9f}", f"{dth_L[1]:.9f}",
            f"{float(u_R[0]):.9f}", f"{float(u_R[1]):.9f}",
            f"{float(u_L[0]):.9f}", f"{float(u_L[1]):.9f}",
            f"{float(u_R_norm[0]):.9f}", f"{float(u_R_norm[1]):.9f}",
            f"{float(u_L_norm[0]):.9f}", f"{float(u_L_norm[1]):.9f}",

        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

class DoublePendulumModel:
    """ Parámetros y utilidades del modelo. """
    def __init__(self):
        self.M1 = 7.0
        self.M2 = 3.5
        self.L1 = 0.425
        self.L2 = 0.408
        self.I1 = (1.0/3.0) * self.M1 * (self.L1**2)
        self.I2 = (1.0/3.0) * self.M2 * (self.L2**2)
        self.G  = 9.81

    def B_matrix(self, th1, th2):
        A = (self.I1 + 0.25*self.M1*self.L1**2 + self.M2*self.L1**2) \
          * (self.I2 + 0.25*self.M2*self.L2**2) \
          - 0.25*(self.M2**2)*(self.L1**2)*(self.L2**2)*math.cos(th1 - th2)**2

        B = (1.0/max(A, 1e-9)) * np.array([
            [ (self.I2 + 0.25*self.M2*self.L2**2),
              0.5*self.M2*self.L1*self.L2*math.cos(th1 - th2) ],
            [ 0.5*self.M2*self.L1*self.L2*math.cos(th1 - th2),
              (self.I1 + 0.25*self.M1*self.L1**2 + self.M2*self.L1**2) ]
        ])
        return B
    
    def M(self, th1, th2):
        a11 = self.I1 + 0.25*self.M1*self.L1**2 + self.M2*self.L1**2
        a22 = self.I2 + 0.25*self.M2*self.L2**2
        a12 = -0.5*self.M2*self.L1*self.L2 * math.cos(th1 - th2)
        return np.array([[a11, a12],
                         [a12, a22]])

    def Pdot(self, th1, th2, p1, p2, tau1, tau2):
        A = (self.I1 + 0.25*self.M1*self.L1**2 + self.M2*self.L1**2) \
          * (self.I2 + 0.25*self.M2*self.L2**2) \
          - 0.25*(self.M2**2)*(self.L1**2)*(self.L2**2)*math.cos(th1 - th2)**2
        
        aux = (1.0/max(2*A, 1e-9))*self.M2*self.L1*self.L2*math.sin(th1 - th2)*p1*p2 
        dp1 = aux - (0.5*self.M1+self.M2)*self.G*self.L1*math.sin(th1) + tau1
        dp2 = aux - 0.5*self.M2*self.G*self.L2*math.sin(th2) + tau2

def trayectoria(t: float) -> np.ndarray:
    th1d = 12.6256 + -8.6350*math.cos(6.2832*t) + -17.3014*math.sin(6.2832*t) \
           + -4.4379*math.cos(12.5664*t) + 7.5505*math.sin(12.5664*t) \
           + 4.0846*math.cos(18.8496*t) + 3.0829*math.sin(18.8496*t) \
           + 1.7269*math.cos(25.1327*t) + -2.2278*math.sin(25.1327*t) \
           + -0.3477*math.cos(31.4159*t) + -3.2555*math.sin(31.4159*t) \
           + -0.1148*math.cos(37.6991*t) + -1.0870*math.sin(37.6991*t) \
           + -1.8513*math.cos(43.9823*t) + -0.9656*math.sin(43.9823*t) \
           + -1.3099*math.cos(50.2655*t) + -0.3047*math.sin(50.2655*t) \
           + -0.5083*math.cos(56.5487*t) + 0.7967*math.sin(56.5487*t) \
           + -0.2451*math.cos(62.8319*t) + -0.0949*math.sin(62.8319*t)

    th2d = 14.5622 + -16.6088*math.cos(6.2832*t) + 1.9419*math.sin(6.2832*t) \
           + -0.0858*math.cos(12.5664*t) + 4.9112*math.sin(12.5664*t) \
           + -0.4556*math.cos(18.8496*t) + -0.5858*math.sin(18.8496*t) \
           + -0.8188*math.cos(25.1327*t) + -0.1501*math.sin(25.1327*t) \
           + 0.6327*math.cos(31.4159*t) + -0.3034*math.sin(31.4159*t) \
           + 0.0312*math.cos(37.6991*t) + -0.0536*math.sin(37.6991*t) \
           + 0.5173*math.cos(43.9823*t) + 0.3177*math.sin(43.9823*t) \
           + 0.5269*math.cos(50.2655*t) + 0.1391*math.sin(50.2655*t) \
           + 0.2488*math.cos(56.5487*t) + 0.4999*math.sin(56.5487*t) \
           + 0.2452*math.cos(62.8319*t) + 0.3931*math.sin(62.8319*t)

    return np.array([math.radians(th1d), math.radians(th2d)])

def trayectoria_dot(t: float) -> np.ndarray:
    th1d_dot = (8.6350*6.2832*math.sin(6.2832*t)) + (-17.3014*6.2832*math.cos(6.2832*t)) \
             + (4.4379*12.5664*math.sin(12.5664*t)) + (7.5505*12.5664*math.cos(12.5664*t)) \
             + (-4.0846*18.8496*math.sin(18.8496*t)) + (3.0829*18.8496*math.cos(18.8496*t)) \
             + (-1.7269*25.1327*math.sin(25.1327*t)) + (-2.2278*25.1327*math.cos(25.1327*t)) \
             + (0.3477*31.4159*math.sin(31.4159*t)) + (-3.2555*31.4159*math.cos(31.4159*t)) \
             + (0.1148*37.6991*math.sin(37.6991*t)) + (-1.0870*37.6991*math.cos(37.6991*t)) \
             + (1.8513*43.9823*math.sin(43.9823*t)) + (-0.9656*43.9823*math.cos(43.9823*t)) \
             + (1.3099*50.2655*math.sin(50.2655*t)) + (-0.3047*50.2655*math.cos(50.2655*t)) \
             + (0.5083*56.5487*math.sin(56.5487*t)) + (0.7967*56.5487*math.cos(56.5487*t)) \
             + (0.2451*62.8319*math.sin(62.8319*t)) + (-0.0949*62.8319*math.cos(62.8319*t))

    th2d_dot = (16.6088*6.2832*math.sin(6.2832*t)) + (1.9419*6.2832*math.cos(6.2832*t)) \
             + (0.0858*12.5664*math.sin(12.5664*t)) + (4.9112*12.5664*math.cos(12.5664*t)) \
             + (0.4556*18.8496*math.sin(18.8496*t)) + (-0.5858*18.8496*math.cos(18.8496*t)) \
             + (0.8188*25.1327*math.sin(25.1327*t)) + (-0.1501*25.1327*math.cos(25.1327*t)) \
             + (-0.6327*31.4159*math.sin(31.4159*t)) + (-0.3034*31.4159*math.cos(31.4159*t)) \
             + (-0.0312*37.6991*math.sin(37.6991*t)) + (-0.0536*37.6991*math.cos(37.6991*t)) \
             + (-0.5173*43.9823*math.sin(43.9823*t)) + (0.3177*43.9823*math.cos(43.9823*t)) \
             + (-0.5269*50.2655*math.sin(50.2655*t)) + (0.1391*50.2655*math.cos(50.2655*t)) \
             + (-0.2488*56.5487*math.sin(56.5487*t)) + (0.4999*56.5487*math.cos(56.5487*t)) \
             + (-0.2452*62.8319*math.sin(62.8319*t)) + (0.3931*62.8319*math.cos(62.8319*t))

    return np.array([math.radians(th1d_dot), math.radians(th2d_dot)])

def QSMC(th, thd, dthd, p, lam, rho, model: DoublePendulumModel):
    B = model.B_matrix(th[0], th[1])
    dth = B @ p
    e   = th - thd
    ep  = dth - dthd
    s   = ep + lam @ e
    dth_des = dthd - lam @ e - rho @ np.tanh(s)
    pd = np.linalg.solve(B, dth_des)
    return pd

def SMC(p, dp, pd, dpd, kp, kl):
    s  = p - pd
    ueq = dpd - dp
    return ueq - kp @ np.sign(s) - kl @ s

def saturation(u: float, umin:float, umax:float):
    return float(np.clip(u, umin, umax))

class Demo(Node):
    def __init__(self):
        super().__init__('mujoco_pub_demo')

        # --- Publicador ---
        self.pub = self.create_publisher(Float64MultiArray, 'mujoco/ctrl', 10)

        # --- Suscriptor ---
        self.sub = self.create_subscription(
            JointState,
            'joint_states',
            self.on_js,
            10
        )

        # --- Variables  ---
        self.names_right = ("Cadera_Derecha", "Rodilla_Derecha")
        self.names_left  = ("Cadera_Izquierda", "Rodilla_Izquierda")
        self.selected_names = [self.names_left[0], self.names_left[1], 
                               self.names_right[0], self.names_right[1]]
        self.th = {name: 0 for name in self.selected_names}
        self.dth = {name: 0 for name in self.selected_names}

        # Estados internos
        self.model = DoublePendulumModel()
        self.logger = CsvLogger(os.path.expanduser('~/logs_sim_mujoco'))

        # --- Timer ---
        self.t = 0.0
        self.fase=0.0
        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz

        self.lam = np.diag([lambda1, lambda2])
        self.rho = np.diag([rho1,rho2])
        self.kp  = np.diag([kp1,kp2])
        self.kl  = np.diag([kl1,kl2])

        self.pdes_prev_I = np.zeros(2)
        self.pdes_prev_D = np.zeros(2)
        self.p_prev_I = np.zeros(2)
        self.p_prev_D = np.zeros(2)

    def leg_state(self, names):
        th   = np.array([ self.th[names[0]],   self.th[names[1]]   ])
        dth  = np.array([ self.dth[names[0]],  self.dth[names[1]]  ])

        M   = self.model.M(th[0], th[1])
        p   = M @ dth
        return th, dth, p

    # ::::::: Publicador :::::::
    def tick(self):
        self.dt = 0.5#Tiempo de muestreo
        self.t += self.dt #Tiempo de simulacion
        #::::::::::::::CONTROL:::::::::::

        # Posiciones deseadas
        th_des_I = [1,1]#trayectoria(self.t*0.00025)
        dth_des_I = [0,0]#trayectoria_dot(self.t*0.00025)
        th_des_D = [0,0]#trayectoria(0.00025*self.t+self.fase)
        dth_des_D = [0,0]#trayectoria_dot(0.00025*self.t+self.fase)

        # Posiciones medidas y momentos conjugados
        th_D, dth_D, p_D,= self.leg_state(self.names_right)
        th_I, dth_I, p_I = self.leg_state(self.names_left)

        # QSMC
        pdes_D = QSMC(th_D, dth_D, th_des_D, dth_des_D, self.lam, self.rho, self.model)
        pdes_I = QSMC(th_I, dth_I, th_des_I, dth_des_I, self.lam, self.rho, self.model)

        # Derivadas de p y pdes
        dp_I  = (p_I  - self.p_prev_I)  / self.dt
        dp_D  = (p_D  - self.p_prev_D)  / self.dt
        dpdes_I = (pdes_I - self.pdes_prev_I) / self.dt
        dpdes_D = (pdes_D - self.pdes_prev_D) / self.dt

        self.pdes_prev_I = pdes_I.copy()
        self.pdes_prev_D = pdes_D.copy()

        #SMC
        u_D = SMC(p_D, dp_D, pdes_D, dpdes_D, self.kp, self.kl)
        u_I = SMC(p_I, dp_I, pdes_I, dpdes_I, self.kp, self.kl)

        u_D1_sat=saturation(-u_D[0],-150,50) #Cadera derecha
        u_D2_sat=saturation(-u_D[1],-100,100) #Rodilla derecha

        u_I1_sat=saturation(-u_I[0],-150,50) #Cadera izquierda
        u_I2_sat=saturation(-u_I[1],-100,100) #Rodilla izquierda

        self.logger.log(self.t,th_des_D,th_des_I,th_D,th_I,dth_D,dth_I,u_D,u_I,[u_D1_sat,u_D2_sat],[u_I1_sat,u_I2_sat])

        #::::::::::::::Publica los torques::::::::::::::::::: 
        msg = Float64MultiArray()
        #MusloDer,MusloIzq,PiernaDer,PiernaIzq
        msg.data = [u_D1_sat,u_I1_sat,u_D2_sat,u_I2_sat]
        self.pub.publish(msg)
        #self.information()

    # ::::::: Callback del joint_states :::::::
    def on_js(self, msg: JointState):
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.th:
                self.th[name] = pos
            if name in self.dth:
                self.dth[name] = vel

    def information(self):
        if all(v is not None for v in self.th.values()):
            self.get_logger().info(
                f'th: {self.th}, dth: {self.dth}'
            )
        else:
            self.get_logger().info('Esperando datos de todas las articulaciones...')

    # ---- Limpieza ----
    def destroy_node(self):
        try:
            self.logger.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    n = Demo()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()