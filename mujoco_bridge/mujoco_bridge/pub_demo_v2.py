#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import time
from datetime import datetime
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ==============================
# 1) MODELO Y TRAYECTORIA DESEADA
# ==============================
class DoublePendulumModel:
    """ Parámetros y utilidades del modelo geométrico/simplificado. """
    def __init__(self):
        # (Barra homogénea alrededor de un extremo) — ajusta si cambian masas/longitudes
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

# ==============================
# 2) CONTROLADORES (QSMC + SMC)
# ==============================
def QSMC(th, thd, dthd, p, lam, rho, model: DoublePendulumModel):
    B = model.B_matrix(th[0], th[1])
    dth = B @ p
    e   = th - thd
    ep  = dth - dthd
    s   = ep + lam @ e
    sat = np.tanh(s)
    dth_des = dthd - lam @ e - rho @ sat
    pd = np.linalg.solve(B, dth_des)
    return pd

def SMC(p, dp, pd, dpd, kp, kl):
    s  = p - pd
    ueq = dpd - dp
    return ueq - kp @ np.sign(s) - kl @ s

# ==============================
# 3) ACONDICIONADOR DEL TORQUE A LOS CABLES
# ==============================
def saturation(u, umax):
    """ Normaliza par """
    if u>umax:
        x=umax
    elif u>=-umax and u<=umax:
        x=u
    else:
        x = -umax
    return x     

class TendonMapper:
    """
    Convierte par (u) -> control de actuador respetando rangos del XML.
    RANGOS:
      - Caderas:  [-1.0, 0.35]  (0.35=extendido, -1.0=flexionado)
      - Rodillas: [ 0.0, 1.0 ]  (1.0=extendido, 0.0=flexionado)
    """
    def __init__(self,
                 hip_range=(-1.0, 0.35),
                 knee_range=(0.0, 1.0),
                 umax_hip=1.0, umax_knee=1.0):
        self.hip_min,  self.hip_max  = hip_range
        self.knee_min, self.knee_max = knee_range
        self.umax_hip   = umax_hip
        self.umax_knee  = umax_knee

    def hip_from_tau(self, tau):
        # lvl=0 -> 0.35 (extendido), lvl=1 -> -1.0 (flexionado)
        tau_s = saturation(tau, self.umax_hip)     # 0..1
        raw = -0.675*tau_s-0.325      # invierte extremo
        return [tau_s,float(np.clip(raw, self.hip_min, self.hip_max))]

    def knee_from_tau(self, tau):
        # lvl=0 -> 1.0 (extendido), lvl=1 -> 0.0 (flexionado)
        tau_s = saturation(tau, self.umax_knee)   # 0..1
        raw = -0.5*tau_s+0.5        # invierte extremo
        return [tau_s,float(np.clip(raw, self.knee_min, self.knee_max))]

# ==============================
# 4) LOGGING A CSV
# ==============================
class CsvLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f'joints_{ts}.csv')
        self._f = open(self.path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow([
            't',                                  # tiempo de simulación (s)
            'r_hip_d','r_knee_d','l_hip_d','l_knee_d',
            'r_hip','r_knee','l_hip','l_knee',
            'r_dhip','r_dknee','l_dhip','l_dknee',
            'u_r_hip','u_r_knee','u_l_hip','u_l_knee',      # ley de control (SMC)
            'u_norm_r_hip','u_norm_r_knee','u_norm_l_hip','u_norm_l_knee',      # ley de control (SMC)
            'cmd_hip_R','cmd_knee_R','cmd_hip_L','cmd_knee_L'
        ])
        self._f.flush()

    def log(self, t, thd_R, thd_L, th_R, th_L, dth_R, dth_L, u_R, u_L, u_R_norm, u_L_norm, cmd):
        self._w.writerow([
            f"{t:.6f}",
            # deseados
            f"{thd_R[0]:.9f}", f"{thd_R[1]:.9f}", f"{thd_L[0]:.9f}", f"{thd_L[1]:.9f}",
            # medidos
            f"{th_R[0]:.9f}",  f"{th_R[1]:.9f}",  f"{th_L[0]:.9f}",  f"{th_L[1]:.9f}",
            f"{dth_R[0]:.9f}", f"{dth_R[1]:.9f}", f"{dth_L[0]:.9f}", f"{dth_L[1]:.9f}",
            # control law
            f"{float(u_R[0]):.9f}", f"{float(u_R[1]):.9f}",
            f"{float(u_L[0]):.9f}", f"{float(u_L[1]):.9f}",
            # control law_normalized
            f"{float(u_R_norm[0]):.9f}", f"{float(u_R_norm[1]):.9f}",
            f"{float(u_L_norm[0]):.9f}", f"{float(u_L_norm[1]):.9f}",
            # comandos
            f"{float(cmd[0]):.9f}", f"{float(cmd[2]):.9f}",
            f"{float(cmd[1]):.9f}", f"{float(cmd[3]):.9f}"
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# ==============================
# 5) NODO ROS2
# ==============================
class Demo(Node):
    def __init__(self):
        super().__init__('mujoco_controller_demo')

        # ---------- Parámetros de usuario ----------
        self.declare_parameter('dt', 0.001)                 # Tiempo de muestreo (s)
        self.declare_parameter('speed', 0.025)              # Factor temporal de trayectoria
        self.declare_parameter('period', 1.0)               # Periodo base (para desfase)
        self.declare_parameter('warmup_time', 5.0)          # s de pre-tensión
        self.declare_parameter('hip_init', 0.35)            # cadera extendida
        self.declare_parameter('knee_init', 1.00)           # rodilla extendida
        self.declare_parameter('log_dir', os.path.expanduser('~/ros2_ws/logs/joint_logs'))
        self.declare_parameter('log_decimate', 1)
        self.declare_parameter('print_every', 0.5)

        # Ganancias de control (ajustables desde YAML si quieres)
        self.declare_parameter('lambda1', 0.80)
        self.declare_parameter('lambda2', 0.65)
        self.declare_parameter('ro1', 1.0)
        self.declare_parameter('ro2', 1.0)
        self.declare_parameter('kp1', 1.0)
        self.declare_parameter('kp2', 1.0)
        self.declare_parameter('kl1', 1.0)
        self.declare_parameter('kl2', 1.0)

        # Mapper (exponemos para tuning)
        self.declare_parameter('umax_hip', 1.0)
        self.declare_parameter('umax_knee', 1.0)


        # ---------- Lee parámetros ----------
        self.dt          = float(self.get_parameter('dt').value)
        self.speed       = float(self.get_parameter('speed').value)
        self.T           = float(self.get_parameter('period').value)
        self.phase       = self.T / 4.0
        self.warmup_time = float(self.get_parameter('warmup_time').value)
        self.hip_init    = float(self.get_parameter('hip_init').value)
        self.knee_init   = float(self.get_parameter('knee_init').value)
        self.log_dir     = self.get_parameter('log_dir').value
        self.log_decim   = int(self.get_parameter('log_decimate').value)
        self.print_every = float(self.get_parameter('print_every').value)

        lam = np.diag([float(self.get_parameter('lambda1').value),
                       float(self.get_parameter('lambda2').value)])
        rho = np.diag([float(self.get_parameter('ro1').value),
                       float(self.get_parameter('ro2').value)])
        kp  = np.diag([float(self.get_parameter('kp1').value),
                       float(self.get_parameter('kp2').value)])
        kl  = np.diag([float(self.get_parameter('kl1').value),
                       float(self.get_parameter('kl2').value)])

        self.lam, self.rho = lam, rho
        self.kp, self.kl   = kp, kl

        self.mapper = TendonMapper(
            hip_range=(-1.0, 0.35), knee_range=(0.0, 1.0),
            umax_hip=float(self.get_parameter('umax_hip').value),
            umax_knee=float(self.get_parameter('umax_knee').value)
        )

        # ---------- IO ROS ----------
        self.pub = self.create_publisher(Float64MultiArray, 'mujoco/ctrl', 10)
        self.sub = self.create_subscription(JointState, 'joint_states', self.on_js, 10)

        # Nombres de juntas (del XML)
        self.names_right = ("Cadera_Derecha", "Rodilla_Derecha")
        self.names_left  = ("Cadera_Izquierda", "Rodilla_Izquierda")

        # ---------- Estados internos ----------
        self.model = DoublePendulumModel()
        self.logger = CsvLogger(self.log_dir)
        self.idx_map_built = False
        self.idx = {}
        self.q = {}; self.dq = {}

        self.p_prev_R = np.zeros(2); self.pd_prev_R = np.zeros(2)
        self.p_prev_L = np.zeros(2); self.pd_prev_L = np.zeros(2)

        # Relojes
        self.sim_t  = 0.0      # avanza siempre
        self._t0    = None

        # Modo
        self._mode = 'warmup'

        # Timer principal respetando dt del usuario
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info(f"Listo. CSV: {self.logger.path} | dt={self.dt:.6f}s | warmup={self.warmup_time:.2f}s")

    # ---------- Callbacks ----------
    def on_js(self, msg: JointState):
        # Construye mapa de índices la primera vez
        if not self.idx_map_built:
            wanted = set(self.names_right + self.names_left)
            for i, nm in enumerate(msg.name):
                if nm in wanted:
                    self.idx[nm] = i
            self.idx_map_built = all(n in self.idx for n in wanted)
            if self.idx_map_built:
                self.get_logger().info(f"Mapeadas juntas: {self.idx}")
            else:
                return

        # Copia posiciones/velocidades (0 si no viene)
        for nm in (self.names_right + self.names_left):
            i = self.idx[nm]
            self.q[nm]  = msg.position[i]
            self.dq[nm] = msg.velocity[i] if len(msg.velocity) > i else 0.0

    # ---------- Helpers de estado ----------
    def leg_state(self, names):
        th  = np.array([ self.q[names[0]],  self.q[names[1]] ])
        dth = np.array([ self.dq[names[0]], self.dq[names[1]] ])
        B   = self.model.B_matrix(th[0], th[1])
        p   = np.linalg.solve(B, dth)
        return th, dth, p

    # ---------- Ciclo principal ----------
    def loop(self):
        if not self.idx_map_built:
            return

        # Relojes
        self.sim_t += self.dt
        if self._t0 is None:
            self._t0 = self.sim_t
        elapsed = self.sim_t - self._t0

        # ==========  MODO WARMUP: cables extendidos  ==========
        if self._mode == 'warmup':
            if elapsed < self.warmup_time:
                # Publica los comandos de extensión para inicializar los cables
                cmd = np.array([self.hip_init, self.hip_init, self.knee_init, self.knee_init], dtype=float)
                self.pub.publish(Float64MultiArray(data=cmd.tolist()))
                return 

            self._mode = 'closed'
            # Reinicia relojes para que las gráficas arranquen en t=0 al iniciar control
            self.sim_t = 0.0

            # Limpia estados/derivadas para evitar picos
            self._thd_prev_R = 0.0; self._thd_prev_L = 0.0
            self.p_prev_R[:] = 0.0;  self.p_prev_L[:]  = 0.0
            self.pd_prev_R[:] = 0.0; self.pd_prev_L[:] = 0.0

            self.get_logger().info("Iniciando control...")
            return

        # ==========  CONTROL CERRADO  ==========
        th_R, dth_R, p_R = self.leg_state(self.names_right)
        thd_R = trayectoria(self.sim_t * self.speed)
        dthd_R = (thd_R - getattr(self, "_thd_prev_R", thd_R)) / self.dt
        self._thd_prev_R = thd_R

        th_L, dth_L, p_L = self.leg_state(self.names_left)
        thd_L = trayectoria(self.sim_t * self.speed + self.phase)
        dthd_L = (thd_L - getattr(self, "_thd_prev_L", thd_L)) / self.dt
        self._thd_prev_L = thd_L

        # QSMC: genera pd (referencia de p)
        pd_R = QSMC(th_R, thd_R, dthd_R, p_R, self.lam, self.rho, self.model)
        pd_L = QSMC(th_L, thd_L, dthd_L, p_L, self.lam, self.rho, self.model)

        # Derivadas numéricas en p
        dp_R  = (p_R  - self.p_prev_R)  / self.dt
        dpd_R = (pd_R - self.pd_prev_R) / self.dt
        dp_L  = (p_L  - self.p_prev_L)  / self.dt
        dpd_L = (pd_L - self.pd_prev_L) / self.dt

        # SMC: ley de control (par virtual)
        u_R = SMC(p_R, dp_R, pd_R, dpd_R, self.kp, self.kl)
        u_L = SMC(p_L, dp_L, pd_L, dpd_L, self.kp, self.kl)

        # Guarda estados anteriores
        self.p_prev_R, self.pd_prev_R = p_R.copy(), pd_R.copy()
        self.p_prev_L, self.pd_prev_L = p_L.copy(), pd_L.copy()

        # Par -> control de tendón (respeta rangos del XML)
        cmd = np.zeros(4, dtype=float)  # [MusloDer, MusloIzq, PiernaDer, PiernaIzq]
        u_R_norm = np.zeros(2, dtype=float)  # [MusloDer, MusloIzq, PiernaDer, PiernaIzq]
        u_L_norm = np.zeros(2, dtype=float)  # [MusloDer, MusloIzq, PiernaDer, PiernaIzq]
        u_R_norm[0],cmd[0] = self.mapper.hip_from_tau(  u_R[0])
        u_R_norm[1],cmd[2] = self.mapper.knee_from_tau( u_R[1])
        u_L_norm[0],cmd[1] = self.mapper.hip_from_tau(  u_L[0])
        u_L_norm[1],cmd[3] = self.mapper.knee_from_tau( u_L[1])

        # Publica a /mujoco/ctrl (orden: MusloDer, MusloIzq, PiernaDer, PiernaIzq)
        self.pub.publish(Float64MultiArray(data=cmd.tolist()))

        # Log
        self.logger.log(self.sim_t, thd_R, thd_L, th_R, th_L, dth_R, dth_L, u_R, u_L, u_R_norm,u_L_norm, cmd)

        '''# Consola (tracking rápido)
        # if time.time() - getattr(self, "_last_print", 0.0) >= self.print_every:
        #     self._last_print = time.time()
        #     self.get_logger().info(
        #         f"(closed) t={self.sim_t:6.3f} | "
        #         f"RH={th_R[0]:+.3f}/{thd_R[0]:+.3f} RK={th_R[1]:+.3f}/{thd_R[1]:+.3f} | "
        #         f"LH={th_L[0]:+.3f}/{thd_L[0]:+.3f} LK={th_L[1]:+.3f}/{thd_L[1]:+.3f}"
        #     )
        '''

    # ---------- Limpieza ----------
    def destroy_node(self):
        try:
            self.logger.close()
        except Exception:
            pass
        super().destroy_node()

# ==============================
# 6) MAIN
# ==============================
def main():
    rclpy.init()
    node = Demo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
