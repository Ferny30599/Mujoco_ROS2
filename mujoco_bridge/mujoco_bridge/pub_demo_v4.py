#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# =======================================
# 1) MODELO, M(q), Mdot(q,dq) y trayectorias
# =======================================
class DoublePendulumModel:
    """Parámetros del doble péndulo y utilidades M, Mdot."""
    def __init__(self):
        self.M1 = 7.0
        self.M2 = 3.5
        self.L1 = 0.425
        self.L2 = 0.408
        self.I1 = (1.0/3.0) * self.M1 * (self.L1**2)
        self.I2 = (1.0/3.0) * self.M2 * (self.L2**2)

    # Matriz de inercia M(q)
    def M(self, th1, th2):
        a11 = self.I1 + 0.25*self.M1*self.L1**2 + self.M2*self.L1**2
        a22 = self.I2 + 0.25*self.M2*self.L2**2
        a12 = -0.5*self.M2*self.L1*self.L2 * math.cos(th1 - th2)
        return np.array([[a11, a12],
                         [a12, a22]])

    # Derivada temporal Mdot(q,dq) (solo depende de a12)
    def Mdot(self, th1, th2, dth1, dth2):
        a12_dot = 0.5*self.M2*self.L1*self.L2 * math.sin(th1 - th2) * (dth1 - dth2)
        return np.array([[0.0,     a12_dot],
                         [a12_dot, 0.0    ]])

def trayectoria(t: float) -> np.ndarray:
    """Ángulos deseados (rad). Serie en grados -> se convierte a rad al final."""
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
    """Velocidades deseadas (rad/s)."""
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

# =======================================
# 2) QSMC + SMC (sin B; con M directamente)
# =======================================
def QSMC(th, dth, thd, dthd, lam, rho, model: DoublePendulumModel):
    """
    Usa velocidades medidas dth (no B @ p). Devuelve p_d = M * dth_des.
    s = (dth - dthd) + Λ (th - thd)
    dth_des = dthd - Λ e - ρ tanh(s)
    """
    e  = th - thd
    ep = dth - dthd
    s  = ep + lam @ e
    dth_des = dthd - lam @ e - rho @ np.tanh(s)
    Md = model.M(th[0], th[1])
    pd = Md @ dth_des
    return pd

def SMC(p, dp, pd, dpd, kp, kl):
    s  = p - pd
    ueq = dpd - dp
    return ueq - kp @ np.sign(s) - kl @ s

# =======================================
# 3) Mapeo torque -> actuadores (con saturación lineal)
# =======================================
def saturation(u, umin, umax):
    return float(np.clip(u, umin, umax))

class TendonMapper:
    """Mapea par virtual -> comando de actuador (lineal)."""
    def __init__(self, hip_range=(-150.0, 50.0), knee_range=(-100.0, 100.0), umax_hip=50.0, umax_knee=100.0,umin_hip=-150.0, umin_knee=-100.0):
        self.hip_min,  self.hip_max  = hip_range
        self.knee_min, self.knee_max = knee_range
        self.umax_hip = umax_hip
        self.umax_knee = umax_knee
        self.umin_hip = umin_hip
        self.umin_knee = umin_knee

    def hip_from_tau(self, tau):
        tau_s = saturation(tau, self.umin_hip, self.umax_hip)
        return [tau_s, -tau_s]

    def knee_from_tau(self, tau):
        tau_s = saturation(tau, self.umin_knee, self.umax_knee)
        return [tau_s, -tau_s]

# =======================================
# 4) Logger CSV
# =======================================
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
            'u_norm_r_hip','u_norm_r_knee','u_norm_l_hip','u_norm_l_knee',
            'cmd_hip_R','cmd_knee_R','cmd_hip_L','cmd_knee_L'
        ])
        self._f.flush()

    def log(self, t, thd_R, thd_L, th_R, th_L, dth_R, dth_L, u_R, u_L, u_R_norm, u_L_norm, cmd):
        self._w.writerow([
            f"{t:.6f}",
            f"{thd_R[0]:.9f}", f"{thd_R[1]:.9f}", f"{thd_L[0]:.9f}", f"{thd_L[1]:.9f}",
            f"{th_R[0]:.9f}",  f"{th_R[1]:.9f}",  f"{th_L[0]:.9f}",  f"{th_L[1]:.9f}",
            f"{dth_R[0]:.9f}", f"{dth_R[1]:.9f}", f"{dth_L[0]:.9f}", f"{dth_L[1]:.9f}",
            f"{float(u_R[0]):.9f}", f"{float(u_R[1]):.9f}",
            f"{float(u_L[0]):.9f}", f"{float(u_L[1]):.9f}",
            f"{float(u_R_norm[0]):.9f}", f"{float(u_R_norm[1]):.9f}",
            f"{float(u_L_norm[0]):.9f}", f"{float(u_L_norm[1]):.9f}",
            f"{float(cmd[0]):.9f}", f"{float(cmd[2]):.9f}",
            f"{float(cmd[1]):.9f}", f"{float(cmd[3]):.9f}"
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# =======================================
# 5) Nodo ROS2
# =======================================
class Demo(Node):
    def __init__(self):
        super().__init__('mujoco_controller_demo')

        # Parámetros de usuario mínimos
        self.declare_parameter('dt', 0.02)                  # s
        self.declare_parameter('speed', 0.025)             # factor temporal de trayectoria
        self.declare_parameter('period', 1.0)              # para desfase pierna izq
        self.declare_parameter('warmup_time', 5.0)         # s de pre-tensión
        self.declare_parameter('hip_init', 0.35)           # cadera extendida
        self.declare_parameter('knee_init', 1.00)          # rodilla extendida
        self.declare_parameter('log_dir', os.path.expanduser('~/ros2_ws/logs/joint_logs'))

        # Ganancias
        self.declare_parameter('lambda1', 0.80)
        self.declare_parameter('lambda2', 0.65)
        self.declare_parameter('ro1', 10.0)
        self.declare_parameter('ro2', 10.0)
        self.declare_parameter('kp1', 10.0)
        self.declare_parameter('kp2', 10.0)
        self.declare_parameter('kl1', 10.0)
        self.declare_parameter('kl2', 10.0)

        # Mapper
        self.declare_parameter('umax_hip', 150.0)
        self.declare_parameter('umax_knee', 100.0)

        # Leer parámetros
        self.dt          = float(self.get_parameter('dt').value)
        self.speed       = float(self.get_parameter('speed').value)
        self.T           = float(self.get_parameter('period').value)
        self.phase       = self.T / 4.0
        self.warmup_time = float(self.get_parameter('warmup_time').value)
        self.hip_init    = float(self.get_parameter('hip_init').value)
        self.knee_init   = float(self.get_parameter('knee_init').value)
        self.log_dir     = self.get_parameter('log_dir').value

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
            hip_range=(-100.0, 50.0),
            knee_range=(-100.0, 100.0),
            umax_hip=float(self.get_parameter('umax_hip').value),
            umax_knee=float(self.get_parameter('umax_knee').value)
        )

        # IO ROS
        self.pub = self.create_publisher(Float64MultiArray, 'mujoco/ctrl', 10)
        self.sub = self.create_subscription(JointState, 'joint_states', self.on_js, 10)

        # Nombres de juntas
        self.names_right = ("Cadera_Derecha", "Rodilla_Derecha")
        self.names_left  = ("Cadera_Izquierda", "Rodilla_Izquierda")

        # Estados internos
        self.model = DoublePendulumModel()
        self.logger = CsvLogger(self.log_dir)
        self.idx_map_built = False
        self.idx = {}
        self.q  = {}
        self.dq = {}
        self.ddq = {}   # aceleraciones medidas (desde JointState.effort)

        # Memorias para dpd por diferencia finita
        self.pd_prev_R = np.zeros(2)
        self.pd_prev_L = np.zeros(2)

        # Reloj y modo
        self.sim_t  = 0.0
        self._mode  = 'warmup'
        self._warmup_elapsed = 0.0

        # Timer principal
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info(f"Listo. CSV: {self.logger.path} | dt={self.dt:.6f}s | warmup={self.warmup_time:.2f}s")

    # ---- Callback de JointState ----
    def on_js(self, msg: JointState):
        if not self.idx_map_built:
            wanted = set(self.names_right + self.names_left)
            for i, nm in enumerate(msg.name):
                if nm in wanted:
                    self.idx[nm] = i
            self.idx_map_built = all(n in self.idx for n in wanted)
            if not self.idx_map_built:
                return

        for nm in (self.names_right + self.names_left):
            i = self.idx[nm]
            self.q[nm]   = msg.position[i]
            self.dq[nm]  = msg.velocity[i] if len(msg.velocity) > i else 0.0
            # ddq desde JointState.effort si existe; si no, 0.0
            self.ddq[nm] = msg.effort[i] if len(msg.effort) > i else 0.0

    # ---- Helper: estado de una pierna ----
    def leg_state(self, names):
        th   = np.array([ self.q[names[0]],   self.q[names[1]]   ])
        dth  = np.array([ self.dq[names[0]],  self.dq[names[1]]  ])
        ddth = np.array([ self.ddq[names[0]], self.ddq[names[1]] ])
        # p = M dth;  p_dot = Mdot dth + M ddth
        M   = self.model.M(th[0], th[1])
        Mdt = self.model.Mdot(th[0], th[1], dth[0], dth[1])
        p   = M @ dth
        dp  = Mdt @ dth + M @ ddth
        return th, dth, ddth, p, dp

    # ---- Lazo principal ----
    def loop(self):
        if not self.idx_map_built:
            return

        # Warm-up (cables extendidos)
        if self._mode == 'warmup':
            if self._warmup_elapsed < self.warmup_time:
                cmd = np.array([self.hip_init, self.hip_init, self.knee_init, self.knee_init], dtype=float)
                self.pub.publish(Float64MultiArray(data=cmd.tolist()))
                self._warmup_elapsed += self.dt
                return
            # Cambio a control cerrado
            self._mode = 'closed'
            self.sim_t = 0.0
            self.get_logger().info("Iniciando control...")
            return

        # ===== Control cerrado =====
        self.sim_t += self.dt

        # Estados medidos (incluye dp analítico)
        th_R, dth_R, ddth_R, p_R, dp_R = self.leg_state(self.names_right)
        th_L, dth_L, ddth_L, p_L, dp_L = self.leg_state(self.names_left)

        # Referencias
        thd_R  = trayectoria(self.sim_t * self.speed)
        dthd_R = trayectoria_dot(self.sim_t * self.speed)

        thd_L  = trayectoria(self.sim_t * self.speed + self.phase)
        dthd_L = trayectoria_dot(self.sim_t * self.speed + self.phase)

        # QSMC -> pd
        pd_R = QSMC(th_R, dth_R, thd_R, dthd_R, self.lam, self.rho, self.model)
        pd_L = QSMC(th_L, dth_L, thd_L, dthd_L, self.lam, self.rho, self.model)

        # dpd por diferencia finita (analítico completo sería más complejo)
        dpd_R = (pd_R - self.pd_prev_R) / self.dt
        dpd_L = (pd_L - self.pd_prev_L) / self.dt
        self.pd_prev_R = pd_R.copy()
        self.pd_prev_L = pd_L.copy()

        # SMC
        u_R = SMC(p_R, dp_R, pd_R, dpd_R, self.kp, self.kl)
        u_L = SMC(p_L, dp_L, pd_L, dpd_L, self.kp, self.kl)

        # Mapeo a actuadores
        cmd = np.zeros(4, dtype=float)
        u_R_norm = np.zeros(2, dtype=float)
        u_L_norm = np.zeros(2, dtype=float)
        u_R_norm[0], cmd[0] = self.mapper.hip_from_tau(  u_R[0])
        u_R_norm[1], cmd[2] = self.mapper.knee_from_tau( u_R[1])
        u_L_norm[0], cmd[1] = self.mapper.hip_from_tau(  u_L[0])
        u_L_norm[1], cmd[3] = self.mapper.knee_from_tau( u_L[1])

        self.pub.publish(Float64MultiArray(data=cmd.tolist()))

        # Log
        self.logger.log(self.sim_t, thd_R, thd_L, th_R, th_L, dth_R, dth_L, u_R, u_L, u_R_norm, u_L_norm, cmd)

    # ---- Limpieza ----
    def destroy_node(self):
        try:
            self.logger.close()
        except Exception:
            pass
        super().destroy_node()

# =======================================
# 6) main
# =======================================
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
