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
#  Modelo (doble péndulo)
# ==============================
M1 = 7.0
M2 = 3.5
L1 = 0.425
L2 = 0.408
I1 = (1.0/3.0)*M1*(L1**2)
I2 = (1.0/3.0)*M2*(L2**2)
G  = 9.81

def B_matrix(th1, th2):
    A = (I1 + 0.25*M1*L1**2 + M2*L1**2)*(I2 + 0.25*M2*L2**2) \
        - 0.25*(M2**2)*(L1**2)*(L2**2)*math.cos(th1 + th2)**2
    B = (1.0/A) * np.array([
        [ (I2 + 0.25*M2*L2**2),            0.5*M2*L1*L2*math.cos(th1 + th2) ],
        [ 0.5*M2*L1*L2*math.cos(th1 + th2), (I1 + 0.25*M1*L1**2 + M2*L1**2) ]
    ])
    return B

# ==============================
#  Trayectoria deseada (serie Fourier)
# ==============================
def trayectoria(t):
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
#  QSMC / SMC
# ==============================
def QSMC(th, thd, dthd, p, lambda1, lambda2, ro1, ro2, phi):
    B = B_matrix(th[0], th[1])
    dth = B @ p
    Lambda = np.diag([lambda1, lambda2])
    Ro     = np.diag([ro1, ro2])
    e  = th - thd
    ep = dth - dthd
    s  = ep + Lambda @ e
    sat = np.array([ (si/phi if abs(si) <= phi else np.sign(si)) for si in s ])
    dthdes = dthd - Lambda @ e - Ro @ sat
    pd = np.linalg.solve(B, dthdes)
    return pd

def SMC(p, dp, pd, dpd, kp1, kp2, kl1, kl2):
    Kp = np.diag([kp1, kp2])
    Kl = np.diag([kl1, kl2])
    s  = p - pd
    ueq = dpd - dp
    return ueq - Kp @ np.sign(s) - Kl @ s

# ==============================
#  Acondicionador par -> control de tendón
# ==============================
def smooth_norm(u, umax=1.0, alpha=2.0):
    x = math.tanh(alpha * (u / max(1e-6, umax)))  # (-1,1)
    return 0.5 * (x + 1.0)                         # (0,1)

def rate_limit(prev, target, max_step):
    delta = target - prev
    if delta > max_step:
        return prev + max_step
    if delta < -max_step:
        return prev - max_step
    return target

class TendonMapper:
    def __init__(self,
                 hip_range=(-1.0, 0.35),
                 knee_range=(0.0, 1.0),
                 umax_hip=1.0, umax_knee=1.0,
                 alpha_hip=2.0, alpha_knee=2.0,
                 max_step=0.03):
        self.hip_min,  self.hip_max  = hip_range
        self.knee_min, self.knee_max = knee_range
        self.umax_hip   = umax_hip
        self.umax_knee  = umax_knee
        self.alpha_hip  = alpha_hip
        self.alpha_knee = alpha_knee
        self.max_step   = max_step
        self._hip_R = 0.0;  self._knee_R = 1.0
        self._hip_L = 0.0;  self._knee_L = 1.0

    @staticmethod
    def lerp(a, b, x):
        return a + (b - a) * x

    def set_state(self, hip_R, hip_L, knee_R, knee_L):
        self._hip_R  = float(hip_R)
        self._hip_L  = float(hip_L)
        self._knee_R = float(knee_R)
        self._knee_L = float(knee_L)

    def hip_ctrl_from_tau(self, tau, side):
        lvl = smooth_norm(tau, self.umax_hip, self.alpha_hip)
        raw = self.lerp(self.hip_max, self.hip_min, lvl)  # 0->0.35, 1->-1
        if side == 'R':
            out = rate_limit(self._hip_R, raw, self.max_step); self._hip_R = out
        else:
            out = rate_limit(self._hip_L, raw, self.max_step); self._hip_L = out
        return float(np.clip(out, self.hip_min, self.hip_max))

    def knee_ctrl_from_tau(self, tau, side):
        lvl = smooth_norm(tau, self.umax_knee, self.alpha_knee)
        raw = self.lerp(self.knee_max, self.knee_min, lvl)  # 0->1, 1->0
        if side == 'R':
            out = rate_limit(self._knee_R, raw, self.max_step); self._knee_R = out
        else:
            out = rate_limit(self._knee_L, raw, self.max_step); self._knee_L = out
        return float(np.clip(out, self.knee_min, self.knee_max))

# ==============================
#  Nodo
# ==============================
class Demo(Node):
    def __init__(self):
        super().__init__('mujoco_controller_demo')

        # ---------- Parámetros ----------
        self.declare_parameter('speed', 0.025)
        self.declare_parameter('period', 1.0)
        self.declare_parameter('log_dir', os.path.expanduser('~/ros2_ws/logs/joint_logs'))
        self.declare_parameter('log_decimate', 1)
        self.declare_parameter('print_every', 0.5)

        # warm-up (cables extendidos)
        self.declare_parameter('warmup_time', 5.0)  # segundos
        self.declare_parameter('hip_init',   0.35)  # cadera extendida
        self.declare_parameter('knee_init',  1.00)  # rodilla extendida

        # mapeador
        self.mapper = TendonMapper(
            hip_range=(-1.0, 0.35),
            knee_range=(0.0, 1.0),
            umax_hip=1.0, umax_knee=1.0,
            alpha_hip=2.0, alpha_knee=2.0,
            max_step=0.03
        )

        self.speed       = float(self.get_parameter('speed').value)
        self.T           = float(self.get_parameter('period').value)
        self.phase       = self.T / 4.0
        self.log_dir     = self.get_parameter('log_dir').value
        self.log_decim   = int(self.get_parameter('log_decimate').value)
        self.print_every = float(self.get_parameter('print_every').value)

        self.warmup_time = float(self.get_parameter('warmup_time').value)
        self.hip_init    = float(self.get_parameter('hip_init').value)
        self.knee_init   = float(self.get_parameter('knee_init').value)

        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f'joints_{ts}.csv')
        self.csvfile = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow([
            't',
            'r_hip_d','r_knee_d','l_hip_d','l_knee_d',
            'r_hip','r_knee','l_hip','l_knee',
            'r_dhip','r_dknee','l_dhip','l_dknee',
            'u_r_hip','u_r_knee','u_l_hip','u_l_knee',
            'cmd_hip_R','cmd_knee_R','cmd_hip_L','cmd_knee_L',
            'mode'
        ])
        self.csvfile.flush()
        self._last_print = time.time()
        self._tick = 0

        # ---------- ROS IO ----------
        self.pub = self.create_publisher(Float64MultiArray, 'mujoco/ctrl', 10)
        self.sub = self.create_subscription(JointState, 'joint_states', self.on_js, 10)

        # nombres de juntas
        self.names_right = ("Cadera_Derecha", "Rodilla_Derecha")
        self.names_left  = ("Cadera_Izquierda", "Rodilla_Izquierda")

        # Control (QSMC/SMC)
        self.lambda1 = 28.0; self.lambda2 = 12.0
        self.ro1 = 28.0;     self.ro2 = 12.0
        self.phi = math.radians(15.0)
        self.kp1 = 5.0; self.kp2 = 5.0
        self.kl1 = 5.0; self.kl2 = 5.0

        # Estado
        self.dt = 0.01
        self.idx_map_built = False
        self.idx = {}
        self.q = {}; self.dq = {}
        self.p_prev_R = np.zeros(2); self.pd_prev_R = np.zeros(2)
        self.p_prev_L = np.zeros(2); self.pd_prev_L = np.zeros(2)

        # Relojes
        self.sim_t = 0.0      # tiempo de simulación (siempre avanza)
        self.ctrl_t = 0.0     # tiempo del controlador (solo en modo 'closed')

        # Modo y marcas
        self._mode = 'warmup'
        self._t0 = None  # se fija al primer loop con joint_states

        # Lazo
        self.timer = self.create_timer(self.dt, self.loop)
        self.get_logger().info(f"Controller listo. CSV: {self.csv_path}")

    # ---------- Helpers ----------
    def _build_index_map(self, msg: JointState):
        wanted = set(self.names_right + self.names_left)
        for i, nm in enumerate(msg.name):
            if nm in wanted:
                self.idx[nm] = i
        ok = all(n in self.idx for n in wanted)
        if ok:
            self.get_logger().info(f"Mapeadas juntas: {self.idx}")
        return ok

    def on_js(self, msg: JointState):
        if not self.idx_map_built:
            self.idx_map_built = self._build_index_map(msg)
            if not self.idx_map_built:
                return

        for nm in (self.names_right + self.names_left):
            i = self.idx[nm]
            self.q[nm]  = msg.position[i]
            self.dq[nm] = msg.velocity[i] if len(msg.velocity) > i else 0.0

    def _leg_state(self, names):
        th  = np.array([ self.q[names[0]],  self.q[names[1]] ])
        dth = np.array([ self.dq[names[0]], self.dq[names[1]] ])
        B   = B_matrix(th[0], th[1])
        p   = np.linalg.solve(B, dth)
        return th, dth, p

    # ---------- Lazo principal ----------
    def loop(self):
        if not self.idx_map_built or any(n not in self.q for n in (self.names_right + self.names_left)):
            return

        # avanza reloj de simulación siempre
        self.sim_t += self.dt
        if self._t0 is None:
            self._t0 = self.sim_t
        elapsed = self.sim_t - self._t0

        # --- MODO WARMUP: NO se evalúa el control ni avanza ctrl_t ---
        if self._mode == 'warmup' and elapsed < self.warmup_time:
            cmd = np.array([self.hip_init, self.hip_init, self.knee_init, self.knee_init], dtype=float)
            self.pub.publish(Float64MultiArray(data=cmd.tolist()))

            # log mínimo durante warmup (sin control)
            self._log_row(mode='warmup', u_R=np.zeros(2), u_L=np.zeros(2),
                          cmd=cmd, thd_R=np.zeros(2), thd_L=np.zeros(2),
                          th_R=np.array([self.q[self.names_right[0]], self.q[self.names_right[1]]]),
                          th_L=np.array([self.q[self.names_left[0]],  self.q[self.names_left[1]]]),
                          dth_R=np.array([self.dq[self.names_right[0]], self.dq[self.names_right[1]]]),
                          dth_L=np.array([self.dq[self.names_left[0]],  self.dq[self.names_left[1]]]))
            # prints
            if time.time() - getattr(self, "_last_print", 0.0) >= self.print_every:
                self._last_print = time.time()
                self.get_logger().info(f"(warmup) t={self.sim_t:6.2f} — cables extendidos")
            return

        # Si estábamos en warmup y acaba de terminar, sincroniza y arranca control en t=0
        if self._mode == 'warmup':
            self._mode = 'closed'
            self.ctrl_t = 0.0
            # al salir del warmup, fija el estado interno del mapper a los valores publicados
            self.mapper.set_state(hip_R=self.hip_init, hip_L=self.hip_init,
                                  knee_R=self.knee_init, knee_L=self.knee_init)
            # limpia derivadas previas para evitar picos
            self._thd_prev_R = None
            self._thd_prev_L = None
            self.p_prev_R[:] = 0.0; self.pd_prev_R[:] = 0.0
            self.p_prev_L[:] = 0.0; self.pd_prev_L[:] = 0.0
            self.get_logger().info("Finalizó warm-up: iniciando control cerrado.")

        # --- CONTROL CERRADO ---
        self.ctrl_t += self.dt

        # Estados y referencias (ojo: usan ctrl_t, NO sim_t)
        th_R, dth_R, p_R = self._leg_state(self.names_right)
        thd_R = trayectoria(self.ctrl_t * self.speed)
        thd_prev_R = getattr(self, "_thd_prev_R", None)
        pd_R, u_R, dthd_R = self._qs_smc(th_R, dth_R, p_R, thd_R, thd_prev_R,
                                         self.p_prev_R, self.pd_prev_R)
        self._thd_prev_R = thd_R
        self.p_prev_R, self.pd_prev_R = p_R.copy(), pd_R.copy()

        th_L, dth_L, p_L = self._leg_state(self.names_left)
        thd_L = trayectoria(self.ctrl_t * self.speed + self.phase)
        thd_prev_L = getattr(self, "_thd_prev_L", None)
        pd_L, u_L, dthd_L = self._qs_smc(th_L, dth_L, p_L, thd_L, thd_prev_L,
                                         self.p_prev_L, self.pd_prev_L)
        self._thd_prev_L = thd_L
        self.p_prev_L, self.pd_prev_L = p_L.copy(), pd_L.copy()

        # Par -> control de tendón
        cmd = np.zeros(4, dtype=float)
        cmd[0] = self.mapper.hip_ctrl_from_tau(  u_R[0], side='R')   # MusloDer
        cmd[2] = self.mapper.knee_ctrl_from_tau( u_R[1], side='R')   # PiernaDer
        cmd[1] = self.mapper.hip_ctrl_from_tau(  u_L[0], side='L')   # MusloIzq
        cmd[3] = self.mapper.knee_ctrl_from_tau( u_L[1], side='L')   # PiernaIzq

        self.pub.publish(Float64MultiArray(data=cmd.tolist()))
        self._log_row('closed', u_R, u_L, cmd, thd_R, thd_L, th_R, th_L, dth_R, dth_L)

        # consola
        if time.time() - getattr(self, "_last_print", 0.0) >= self.print_every:
            self._last_print = time.time()
            self.get_logger().info(
                f"(closed) t={self.ctrl_t:6.2f} | "
                f"RH={th_R[0]:+.3f}/{thd_R[0]:+.3f} RK={th_R[1]:+.3f}/{thd_R[1]:+.3f} | "
                f"LH={th_L[0]:+.3f}/{thd_L[0]:+.3f} LK={th_L[1]:+.3f}/{thd_L[1]:+.3f}"
            )

    # ---------- utilidades ----------
    def _qs_smc(self, th, dth, p, thd, thd_prev, p_prev, pd_prev):
        dthd = (thd - thd_prev) / self.dt if thd_prev is not None else np.zeros(2)
        pd   = QSMC(th, thd, dthd, p, self.lambda1, self.lambda2, self.ro1, self.ro2, self.phi)
        dp   = (p  - p_prev ) / self.dt
        dpd  = (pd - pd_prev) / self.dt
        u    = SMC(p, dp, pd, dpd, self.kp1, self.kp2, self.kl1, self.kl2)
        return pd, u, dthd

    def _log_row(self, mode, u_R, u_L, cmd, thd_R, thd_L, th_R, th_L, dth_R, dth_L):
        self._tick += 1
        if self._tick % max(self.log_decim, 1) != 0:
            return
        self.writer.writerow([
            f"{self.sim_t:.6f}",
            # deseados
            f"{thd_R[0]:.9f}", f"{thd_R[1]:.9f}", f"{thd_L[0]:.9f}", f"{thd_L[1]:.9f}",
            # medidos
            f"{th_R[0]:.9f}",  f"{th_R[1]:.9f}",  f"{th_L[0]:.9f}",  f"{th_L[1]:.9f}",
            f"{dth_R[0]:.9f}", f"{dth_R[1]:.9f}", f"{dth_L[0]:.9f}", f"{dth_L[1]:.9f}",
            # ley de control (si warmup, quedarán 0)
            f"{float(u_R[0]):.9f}", f"{float(u_R[1]):.9f}",
            f"{float(u_L[0]):.9f}", f"{float(u_L[1]):.9f}",
            # comandos enviados
            f"{float(cmd[0]):.9f}", f"{float(cmd[2]):.9f}",
            f"{float(cmd[1]):.9f}", f"{float(cmd[3]):.9f}",
            mode
        ])
        self.csvfile.flush()

    def on_js(self, msg: JointState):
        if not self.idx_map_built:
            self.idx_map_built = self._build_index_map(msg)
            if not self.idx_map_built:
                return
        for nm in (self.names_right + self.names_left):
            i = self.idx[nm]
            self.q[nm]  = msg.position[i]
            self.dq[nm] = msg.velocity[i] if len(msg.velocity) > i else 0.0

    def destroy_node(self):
        try:
            self.csvfile.close()
        except Exception:
            pass
        super().destroy_node()

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
