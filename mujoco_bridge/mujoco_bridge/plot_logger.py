import os, time, csv
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import matplotlib.pyplot as plt

class PlotLogger(Node):
    """
    Suscribe a:
      traj/right/hip_d, traj/right/knee_d,
      meas/right/hip,  meas/right/knee
    y grafica + guarda CSV en tiempo real.
    """

    def __init__(self):
        super().__init__('plot_logger')

        # ---- Parámetros opcionales ----
        self.declare_parameter('hz', 20.0)      # frecuencia de logging/plot
        self.declare_parameter('save_dir', os.path.expanduser('~/ros2_ws/logs/plot_logs'))
        self.hz = float(self.get_parameter('hz').value)
        self.dt = 1.0 / max(self.hz, 1e-3)
        self.save_dir = self.get_parameter('save_dir').value

        # ---- Estado y último valor recibido de cada señal ----
        self.latest = {
            'hip_d': None, 'knee_d': None,
            'hip_m': None, 'knee_m': None,
        }
        self.t0 = time.time()
        self.t_series = []
        self.hip_d_series, self.hip_m_series = [], []
        self.knee_d_series, self.knee_m_series = [], []

        # ---- Subscripciones ----
        self.create_subscription(Float64, 'traj/right/hip_d',  lambda m: self._set('hip_d',  m.data), 10)
        self.create_subscription(Float64, 'traj/right/knee_d', lambda m: self._set('knee_d', m.data), 10)
        self.create_subscription(Float64, 'meas/right/hip',    lambda m: self._set('hip_m',  m.data), 10)
        self.create_subscription(Float64, 'meas/right/knee',   lambda m: self._set('knee_m', m.data), 10)

        # ---- CSV ----
        os.makedirs(self.save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.save_dir, f'right_leg_{ts}.csv')
        self.csvfile = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['t', 'hip_d', 'hip_m', 'knee_d', 'knee_m'])
        self.csvfile.flush()
        self.get_logger().info(f'Guardando CSV en: {self.csv_path}')

        # ---- Plot en vivo ----
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        self.l_hip_d,  = self.ax1.plot([], [], label='Hip deseado')
        self.l_hip_m,  = self.ax1.plot([], [], '--', label='Hip medido')
        self.l_knee_d, = self.ax2.plot([], [], label='Knee deseado')
        self.l_knee_m, = self.ax2.plot([], [], '--', label='Knee medido')
        self.ax1.set_ylabel('Ángulo [rad]'); self.ax1.legend(loc='upper right')
        self.ax2.set_ylabel('Ángulo [rad]'); self.ax2.set_xlabel('Tiempo [s]'); self.ax2.legend(loc='upper right')
        plt.show(block=False)   # <-- la ventana queda visible ya

        # ---- Timer de muestreo/plot ----
        self.timer = self.create_timer(self.dt, self._tick)

    # ===== Helpers =====
    def _now(self): 
        return time.time() - self.t0

    def _set(self, key, value):
        self.latest[key] = float(value)

    def _have_all(self):
        return all(self.latest[k] is not None for k in self.latest.keys())

    def _tick(self):
        # Sólo loguea cuando ya llegó al menos un valor de cada señal
        if not self._have_all():
            return

        t = self._now()
        hip_d  = self.latest['hip_d']
        hip_m  = self.latest['hip_m']
        knee_d = self.latest['knee_d']
        knee_m = self.latest['knee_m']

        # ---- Guardar en CSV (una fila por tick) ----
        self.writer.writerow([f'{t:.6f}', f'{hip_d:.9f}', f'{hip_m:.9f}', f'{knee_d:.9f}', f'{knee_m:.9f}'])
        self.csvfile.flush()  # <-- asegura que se escriba al disco en tiempo real

        # ---- Acumular para plot ----
        self.t_series.append(t)
        self.hip_d_series.append(hip_d)
        self.hip_m_series.append(hip_m)
        self.knee_d_series.append(knee_d)
        self.knee_m_series.append(knee_m)

        # ---- Actualizar líneas ----
        self.l_hip_d.set_data(self.t_series, self.hip_d_series)
        self.l_hip_m.set_data(self.t_series, self.hip_m_series)
        self.l_knee_d.set_data(self.t_series, self.knee_d_series)
        self.l_knee_m.set_data(self.t_series, self.knee_m_series)

        # Autoscale
        for ax in (self.ax1, self.ax2):
            ax.relim(); ax.autoscale_view()

        # Redibujar
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def destroy_node(self):
        try:
            self.csvfile.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    n = PlotLogger()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()
        # deja la última figura visible al salir
        try:
            import matplotlib.pyplot as plt
            plt.ioff(); plt.show()
        except Exception:
            pass

if __name__ == '__main__':
    main()
