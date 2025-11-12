import os
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time as RosTime
import mujoco

try:
    import mujoco.viewer as viewer_mod
except Exception:
    viewer_mod = None


class MuJoCoBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')

        # Parámetros
        self.declare_parameter('model_path', os.path.expanduser('~/dummy.xml'))
        # 🔹 SIN SlideX: solo los 4 actuadores de tendón
        self.declare_parameter('actuator_names', [
            'MusloDerechoCable',
            'MusloIzquierdoCable',
            'PiernaDerechoCable',
            'PiernaIzquierdoCable',
        ])
        self.declare_parameter('realtime_rate', 1.0)   # 1.0 = tiempo real
        self.declare_parameter('use_viewer', True)     # abrir ventana GLFW

        # <<< NUEVOS PARÁMETROS >>>
        self.declare_parameter('publish_every_n_steps', 1)   # publica joint_states/clock cada N pasos
        self.declare_parameter('substeps', 1)                # pasos de física por iteración del viewer loop

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.actuator_names = [
            s for s in self.get_parameter('actuator_names').get_parameter_value().string_array_value
        ]
        self.rate = float(self.get_parameter('realtime_rate').value)
        self.use_viewer = bool(self.get_parameter('use_viewer').value)

        self.pub_every = max(1, int(self.get_parameter('publish_every_n_steps').value))  # <<<
        self.substeps = max(1, int(self.get_parameter('substeps').value))                # <<<
        self._step_count = 0  # contador de pasos para decimación                   # <<<

        # Cargar modelo MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # Mapeo actuadores
        self.act_ids = []
        for name in self.actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f'Actuador no encontrado: {name}')
            self.act_ids.append(aid)
        self.ctrl = np.zeros(len(self.act_ids), dtype=np.float64)

        # (Opcional) Log de rangos para depurar
        ranges_txt = []
        for name, aid in zip(self.actuator_names, self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            ranges_txt.append(f"{name}[{lo:.3f},{hi:.3f}]")
        self.get_logger().info("Rangos ctrl: " + ", ".join(ranges_txt))

        # ROS pubs/subs
        self.cmd_sub = self.create_subscription(Float64MultiArray, 'mujoco/ctrl', self.cmd_cb, 10)
        self.js_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.clock_pub = self.create_publisher(RosTime, '/clock', 10)

        # Si NO hay viewer, usamos un timer para step
        if not self.use_viewer:
            self.timer = self.create_timer(self.dt / max(self.rate, 1e-6), self.step)

        self.get_logger().info(f'Cargado {model_path}')
        self.get_logger().info('Actuators: ' + ', '.join(self.actuator_names))
        self.get_logger().info('Escucha comandos en: /mujoco/ctrl (Float64MultiArray)')
        self.get_logger().info(f'publish_every_n_steps={self.pub_every} | substeps={self.substeps}')  # <<<

    def cmd_cb(self, msg: Float64MultiArray):
        # Copia segura: si llega más largo que act_ids, se trunca; si más corto, mantiene lo anterior
        n = min(len(msg.data), len(self.ctrl))
        if n > 0:
            self.ctrl[:n] = np.asarray(msg.data[:n], dtype=np.float64)

    def _publish_state(self):
        """Publica joint_states y /clock usando tiempo de simulación."""  # <<<
        # Publicar joint_states (posiciones y velocidades)
        js = JointState()

        # <<< Usar tiempo de simulación MUJOCO (data.time) para el header.stamp >>>
        sec = int(self.data.time)
        nsec = int((self.data.time - sec) * 1e9)
        js.header.stamp = RosTime(sec=sec, nanosec=nsec)

        names = []
        positions = []
        velocities = []

        for j in range(self.model.njnt):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            jtype = self.model.jnt_type[j]
            qposadr = self.model.jnt_qposadr[j]
            dofadr = self.model.jnt_dofadr[j]

            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                names.extend([f"{nm}.x", f"{nm}.y", f"{nm}.z",
                              f"{nm}.qw", f"{nm}.qx", f"{nm}.qy", f"{nm}.qz"])
                qpos_slice = self.data.qpos[qposadr:qposadr+7]
                positions.extend(qpos_slice.tolist())

                v_slice = self.data.qvel[dofadr:dofadr+6]
                velocities.extend([float(v_slice[0]), float(v_slice[1]), float(v_slice[2]),
                                   0.0,
                                   float(v_slice[3]), float(v_slice[4]), float(v_slice[5])])
            else:
                names.append(nm)
                positions.append(float(self.data.qpos[qposadr]))
                velocities.append(float(self.data.qvel[dofadr]))

        js.name = names
        js.position = positions
        js.velocity = velocities
        self.js_pub.publish(js)

        # Publicar /clock con el MISMO tiempo de simulación
        clk = RosTime(sec=sec, nanosec=nsec)
        self.clock_pub.publish(clk)

    def step(self):
        # Aplicar ctrl respetando ctrlrange
        for i, aid in enumerate(self.act_ids):
            val = self.ctrl[i]
            lo, hi = self.model.actuator_ctrlrange[aid]
            if np.isfinite(lo) and np.isfinite(hi):
                val = float(np.clip(val, lo, hi))
            self.data.ctrl[aid] = val

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1  # <<<

        # Publica solo cada N pasos (decimación)
        if (self._step_count % self.pub_every) == 0:          # <<<
            self._publish_state()                              # <<<

    def _viewer_loop(self):
        """Bucle cuando hay viewer: hacemos 'substeps' pasos por iteración de render."""
        render_hz = 60.0
        render_dt = 1.0 / render_hz
        next_render = time.time()

        with viewer_mod.launch_passive(self.model, self.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.lookat[:] = [1.5, -0.5, 1.1]
            v.cam.azimuth = 87.0
            v.cam.elevation = -1.0
            v.cam.distance = 6

            while v.is_running() and rclpy.ok():
                for _ in range(self.substeps):  # <<< permite avanzar física más veces por frame
                    self.step()

                now = time.time()
                if now >= next_render:
                    v.sync()
                    next_render = now + render_dt


def spin_in_thread(node: Node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = MuJoCoBridge()

    use_viewer = node.use_viewer
    if use_viewer and viewer_mod is None:
        node.get_logger().warn("use_viewer=true pero no se pudo importar mujoco.viewer; ejecutando headless")
        use_viewer = False

    spin_thread = None
    try:
        if use_viewer:
            spin_thread = threading.Thread(target=spin_in_thread, args=(node,), daemon=True)
            spin_thread.start()
            node._viewer_loop()  # <<< usar loop con substeps
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        node.destroy_node()


if __name__ == '__main__':
    main()
