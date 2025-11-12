import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class Demo(Node):
    def __init__(self):
        super().__init__('mujoco_pub_demo')
        self.pub = self.create_publisher(Float64MultiArray, 'mujoco/ctrl', 10)
        self.t = 0.0
        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz

    def tick(self):
        self.t += 0.02
        s = math.sin(2.0 * math.pi * 0.5 * self.t)      # 0.5 Hz
        msg = Float64MultiArray()
        # [SlideX, MusloDer, MusloIzq, PiernaDer, PiernaIzq]
        msg.data = [-0.8, -0.8, s, -s]
        self.pub.publish(msg)

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
