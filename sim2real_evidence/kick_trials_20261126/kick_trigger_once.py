# /kick_trigger 를 한 번 보내고 1 s 동안 살아 있어 전달을 보장한다 (ros2 topic pub --once 는 바로 종료해 킥 러너가 놓친 적이 있다).
import time, rclpy
from std_msgs.msg import Bool
import sys
NEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rclpy.init(); n = rclpy.create_node("kick_trigger_once"); p = n.create_publisher(Bool, "/kick_trigger", 10)
t = time.time()
while p.get_subscription_count() < NEED and time.time() - t < 8.0:
    rclpy.spin_once(n, timeout_sec=0.05)
print("subscribers", p.get_subscription_count(), flush=True)
p.publish(Bool(data=True))
t = time.time()
while time.time() - t < 2.0:
    rclpy.spin_once(n, timeout_sec=0.05)
n.destroy_node(); rclpy.shutdown()
