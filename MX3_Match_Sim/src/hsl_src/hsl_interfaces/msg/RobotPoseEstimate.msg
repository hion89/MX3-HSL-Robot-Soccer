# B-Human RobotPose 대응. 필드 좌표(map): 원점 센터, +x = 상대 골대 방향, 단위 m / rad
std_msgs/Header header
float32 x
float32 y
float32 theta
float32[9] covariance          # 3x3 (x, y, theta) row-major
float32 validity               # 최선 가설의 유효도 0~1 (관측을 얼마나 잘 설명하는가)
uint8 quality                  # 0=POOR(다봉) 1=OKAY 2=SUPERB
uint8 QUALITY_POOR=0
uint8 QUALITY_OKAY=1
uint8 QUALITY_SUPERB=2
uint8 num_hypotheses
uint32 num_landmarks_used      # 이번 주기 최선 가설에 반영된 랜드마크 수
builtin_interfaces/Time last_jump    # 위치가 크게 튄(리셋) 마지막 시각
