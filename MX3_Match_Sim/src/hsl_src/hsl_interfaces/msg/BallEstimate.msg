# B-Human BallModel 대응. 자기 관측 기반 공 추정
std_msgs/Header header
bool valid                     # 최근(ball_timeout 안)에 본 적 있음
float32 rel_x                  # 몸통(base_link) 기준 위치 [m]
float32 rel_y
float32 field_x                # 필드 좌표 위치 [m] (로봇 pose 로 변환)
float32 field_y
float32 vel_x                  # 필드 좌표 속도 [m/s]
float32 vel_y
float32[4] covariance          # 위치 2x2
float32 time_since_seen        # [s]
float32 time_since_disappeared # 보여야 하는데 안 보인 시간 [s] (B-Human BallLost 대응 단순화)
bool cam_valid                 # 아래 카메라 좌표가 유효 (목 제어용 — 행동 노드가 TF 를 안 듣게)
float32 cam_x                  # 카메라 광학 좌표 (x 오른쪽, y 아래, z 앞) [m]
float32 cam_y
float32 cam_z
