# 전략 → 스킬 요청 (B-Human SkillRequest 대응)
std_msgs/Header header
uint8 type
uint8 STAND=0
uint8 WALK_TO_POSE=1          # target 으로 이동 후 정지
uint8 PLAY_BALL=2             # 공으로 가서 kick_target 방향으로 찬다
uint8 SEARCH_BALL=3           # 제자리/이동 탐색
uint8 OBSERVE_BALL=4          # target 에 서서 공을 본다 (수비/지원)
float32 target_x               # 필드 좌표 목표 pose
float32 target_y
float32 target_theta
float32 kick_target_x          # 필드 좌표 킥 목표점
float32 kick_target_y
uint8 position_role
uint8 ROLE_NONE=0
uint8 ROLE_GOALKEEPER=1
uint8 ROLE_DEFENDER=2
uint8 ROLE_MIDFIELDER=3
uint8 ROLE_FORWARD=4
bool active                    # 공을 다루는 선수인가 (PlayBall)
