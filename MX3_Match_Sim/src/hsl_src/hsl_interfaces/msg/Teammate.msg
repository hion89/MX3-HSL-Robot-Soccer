# 팀 통신으로 받은 팀원 상태 (B-Human Teammate 대응)
builtin_interfaces/Time received
uint8 player_number
bool penalized
bool fallen
float32 x
float32 y
float32 theta
float32 pose_validity
bool ball_valid
float32 ball_x                 # 필드 좌표
float32 ball_y
float32 ball_vel_x
float32 ball_vel_y
float32 ball_age               # 보낸 시점 기준 마지막으로 본 뒤 경과 [s]
float32 time_to_ball           # 보낸 쪽이 계산한 공 도달 예상 시간 [s]
uint8 position_role            # SkillRequest.ROLE_*
bool playing_ball
