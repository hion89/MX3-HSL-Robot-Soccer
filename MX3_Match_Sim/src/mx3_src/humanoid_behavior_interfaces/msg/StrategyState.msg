# ==========================================================================
# StrategyState.msg
# --------------------------------------------------------------------------
# strategy_fsm_node(다음 단계, step3)가 발행할 "현재 전략 상태".
# step1 에서는 *메시지 정의만* 해 둔다. role_assigner_node 는 RoleAssignment 만
# 발행하고, 이 메시지는 FSM 노드가 들어오면서 채워진다.
#
# [의도]
#   경기상태 + 역할 + 공 책임여부를 한 토픽으로 묶어, 모니터링/Foxglove 에서
#   "지금 이 로봇이 무슨 판단을 하는지"를 한눈에 보게 한다.
# ==========================================================================
std_msgs/Header header

uint8 game_state              # GameState.state 미러(0=INITIAL 1=READY 2=SET 3=PLAYING 4=FINISHED)
string strategy_phase         # FSM 단계명: IDLE/POSITION/CHASE/SUPPORT/KEEP/STOP 등

humanoid_behavior_interfaces/RoleAssignment role   # 현재 역할 배정(role_assigner 결과)

bool ball_in_my_zone          # 공이 내 책임 구역에 있는지
bool ball_valid               # 공 관측 유효 여부(최근 본 적 있는지)
float32 ball_map_x            # 마지막으로 본 공 좌표(map 프레임, m)
float32 ball_map_y
