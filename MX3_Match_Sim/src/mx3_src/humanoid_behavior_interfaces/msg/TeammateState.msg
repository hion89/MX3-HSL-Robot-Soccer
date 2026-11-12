# ==========================================================================
# TeammateState.msg
# --------------------------------------------------------------------------
# team_comm_node 가 UDP(포트 10000+team_number)로 받은 "팀메이트 1명의 상태".
# 우리 로봇이 GameController 로 보내는 데이터(자세/공/넘어짐)에 전략 정보
# (역할/단계/공거리/공 주장)를 얹은 것이다. TeamComm.msg 의 배열 원소.
#
# [좌표계] 모두 map 프레임. 거리/좌표 단위는 m, 각도는 rad.
#   (GC return 패킷은 규격상 mm 였지만, 팀 내부 통신은 우리 포맷이라 m 로 통일)
# ==========================================================================
std_msgs/Header header

uint8 player_number       # 보낸 로봇 선수 번호(1부터)
bool  fallen              # true=넘어짐/복구중(서서 플레이 불가)

# ── 자세(map, m/rad) ──
bool    pose_valid        # 자세 추정값을 한 번이라도 받았는지
float32 pose_x
float32 pose_y
float32 pose_theta

# ── 공(map 절대좌표, m) ──
bool    ball_valid        # 최근 공을 봤는지(ball_age 가 신선한지)
float32 ball_x
float32 ball_y
float32 ball_age          # 마지막 공 관측 후 경과초(-1=한번도 못봄)
float32 ball_conf         # 공 인식 신뢰도 0~1 (★예약 필드, KHTM v2).
                          #   현재는 placeholder: ball_valid 면 1.0, 아니면 0.0.
                          #   향후 '③ 풀세트(cost+합의)'에서 다수 합의 이상치(유령
                          #   공) 배제 가중치로 사용 예정. 지금 단순판은 읽지 않는다.
                          #   ※실제 신뢰도 소스(YOLO conf / detection_memory 안정도)는
                          #     나중에 team_comm_node 가 구독해 채운다(현재 TODO).

# ── 전략 ──
uint8  role               # RoleAssignment.ROLE_* enum (0=NONE)
uint8  phase              # 아래 PHASE_* 코드(FSM strategy_phase 매핑)
float32 ball_dist         # 이 로봇 ~ 공 거리[m] (-1=공 위치 모름). owner 선출 입력.
bool   want_ball          # 이 로봇이 공을 잡으러 가는 중인지(CHASE/APPROACH/ALIGN_KICK)

# ── 통신 메타 ──
uint32  seq               # 송신측 시퀀스 번호(손실/순서 감지)
float32 age_sec           # 이 정보를 마지막으로 수신한 뒤 경과초(수신측이 채움). stale 판정용.
bool    stale             # age_sec 가 timeout 초과 → 끊김/낙상 의심(true 면 비활성 취급)

# ── 상수: phase 코드(FSM strategy_phase 문자열 ↔ 코드) ──
uint8 PHASE_UNKNOWN=0
uint8 PHASE_IDLE=1
uint8 PHASE_POSITION=2
uint8 PHASE_POSITION_DONE=3
uint8 PHASE_CHASE=4
uint8 PHASE_SUPPORT=5
uint8 PHASE_STOP=6
uint8 PHASE_APPROACH=7
uint8 PHASE_ALIGN_KICK=8
uint8 PHASE_NO_OWNER=9
uint8 PHASE_BALL_SEARCH=10
