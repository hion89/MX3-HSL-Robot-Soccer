# ==========================================================================
# GameState.msg
# --------------------------------------------------------------------------
# RoboCup HSL GameController(헤더 "RGme", STRUCT_VERSION 20) 패킷을
# 구조체 정의 그대로 파싱한 뒤, "우리 팀 관점"으로 정리해 발행하는 메시지.
#
# - 원본 패킷은 teams[2] 안에 양 팀 정보를 담지만, 이 메시지는 team_number
#   파라미터로 우리 팀을 골라 our_/opponent_ 로 풀어서 담는다.
#   (control_ws 의 문자열 game_status 토픽을 대체한다)
# - 모든 enum 값은 RoboCupGameControlData.h 의 #define 과 1:1 동일하다.
# - 아래 상수를 쓰면 다운스트림 코드에서 매직 넘버 없이
#   GameState.STATE_PLAYING 처럼 의미로 분기할 수 있다.
# ==========================================================================

std_msgs/Header header

# ── 패킷 메타 ──
uint8 version              # 수신 패킷의 STRUCT_VERSION (정상 = 20)
uint8 packet_number        # 패킷 일련번호(0~255 랩어라운드)

# ── 경기 메타 ──
uint8 players_per_team     # 팀당 선수 수
uint8 competition_type     # 0=SMALL 1=MIDDLE 2=LARGE
bool  stopped              # true = 현재 플레이 정지(공이 인플레이 아님)
uint8 game_phase           # 0=NORMAL 1=PENALTY_SHOOTOUT 2=EXTRA_TIME 3=TIMEOUT
uint8 state                # 0=INITIAL 1=READY 2=SET 3=PLAYING 4=FINISHED
uint8 set_play             # 0=NONE 1=DIRECT_FK 2=INDIRECT_FK 3=PENALTY_KICK
                           #   4=THROW_IN 5=GOAL_KICK 6=CORNER_KICK
bool  first_half           # true = 전반
int16 secs_remaining       # 현재 하프 남은 시간(초, 추정)
int16 secondary_time       # 보조 시간(ready 남은 시간, free ball 까지 등)

# ── 우리 팀 관점 파생 정보 ──
uint8 our_team_number      # 우리 팀 번호(team_number 파라미터)
bool  team_found           # 패킷 안에서 우리 팀을 찾았는지. false면 아래 our_* 무효
bool  we_are_kicking       # 다음 재개(킥오프/프리킥 등) 주도권이 우리면 true
uint8 our_score
uint8 opponent_score
uint8 our_goalkeeper       # 우리 팀 골키퍼의 선수 번호

# ── 우리 팀 로봇별 상태 (index i  =  선수 번호 i+1) ──
# 길이는 players_per_team. penalty 0 = 정상.
uint8[] penalties              # 각 선수 penalty 코드
uint8[] secs_till_unpenalised  # 패널티 해제까지 남은 초(추정)
uint8[] cautions               # 경고(옐로카드) 누적 수

# ── 상수: state ──
uint8 STATE_INITIAL=0
uint8 STATE_READY=1
uint8 STATE_SET=2
uint8 STATE_PLAYING=3
uint8 STATE_FINISHED=4

# ── 상수: game_phase ──
uint8 GAME_PHASE_NORMAL=0
uint8 GAME_PHASE_PENALTY_SHOOTOUT=1
uint8 GAME_PHASE_EXTRA_TIME=2
uint8 GAME_PHASE_TIMEOUT=3

# ── 상수: set_play ──
uint8 SET_PLAY_NONE=0
uint8 SET_PLAY_DIRECT_FREE_KICK=1
uint8 SET_PLAY_INDIRECT_FREE_KICK=2
uint8 SET_PLAY_PENALTY_KICK=3
uint8 SET_PLAY_THROW_IN=4
uint8 SET_PLAY_GOAL_KICK=5
uint8 SET_PLAY_CORNER_KICK=6

# ── 상수: competition_type ──
uint8 COMPETITION_TYPE_SMALL=0
uint8 COMPETITION_TYPE_MIDDLE=1
uint8 COMPETITION_TYPE_LARGE=2

# ── 상수: penalty ──
uint8 PENALTY_NONE=0
uint8 PENALTY_ILLEGAL_POSITIONING=1
uint8 PENALTY_MOTION_IN_SET=2
uint8 PENALTY_MOTION_IN_STOP=3
uint8 PENALTY_LOCAL_GAME_STUCK=4
uint8 PENALTY_INCAPABLE_ROBOT=5
uint8 PENALTY_PICK_UP=6
uint8 PENALTY_BALL_HOLDING=7
uint8 PENALTY_LEAVING_THE_FIELD=8
uint8 PENALTY_PLAYING_WITH_ARMS_HANDS=9
uint8 PENALTY_PUSHING=10
uint8 PENALTY_CAUTIONED=11
uint8 PENALTY_SENT_OFF=12
uint8 PENALTY_SUBSTITUTE=13
