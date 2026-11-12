# ==========================================================================
# RoleAssignment.msg
# --------------------------------------------------------------------------
# role_assigner_node 가 발행하는 "이 로봇의 역할 배정" 결과.
#   - step1(관찰 전용): 디버그/RViz 용. 제어에 직접 관여하지 않는다.
#   - 다음 단계: positioning(home 좌표) / 전략 FSM 의 입력으로 그대로 쓰인다.
#
# [배정 원리 — 분산 일관성]
#   모든 로봇이 같은 /game_state(penalty 포함)를 보고 *동일한* 규칙으로 계산하므로,
#   팀 통신 없이도 로봇끼리 역할이 겹치지 않는다(아직 team_comm 미구현 단계의 안전책).
#   한계: penalty 가 아닌 "낙상한 팀메이트"는 남들 시야에선 여전히 활성으로 보여
#         슬롯을 차지한다 → team_comm(다음 단계)에서 보완.
# ==========================================================================
std_msgs/Header header

uint8 my_player_number        # 이 로봇 선수 번호(player_number 파라미터)
uint8 keeper_player_number    # GameState.our_goalkeeper (키퍼 번호). 규칙상 고정
uint8 active_field_count      # penalty 기준 활성 "필드플레이어" 수(키퍼 제외)
uint8 active_total_count      # 활성 필드 + (키퍼 활성 시 +1)

uint8 role                    # 아래 ROLE_* enum 값
string role_name              # 사람이 읽는 역할명(디버그/RViz 텍스트)
float32 home_x                # 이 역할의 home 좌표(map 프레임, m) — positioning 입력
float32 home_y
bool is_active                # 내가 penalty/낙상으로 빠졌으면 false

# ── 역할 enum ──
uint8 ROLE_NONE=0             # 비활성(penalty/낙상) — 아무 것도 안 함
uint8 ROLE_GOALKEEPER=1       # 키퍼(규칙상 고정). 우리 골대 = -X
uint8 ROLE_A_CENTER=2         # 중앙/수비 커버 + 킥오프 패스
uint8 ROLE_B_WING_POS=3       # y+ 윙 (4대 포메이션)
uint8 ROLE_C_WING_NEG=4       # y- 윙 (4대 포메이션)
uint8 ROLE_B_ATTACK=5         # x+ 공격 (3대 포메이션)
uint8 ROLE_STRIKER_ROAM=6     # 전구역 스트라이커 (2대 이하)
uint8 ROLE_SUPPORTER=7        # 여분 필드플레이어(4대 초과 시)
