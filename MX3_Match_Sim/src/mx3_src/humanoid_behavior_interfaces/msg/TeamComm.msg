# ==========================================================================
# TeamComm.msg
# --------------------------------------------------------------------------
# team_comm_node 가 발행하는 "팀 통신 통합 상태".
# 자기 자신 + 수신된 모든 팀메이트의 상태를 모으고, 거리중재(누가 공을
# 잡을지) 결과를 파생값으로 제공한다. role_assigner/strategy_fsm 가 이걸
# 구독해 "낙상 팀메이트 제외 재배정" "owner 양보" 같은 협응을 하게 된다.
#
# [거리중재 — ball_owner_player]
#   활성(stale 아님 + fallen 아님 + ball_dist>=0)인 로봇 중 ball_dist 가
#   가장 작은 로봇을 공 owner 로 본다. 동점이면 player_number 가 작은 쪽.
#   후보가 없으면 0. 모든 로봇이 같은 입력으로 같은 결론을 내므로(분산 일관성)
#   통신만 살아 있으면 owner 가 겹치지 않는다.
# ==========================================================================
std_msgs/Header header

uint8 my_player_number    # 이 노드를 실행하는 로봇의 선수 번호
uint8 team_number

# ── 자기 자신 + 팀메이트 전체(자기 포함) ──
# index 순서는 보장하지 않음. player_number 로 식별할 것.
TeammateState[] members

# ── 파생: 거리중재 결과 ──
uint8 ball_owner_player   # 공을 잡아야 할 로봇 선수 번호(0=후보 없음)
bool  i_am_owner          # my_player_number == ball_owner_player
uint8 active_count        # 활성(stale/fallen 아님) 로봇 수(자기 포함)
