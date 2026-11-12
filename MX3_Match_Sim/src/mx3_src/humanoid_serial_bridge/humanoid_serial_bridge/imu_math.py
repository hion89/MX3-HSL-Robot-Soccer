#!/usr/bin/env python3
"""
imu_math.py
===========

IMU 관련 공용 수식 모음.

────────────────────────────────────────────────────────────────────────────
[2026-08 CPU 최적화] projected gravity 를 구독 대신 파생 계산하기 위해 신설
────────────────────────────────────────────────────────────────────────────
■ 배경

  imu_parser_node 는 200Hz 로 두 토픽을 발행해 왔다.

      /imu/data                (sensor_msgs/Imu)       200Hz
      /imu/projected_gravity   (geometry_msgs/Vector3Stamped) 200Hz

  그런데 후자는 MadgwickAHRS.gravity_vector_body() 의 결과이고, 그 함수는
  쿼터니언만의 순수 함수다(필터 상태도, 추가 센서 입력도 쓰지 않는다).
  그리고 그 쿼터니언은 이미 /imu/data.orientation 에 실려 나간다.

  → /imu/projected_gravity 는 정보량이 0 인 중복 토픽이다.

  두 토픽을 다 구독하던 노드(정책 러너 2개, fall_manager)는 같은 정보를 받으려고
  executor 를 초당 400번 깨우고 있었다. rclpy 는 깨어날 때마다 wait set 을
  Python 레벨에서 재구축하므로, 이 중복이 그대로 CPU 비용이 된다.
  (실측 근거: mcu_bridge 에서 200 깨우기/초를 제거했더니 27.9% → 14.5%)

  이 모듈의 함수로 /imu/data 하나만 받아 중력벡터를 직접 만들면
  깨우기가 400 → 200 으로 줄고, 값과 신선도는 완전히 동일하다.

■ ★ 동기화 의무 ★

  아래 수식은 imu_parser_node.MadgwickAHRS.gravity_vector_body() 와
  수식·연산 순서·부호가 모두 동일하다. 부동소수 반올림까지 같게 만들려고
  일부러 같은 순서로 적었다.

  둘 중 한쪽을 고치면 반드시 다른 쪽도 같이 고칠 것.
  이 값은 정책 관측(observation)에 직접 들어가므로, 어긋나면 학습 시점과
  다른 입력이 들어가 보행이 무너진다.

■ 좌표/부호 규약

  Isaac-style projected gravity (body frame).
  정자세일 때 대체로 (0, 0, -1) 이다. 즉 '중력이 향하는 방향'이지
  '중력 반대 방향(up)'이 아니다. 부호를 헷갈리면 로봇이 뒤집힌 것으로
  판정되므로 주의할 것.
────────────────────────────────────────────────────────────────────────────
"""


def projected_gravity_from_quat(q):
    """쿼터니언에서 body frame projected gravity 를 직접 계산한다.

    Args:
        q: x, y, z, w 속성을 가진 객체.
           geometry_msgs/Quaternion (즉 sensor_msgs/Imu.orientation) 을
           그대로 넘기면 된다.

    Returns:
        (gx, gy, gz) tuple[float, float, float]
        정자세에서 대략 (0.0, 0.0, -1.0).

    Note:
        imu_parser_node.MadgwickAHRS 는 내부적으로 (q1, q2, q3, q4) =
        (w, x, y, z) 순서를 쓰고, quaternion() 이 (q2, q3, q4, q1) =
        (x, y, z, w) 로 바꿔 반환한다. 아래 식은 그 대응을 반영한 것이다.

            원본 gravity_vector_body():
                gx = 2.0 * (q2 * q4 - q1 * q3)      q1=w, q2=x, q3=y, q4=z
                gy = 2.0 * (q1 * q2 + q3 * q4)
                gz = q1*q1 - q2*q2 - q3*q3 + q4*q4
                return -gx, -gy, -gz
    """
    # 지역 변수로 뽑아둔다. 200Hz 경로에서 속성 접근을 반복하지 않기 위함이며,
    # 동시에 원본 함수와 변수 대응을 눈으로 확인하기 쉬워진다.
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    # 회전행렬 3번째 열(= body frame 에서 본 world 의 +z 축)에 해당한다.
    # 원본과 곱셈 순서까지 동일하게 유지 — 부동소수 결과를 비트 단위로 맞추기 위함.
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (w * x + y * z)
    gz = w * w - x * x - y * y + z * z

    # 부호 반전: '위쪽(up)' 이 아니라 '중력이 향하는 방향' 으로 돌려준다.
    return -gx, -gy, -gz
