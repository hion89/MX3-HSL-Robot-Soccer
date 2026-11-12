cmake_minimum_required(VERSION 3.8)
project(humanoid_behavior_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# GameState.msg 가 std_msgs/Header 를 쓰므로 의존성 추가
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/RobotInit.srv"
  "srv/TorqueEnable.srv"
  "srv/TorqueEnableJoints.srv"
  "srv/GetJointLimits.srv"
  "srv/StandUp.srv"
  "srv/MotorReset.srv"
  "srv/Kick.srv"
  # ── 신규: GameController 상태 메시지 ──
  "msg/GameState.msg"
  # ── 신규: 전략 레이어 메시지 (humanoid_strategy 패키지가 소비) ──
  #   StrategyState 가 RoleAssignment 를 참조하므로 같은 패키지 안에 함께 등록한다.
  "msg/RoleAssignment.msg"
  "msg/StrategyState.msg"
  "msg/TeammateState.msg"
  "msg/TeamComm.msg"
  # GameState/StrategyState 가 std_msgs/Header 를 참조하므로 DEPENDENCIES 에 명시해야 함
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
