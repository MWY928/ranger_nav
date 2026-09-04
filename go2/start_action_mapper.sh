#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPER_SCRIPT="$SCRIPT_DIR/unitree_action_mapper.py"

if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f /home/mobile/catkin_ws/devel/setup.bash ]; then
  source /home/mobile/catkin_ws/devel/setup.bash
fi
if [ -f /home/mobile/ranger_ws/devel/setup.bash ]; then
  source /home/mobile/ranger_ws/devel/setup.bash
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
  UNITREE_PYTHON_ENV="venv:$VIRTUAL_ENV"
elif [ -n "${UNITREE_CONDA_ENV:-}" ]; then
  CONDA_SETUP="${UNITREE_CONDA_SH:-/home/mobile/miniconda3/etc/profile.d/conda.sh}"
  if [ ! -f "$CONDA_SETUP" ]; then
    echo "ERROR: Conda setup script not found: $CONDA_SETUP" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SETUP"
  if [ "${CONDA_DEFAULT_ENV:-}" != "$UNITREE_CONDA_ENV" ]; then
    if ! conda activate "$UNITREE_CONDA_ENV"; then
      echo "ERROR: Cannot activate UNITREE_CONDA_ENV=$UNITREE_CONDA_ENV" >&2
      echo "Unset UNITREE_CONDA_ENV when SDK2 is installed outside Conda." >&2
      exit 1
    fi
  fi
  UNITREE_PYTHON_ENV="conda:$UNITREE_CONDA_ENV"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  # Keep an already-active Conda environment, but do not require Conda.
  UNITREE_PYTHON_ENV="conda-current:$CONDA_DEFAULT_ENV"
else
  UNITREE_PYTHON_ENV="current-python"
fi

ACTION_TOPIC="${ACTION_TOPIC:-/falcon/action_id}"
UNITREE_NETWORK_INTERFACE="${UNITREE_NETWORK_INTERFACE:-enxec9a0c1bc5be}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
UNITREE_TIMEOUT_SEC="${UNITREE_TIMEOUT_SEC:-10.0}"
FORWARD_SPEED="${FORWARD_SPEED:-0.6}"
TURN_SPEED="${TURN_SPEED:-0.6}"
SEARCH_TURN_SPEED="${SEARCH_TURN_SPEED:-0.25}"
ACTION_TIMEOUT_SEC="${ACTION_TIMEOUT_SEC:-0.3}"
WATCHDOG_RATE_HZ="${WATCHDOG_RATE_HZ:-20.0}"
VELOCITY_SMOOTHING_ENABLED="${VELOCITY_SMOOTHING_ENABLED:-true}"
LINEAR_ACCEL_LIMIT="${LINEAR_ACCEL_LIMIT:-1.0}"
LINEAR_DECEL_LIMIT="${LINEAR_DECEL_LIMIT:-1.5}"
YAW_ACCEL_LIMIT="${YAW_ACCEL_LIMIT:-2.0}"
YAW_DECEL_LIMIT="${YAW_DECEL_LIMIT:-3.0}"
UNITREE_DRY_RUN="${UNITREE_DRY_RUN:-false}"
UNITREE_BALANCE_STAND_ON_START="${UNITREE_BALANCE_STAND_ON_START:-false}"
UNITREE_PYTHON_BIN="${UNITREE_PYTHON_BIN:-python3}"

if ! PYTHON_EXECUTABLE="$(command -v "$UNITREE_PYTHON_BIN")"; then
  echo "ERROR: Python executable not found: $UNITREE_PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$MAPPER_SCRIPT" ]; then
  echo "ERROR: Unitree action mapper script not found: $MAPPER_SCRIPT" >&2
  exit 1
fi

if ! "$PYTHON_EXECUTABLE" -c "import rospy" >/dev/null 2>&1; then
  echo "ERROR: rospy is not importable with $PYTHON_EXECUTABLE" >&2
  echo "Make sure /opt/ros/noetic/setup.bash has been sourced." >&2
  exit 1
fi

if [ "$UNITREE_DRY_RUN" != "true" ] && \
   ! "$PYTHON_EXECUTABLE" -c "import unitree_sdk2py" >/dev/null 2>&1; then
  echo "ERROR: unitree_sdk2py is not importable with $PYTHON_EXECUTABLE" >&2
  echo "Activate the correct venv before starting this script." >&2
  exit 1
fi

echo "Action topic:              $ACTION_TOPIC"
echo "Unitree Python env:        $UNITREE_PYTHON_ENV"
echo "Unitree Python executable: $PYTHON_EXECUTABLE"
echo "Unitree network interface: $UNITREE_NETWORK_INTERFACE"
echo "Unitree dry run:           $UNITREE_DRY_RUN"
echo "Control/watchdog rate:     $WATCHDOG_RATE_HZ Hz"
echo "Velocity smoothing:        $VELOCITY_SMOOTHING_ENABLED"
echo "Normal/search turn speed:  $TURN_SPEED / $SEARCH_TURN_SPEED rad/s"
echo "Linear accel/decel:        $LINEAR_ACCEL_LIMIT / $LINEAR_DECEL_LIMIT m/s^2"
echo "Yaw accel/decel:           $YAW_ACCEL_LIMIT / $YAW_DECEL_LIMIT rad/s^2"

exec "$PYTHON_EXECUTABLE" "$MAPPER_SCRIPT" \
  --action_topic "$ACTION_TOPIC" \
  --network_interface "$UNITREE_NETWORK_INTERFACE" \
  --domain_id "$UNITREE_DOMAIN_ID" \
  --sdk_timeout_sec "$UNITREE_TIMEOUT_SEC" \
  --forward_speed "$FORWARD_SPEED" \
  --turn_speed "$TURN_SPEED" \
  --search_turn_speed "$SEARCH_TURN_SPEED" \
  --action_timeout_sec "$ACTION_TIMEOUT_SEC" \
  --watchdog_rate_hz "$WATCHDOG_RATE_HZ" \
  --smoothing_enabled "$VELOCITY_SMOOTHING_ENABLED" \
  --linear_accel_limit "$LINEAR_ACCEL_LIMIT" \
  --linear_decel_limit "$LINEAR_DECEL_LIMIT" \
  --yaw_accel_limit "$YAW_ACCEL_LIMIT" \
  --yaw_decel_limit "$YAW_DECEL_LIMIT" \
  --dry_run "$UNITREE_DRY_RUN" \
  --balance_stand_on_start "$UNITREE_BALANCE_STAND_ON_START"
