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
else
  UNITREE_CONDA_ENV="${UNITREE_CONDA_ENV:-${CONDA_DEFAULT_ENV:-unitree}}"
  if [ -f /home/mobile/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/mobile/miniconda3/etc/profile.d/conda.sh
    if [ "${CONDA_DEFAULT_ENV:-}" != "$UNITREE_CONDA_ENV" ]; then
      conda activate "$UNITREE_CONDA_ENV"
    fi
  fi
  UNITREE_PYTHON_ENV="conda:${CONDA_DEFAULT_ENV:-$UNITREE_CONDA_ENV}"
fi

ACTION_TOPIC="${ACTION_TOPIC:-/falcon/action_id}"
UNITREE_NETWORK_INTERFACE="${UNITREE_NETWORK_INTERFACE:-eth0}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
UNITREE_TIMEOUT_SEC="${UNITREE_TIMEOUT_SEC:-10.0}"
FORWARD_SPEED="${FORWARD_SPEED:-0.6}"
TURN_SPEED="${TURN_SPEED:-0.6}"
ACTION_TIMEOUT_SEC="${ACTION_TIMEOUT_SEC:-0.3}"
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

exec "$PYTHON_EXECUTABLE" "$MAPPER_SCRIPT" \
  --action_topic "$ACTION_TOPIC" \
  --network_interface "$UNITREE_NETWORK_INTERFACE" \
  --domain_id "$UNITREE_DOMAIN_ID" \
  --sdk_timeout_sec "$UNITREE_TIMEOUT_SEC" \
  --forward_speed "$FORWARD_SPEED" \
  --turn_speed "$TURN_SPEED" \
  --action_timeout_sec "$ACTION_TIMEOUT_SEC" \
  --dry_run "$UNITREE_DRY_RUN" \
  --balance_stand_on_start "$UNITREE_BALANCE_STAND_ON_START"
