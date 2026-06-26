#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f /home/mobile/catkin_ws/devel/setup.bash ]; then
  source /home/mobile/catkin_ws/devel/setup.bash
fi
if [ -f /home/mobile/ranger_ws/devel/setup.bash ]; then
  source /home/mobile/ranger_ws/devel/setup.bash
fi

UNITREE_CONDA_ENV="${UNITREE_CONDA_ENV:-unitree}"
if [ -f /home/mobile/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/mobile/miniconda3/etc/profile.d/conda.sh
  conda activate "$UNITREE_CONDA_ENV"
fi

ACTION_TOPIC="${ACTION_TOPIC:-/falcon/action_id}"
UNITREE_NETWORK_INTERFACE="${UNITREE_NETWORK_INTERFACE:-eth0}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
UNITREE_TIMEOUT_SEC="${UNITREE_TIMEOUT_SEC:-10.0}"
FORWARD_SPEED="${FORWARD_SPEED:-0.6}"
TURN_SPEED="${TURN_SPEED:-0.6}"
ACTION_TIMEOUT_SEC="${ACTION_TIMEOUT_SEC:-0.3}"
UNITREE_DRY_RUN="${UNITREE_DRY_RUN:-0}"
UNITREE_BALANCE_STAND_ON_START="${UNITREE_BALANCE_STAND_ON_START:-0}"

ARGS=(
  --action_topic "$ACTION_TOPIC"
  --network_interface "$UNITREE_NETWORK_INTERFACE"
  --domain_id "$UNITREE_DOMAIN_ID"
  --sdk_timeout_sec "$UNITREE_TIMEOUT_SEC"
  --forward_speed "$FORWARD_SPEED"
  --turn_speed "$TURN_SPEED"
  --action_timeout_sec "$ACTION_TIMEOUT_SEC"
)

if [ "$UNITREE_DRY_RUN" = "1" ]; then
  ARGS+=(--dry_run)
fi

if [ "$UNITREE_BALANCE_STAND_ON_START" = "1" ]; then
  ARGS+=(--balance_stand_on_start)
fi

echo "Action topic:              $ACTION_TOPIC"
echo "Unitree conda env:         $UNITREE_CONDA_ENV"
echo "Unitree network interface: $UNITREE_NETWORK_INTERFACE"
echo "Unitree dry run:           $UNITREE_DRY_RUN"

exec python "$REPO_ROOT/go2/unitree_action_mapper.py" "${ARGS[@]}"
