#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_SCRIPT="$SCRIPT_DIR/sport_mode_state_to_odom.py"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f /home/mobile/catkin_ws/devel/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /home/mobile/catkin_ws/devel/setup.bash
fi
if [[ -f /home/mobile/ranger_ws/devel/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /home/mobile/ranger_ws/devel/setup.bash
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  UNITREE_PYTHON_ENV="venv:$VIRTUAL_ENV"
elif [[ -n "${UNITREE_CONDA_ENV:-}" ]]; then
  CONDA_SETUP="${UNITREE_CONDA_SH:-/home/mobile/miniconda3/etc/profile.d/conda.sh}"
  if [[ ! -f "$CONDA_SETUP" ]]; then
    echo "ERROR: Conda setup script not found: $CONDA_SETUP" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SETUP"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$UNITREE_CONDA_ENV" ]]; then
    if ! conda activate "$UNITREE_CONDA_ENV"; then
      echo "ERROR: Cannot activate UNITREE_CONDA_ENV=$UNITREE_CONDA_ENV" >&2
      echo "Unset UNITREE_CONDA_ENV when SDK2 is installed outside Conda." >&2
      exit 1
    fi
  fi
  UNITREE_PYTHON_ENV="conda:$UNITREE_CONDA_ENV"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  # Keep an already-active Conda environment, but do not require Conda.
  UNITREE_PYTHON_ENV="conda-current:$CONDA_DEFAULT_ENV"
else
  UNITREE_PYTHON_ENV="current-python"
fi

UNITREE_PYTHON_BIN="${UNITREE_PYTHON_BIN:-python3}"
UNITREE_NETWORK_INTERFACE="${UNITREE_NETWORK_INTERFACE:-enxec9a0c1bc5be}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
SPORT_STATE_TOPIC="${SPORT_STATE_TOPIC:-rt/sportmodestate}"
ODOM_TOPIC="${ODOM_TOPIC:-/go2/sport_odom}"
ODOM_FRAME_ID="${ODOM_FRAME_ID:-go2_odom}"
BASE_FRAME_ID="${BASE_FRAME_ID:-base_link}"
SPORT_STATE_TIMEOUT_SEC="${SPORT_STATE_TIMEOUT_SEC:-0.5}"
SPORT_ODOM_RATE_HZ="${SPORT_ODOM_RATE_HZ:-50.0}"

if ! PYTHON_EXECUTABLE="$(command -v "$UNITREE_PYTHON_BIN")"; then
  echo "ERROR: Python executable not found: $UNITREE_PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BRIDGE_SCRIPT" ]]; then
  echo "ERROR: Go2 odometry bridge not found: $BRIDGE_SCRIPT" >&2
  exit 1
fi
if ! "$PYTHON_EXECUTABLE" -c "import rospy; import nav_msgs.msg; import unitree_sdk2py" >/dev/null 2>&1; then
  echo "ERROR: rospy, nav_msgs and unitree_sdk2py must be importable by $PYTHON_EXECUTABLE" >&2
  exit 1
fi

echo "Unitree network interface: $UNITREE_NETWORK_INTERFACE"
echo "Unitree DDS domain:        $UNITREE_DOMAIN_ID"
echo "Unitree Python env:        $UNITREE_PYTHON_ENV"
echo "Unitree Python executable: $PYTHON_EXECUTABLE"
echo "SportModeState topic:     $SPORT_STATE_TOPIC"
echo "ROS odometry topic:       $ODOM_TOPIC"
echo "ROS odometry rate:        $SPORT_ODOM_RATE_HZ Hz"

exec "$PYTHON_EXECUTABLE" "$BRIDGE_SCRIPT" \
  --network_interface "$UNITREE_NETWORK_INTERFACE" \
  --domain_id "$UNITREE_DOMAIN_ID" \
  --sdk_topic "$SPORT_STATE_TOPIC" \
  --odom_topic "$ODOM_TOPIC" \
  --odom_frame "$ODOM_FRAME_ID" \
  --base_frame "$BASE_FRAME_ID" \
  --state_timeout_sec "$SPORT_STATE_TIMEOUT_SEC" \
  --publish_rate_hz "$SPORT_ODOM_RATE_HZ"
