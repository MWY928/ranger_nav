#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/fake_polar_goal_pub.py" \
  --topic /tag_polar \
  --mode fixed \
  --r 1.5 \
  --theta 0.0 \
  --rate 20
