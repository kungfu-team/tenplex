#!/bin/sh
set -e

# https://superuser.com/questions/1146388/systemd-state-stop-sigterm-timed-out

pid=$(pgrep -f /usr/bin/mlfsd)
kill -9 $pid

echo "killed $pid"
