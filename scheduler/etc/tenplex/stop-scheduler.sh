#!/bin/sh
set -e

pid=$(pgrep -f /usr/bin/tenplex-scheduler)
kill -9 $pid

echo "killed $pid"
