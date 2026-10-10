#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_python="${DIESEL_KG_PYTHON:-python3.12}"
"$task_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Please use Python 3.12"'
"$task_python" -m venv --clear "$task_root/.venv"
"$task_root/.venv/bin/python" -m pip install -r "$task_root/requirements.lock.txt"
"$task_root/.venv/bin/python" -m pip check
echo "环境已安装。运行 $task_root/run doctor 检查项目。"
