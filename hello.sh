#!/bin/bash
# 第一行告诉系统用什么解释器执行这个脚本

echo "========== 系统信息 ==========="
echo "当前时间: $(date)"
echo "当前用户: $(whoami)"
echo "当前目录: $(pwd)"
echo "操作系统: $(cat /etc/os-release)"
echo "============== 完毕 =================="
