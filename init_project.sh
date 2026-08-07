#!/bin/bash
# 快速创建项目骨架

# 任何命令失败立即退出 -u：使用未定义变量时报错；-o pipefail：管道中任一步失败都视为失败
set -e

# 从命令行参数获取项目名，如果没有则用默认名
project_name="${1:-my_first_project}"

echo "🚀开始创建项目：$project_name"

# 创建目录结构
mkdir -p "$project_name"/{src/{api,core,models,tools,utils},tests,docs,scripts,config,logs}

for i in {1..16}; do
  echo "当前循环是：$i"
done