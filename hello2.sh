#!/bin/bash

# 定义变量
name="AI Agent"
count=42
pi=3.14

# 使用变量
echo "项目名：$name"
echo "数量：$count"

# 使用变量，花括号和$都可以使用变量，花括号为了消除歧义
echo "项目名：${name}"
echo "数量：${count}"

echo "$name 名字！"

# 命令替换，使用$()
current_date=$(date +%Y-%m-%d)
echo "当前时间是：$current_date"

py_file_count=$(ls | grep ".*\.py" | wc -l)
echo "当前目录有：$py_file_count个py文件"

