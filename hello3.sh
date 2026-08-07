#!/bin/bash

filename="config.yaml"

# 文件测试
if [ -f "$filename" ]; then
  echo "$filename 存在"
else
  echo "$filename 不存在，创建默认配置"
fi

# -f path：文件存在且是普通文件 （不是目录/设备/管道）
# -d path：目录是否存在
# -e path：路径是否存在（不区分是文件还是目录）
# -r path：是否可读
# -w path：是否可写
# -x path：是否可执行
# -s path：文件是否非空

# 字符串比较
if [ "$1" = "dev" ]; then
  echo "开发模式"
elif [ "$1" = "prod" ]; then
  echo "生产模式"
else
  "其他模式"
fi

count=$2
# 数字比较
if [ "$count" -lt 10 ]; then
  echo "$count 小于10"
else
  echo "$count 非小于10"
fi

echo $(ls | grep ".*\.py")

# 循环
for lan in $(ls | grep ".*\.py"); do
  echo "当前目录下的py文件有：$lan"
done

# 函数
function greet() {
  local name="$1" # local 限制变量作用域
  echo "第一个入参是：$name"
}

# 或省略function
greet_v2() {
  local name="$1" # local 限制变量作用域
  echo "第一个入参是：$name"
}

# 调用
greet $1
greet_v2 $2

# 解析短选项
#!/bin/bash
# getopts核心机制
# f: - 表示-f后面必须跟一个值
# n  - 没有冒号，纯开关，不跟值
# $OPTARG - getopts内置变量，自动存放当前选项后面跟着的那个值
while getopts "f:o:nh" opt; do
    case $opt in
        f) input_file="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        n) dry_run=true ;;          # 无参数选项
        h) echo "用法: $0 -f <文件> [-o <目录>] [-n]"; exit 0 ;;
        *) exit 1 ;;
    esac
done

input_file="${input_file:?请用 -f 指定输入文件}"
output_dir="${output_dir:-/tmp/output}"

# 调用：./script -f data.csv -o /out -n
