import time

from types import TracebackType

print("abc")

arr = range(0, 10)
for item in arr:
    print(f"item: {item}")

def sum_all(*args):
    return sum(args)

# 不能传入元组
# 可变参数 *args（接收任意数量的位置参数，打包成元组）
result = sum_all(1,2,3,4,5)
print("result: ", result)

# 关键字参数 **args（接收任意数量的关键字参数，打包成字典）
def print_info(**args):
    for k, v in args.items():
        print(f"key: {k}, value: {v}")

print_info(name="张飞", age=12)

class Agent:
    """AI Agent 基类"""

    # 类属性
    category = "AI"

    def __init__(self, name, model="gpt-4"):
        self.name = name
        self.model = model
        self._memory = []   # 单下划线：约定为内部使用

    # 实例方法
    def introduce(self):
        return f"我是 {self.name}，使用 {self.model} 模型"

# 继承
class CodingAgent(Agent):
    def __init__(self, name, model="gpt-4", language=None):
        super().__init__(name, model)
        self.language = language or ["Python"]

    def code_review(self, code: str) -> str:
        return f"[{self.name}] 代码审查完成，发现3个优化建议"

agent = CodingAgent(name="test", language=["Go", "PHP"])
print(agent.introduce())
print(agent.category)
print(agent.code_review("print('hello!')"))

tools = ["1", 2, 3, "4"]
tools.append(5)
print("tools: ", tools)
tools.insert(10, 10)
print("tools: ", tools)
tools.insert(-1, "11")
print("tools: ", tools)

# 装饰器
def timing_decorator(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[计时]: {func.__name__} 执行耗时: {elapsed:.3f}秒")
        return result
    return wrapper


@timing_decorator
def slow_func():
    time.sleep(1.5)
    return "success"

print(slow_func())

# 带参数的装饰器
def retry(max_attempts=3, delay=1):
    """重试装饰器"""
    import time
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    if max_attempts == i:
                        raise
                    print(f"重试：{i} 次")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def call_http():
    print(f"打印 call_http -------------------")
    raise "错误了"

call_http()

def log_tool_call(func):
    def wrapper(*args, **kwargs):
        print(f"[Tool] 调用 {func.__name__}(args={args}, kwargs={kwargs})")
        result = func(*args, **kwargs)
        print(f"[Tool] {func.__name__} 返回 {result}")
    return wrapper

@log_tool_call
def search_web(query: str):
    return f"搜索结果是---------"

# 限流场景保护
# last_called闭包。装饰器执行一次 ≠ 装饰器内部所有变量都永久存在。只有被返回函数形成闭包引用的变量，才会随着被装饰函数生命周期一直存在。
def rate_limit(max_calls_per_second):
    import time
    interval = 1.0/max_calls_per_second
    last_called = [0]
    def decorator(func):
        def wrapper(*args, **kwargs):
            ts = time.time() - last_called[0]
            if ts < interval:
                time.sleep(interval-ts)
            func(*args, **kwargs)
            last_called[0]=time.time()
        return wrapper
    return decorator

@rate_limit(max_calls_per_second=3)
def print_num():
    print("hello")

for i in range(5):
    print_num()

# 自定义上下文管理器
class TimerContext:
    import time
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None):
        ts = time.time() - self.start
        print(f"执行耗时：{ts}")
        return False

# with TimerContext():
#     print([(y, i) for y in ["a", "b"] for i in range(10_000)])

with open("../moe-config.yaml", "r", encoding="utf-8") as fs:
    print(f"fs.read(): {fs.read()}")

# 复合类型
from typing import Optional, Union, List, Dict, Tuple
# Optional - 可以是 None
def get_user(id: int) -> Optional[str]:
    """可能返回None"""
    users = {1: "Alice", 2: "Tom"}
    return users.get(id)

# Union - 多种类型之一
def process(data: Union[str, bytes]) -> str:
    """接收str或bytes，返回字符串"""
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data

# tools: list[str] = ["abc", "def"]
# config: dict[str, str] = {"name": "app", "addr": "127.0.0.1"}
# # 固定长度和类型的元组
# tpl: tuple[float, float] = (1.1, 1.2)

import math

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, math.sqrt(n)+1):
        if n % i == 0:
            break
    else:
        return True
    return False

class Student:
    name: str
    scores: dict[str, float]

    def __init__(self, name: str, scores: dict[str, float]):
        self.name = name
        self.scores = scores

    def average_score(self) -> float:
        return round(sum(self.scores.values()) / len(self.scores), 2)

    def best_subject(self) -> str:
        return max(self.scores, key=self.scores.get)

    # 所有大于60分的学科名字
    def large60_subject(self) -> list[str]:
        return [k for k, v in self.scores.items() if v > 60]

stu = Student(name="张飞", scores={"math": 90, "chinese": 30, "english": 93})
print(f"\n stu.average_score(): {stu.average_score()}")
print(f"\n stu.best_subject(): {stu.best_subject()}")
print(f"\n stu.large60_subject(): {stu.large60_subject()}")

# 保留原函数信息 functools.wraps
from functools import wraps

# 元组可以直接作为map的key
def cache_result(func):
    result = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args in result:
            print("走了缓存读取##############")
            return result.get(args)
        print("走了真实计算--------------")
        cal_result = func(*args, **kwargs)
        result[args] = cal_result
        return cal_result
    return wrapper

@cache_result
def cal_func(a: int, b: int) -> int:
    return a + b

print(cal_func(1, 3))
print(cal_func(2, 4))
print(cal_func(1, 3))
print(cal_func.__name__)    # 不加@wraps(func)，会输出wrapper。加了之后才会输出cal_func

# 装饰器实现一个工具注册表
tool_registry = {}

def registry(name: str):            # 第一层：接收配置参数
    def decorator(func):            # 第二层：接收函数
        tool_registry[name] = func  # 注册
        return func                 # 关键：原样返回，不破坏函数
    return decorator

@registry(name="web_search")
def web_search(query: str) -> list[str]:
    return [f"请求：{query}，结果是。。。。"]

search_res = web_search("test")
print(f"search_res: {search_res}")
print(f"tool_registry: {tool_registry}")

"""
● 装饰器的本质

  装饰器就是一个接收函数、返回函数的函数。@xxx 只是语法糖，等价于 fn = xxx(fn)。

  ---
  三种结构

  1. 最简单：无参装饰器（两层）
  def decorator(func):
      def wrapper(*args, **kwargs):
          # 前置逻辑
          result = func(*args, **kwargs)
          # 后置逻辑
          return result
      return wrapper

  @decorator
  def my_func(): ...

  2. 带参数装饰器（三层）
  def decorator(param):        # 第一层：接收配置
      def inner(func):         # 第二层：接收函数
          def wrapper(*args, **kwargs):  # 第三层：实际执行
              return func(*args, **kwargs)
          return wrapper
      return inner

  @decorator(param="xxx")      # 必须加括号
  def my_func(): ...

  3. 注册型装饰器（不替换函数）
  def registry(func):
      tool_registry[func.__name__] = func
      return func              # 原样返回，不包装

  @registry
  def my_func(): ...

  ---
  核心规则

  ┌───────────────────────────┬─────────────────────────────────────────┐
  │           规则            │                  说明                   │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ 必须返回函数              │ wrapper 忘记 return，调用方得到 None    │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ 带参数必须加括号          │ 三层结构 @deco() vs 两层 @deco          │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ 用 *args, **kwargs        │ 保证 wrapper 兼容任意签名               │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ 加 @functools.wraps(func) │ 保留原函数的 __name__、__doc__ 等元信息 │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ 注册型原样 return func    │ 不需要包装时直接返回，不破坏函数本身    │
  └───────────────────────────┴─────────────────────────────────────────┘
"""

"""
装饰器 = 函数增强器

组成：
1. decorator 接收函数
2. wrapper 包裹函数
3. return wrapper


规则：
1. @xxx 等价于 func=xxx(func)
2. 定义时执行装饰器
3. 调用时执行 wrapper
4. 参数用 *args/**kwargs
5. 返回值必须 return
6. wraps 保留元信息
7. 多装饰器从下往上套
8. 带参数装饰器多一层
"""