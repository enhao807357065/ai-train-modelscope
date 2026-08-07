# user_input = input("输入一个数字：")
user_input = 5
try:
    num = int(user_input)
    print(f"输入的数字是：{num}")
except Exception as e:
    # 如果try抛出异常，执行这里
    print("输入的内容格式有误!")

    # 3种抛出异常的方式
    # raise   # 不带参数的raise，原样重新抛出
    # raise ValueError("不能除0") # 直接抛出
    # raise ValueError(f"测试测试抛出异常") from e    # 用一个异常包装另一个（添加更多上下文）。from e保留了原始异常链，调试时能看到完整路径
else:
    # try没有抛出异常时，执行这里
    print(f"计算结果：{num}")
finally:
    # 无论是否发生异常，都会执行这里
    print("计算结束！")


# 自定义异常类，需要继承Exception，并且结尾需要是Error或Exception（约定）
class AgentConfigError(Exception):
    """Agent 配置错误"""
    pass


# 带数据的自定义异常
class ToolExecutionException(Exception):
    """工具执行失败"""

    def __init__(self, tool_name: str, params: dict, original_error: Exception):
        self.tool_name = tool_name
        self.params = params
        self.original_error = original_error
        super().__init__(
            f"工具：{self.tool_name}"
            f"参数：{self.params}"
            f"原因：{self.original_error}"
        )


# 使用
# try:
#     result = call_tools()
# except ToolExecutionException as e:
#     raise ToolExecutionException(
#         tool_name="call_tools",
#         params={"query": "test"},
#         original_error=e
#     )

# 最基础的重试
import httpx
import time
import random


def call_llm_api(messages: list[dict], max_retries: int = 3) -> dict:
    """带重试的http请求"""
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                url="http://ai-service.tal.com/openai-compatible/v1/chat/completions",
                headers={
                    "api-key": "300000476:a289413b081f5f6f0bd9b4e9f85f8205"
                },
                json={
                    "model": "deepseek-v4-flash",
                    "messages": messages,
                    "stream": False
                },
                timeout=30,
            )
            # 如果 HTTP 响应状态码是 4xx 或 5xx，则抛出异常；否则什么都不做
            response.raise_for_status()
            print(f"resp: {response}")
            return response.json()
        except httpx.TimeoutException as e:
            print(f"[重试 {attempt} / {max_retries}] 错误：{e}")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                # 限流
                retry_after = int(e.response.headers.get("Retry-After", 5))
                print(f"[重试 {attempt + 1} / {max_retries}] 限流，等待 {retry_after} s")
                time.sleep(retry_after)
            elif status == 500:
                # 500 服务端错误，可以重试
                print(f"[重试 {attempt + 1} / {max_retries}] 服务器错误 {status}")
                time.sleep(2 ** attempt)
            else:
                # 4xx，说明请求本身有问题，不应该重试
                print(f"请求参数错误（{status}），不以重试")
                break
    raise RuntimeError(f"API 调用失败，已重试：{max_retries} 次")


def call_llm_api_v2(messages: list[dict], max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0) -> dict:
    """带重试的http请求"""

    last_exception = None

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                url="http://ai-service.tal.com/openai-compatible/v1/chat/completions",
                headers={
                    "api-key": "300000476:a289413b081f5f6f0bd9b4e9f85f8205"
                },
                json={
                    "model": "deepseek-v4-flash",
                    "messages": messages,
                    "stream": False
                },
                timeout=30,
            )
            return response.json()

        except httpx.TimeoutException as e:
            last_exception = e
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay*0.5) # 加0~50%的随机
            total_wait = delay + jitter
            print(f"[Timeout] 第 {attempt+1} 次超时，{total_wait:.1f}s 后重试")
            time.sleep(total_wait)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                # 限流
                retry_after = int(e.response.headers.get("Retry-After", 5))
                print(f"[重试 {attempt + 1} / {max_retries}] 限流，等待 {retry_after} s")
                time.sleep(retry_after)
            elif status in (500, 502, 503, 504):
                # 500 服务端错误，可以重试
                if attempt == max_retries:
                    last_exception = e
                    break
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.5)  # 加0~50%的随机
                total_wait = delay + jitter
                print(f"[{status}] 服务器错误，{total_wait:.1f}s 后重试...")
                time.sleep(total_wait)

            else:
                # 4xx，说明请求本身有问题，不应该重试
                print(f"请求参数错误（{status}），不以重试")
                break
    raise RuntimeError(f"API 调用失败，已重试：{max_retries} 次") from last_exception

from functools import wraps

# 用装饰器包装
def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0, retryable_exceptions: tuple = (Exception,)):
    """
    通用的重试装饰器。
    用法：
        @retry_on_failure(max_retries=3, base_delay=1, max_delay=30, retryable_exceptions=(httpx.TimeoutException,))
        def call_api():...

    :param max_retries: 最大重试次数
    :param base_delay:  基础等待时间
    :param max_delay:   最大等待时间上限
    :param retryable_exceptions:    哪些异常可以重试
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries+1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)  # 加0~50%的随机
                    total_wait = delay + jitter
                    print(f"[Retry] {func.__name__} 第 {attempt+1} 次失败，{total_wait:.1f}s 后重试")
                    time.sleep(total_wait)
            raise RuntimeError(f"API 调用失败，已重试：{max_retries} 次，最后一次的错误是：{last_exception}") from last_exception
        return wrapper
    return decorator

# 使用装饰器后的API调用给你
@retry_on_failure(max_retries=3, base_delay=1.0, max_delay=30.0, retryable_exceptions=(httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError))
def call_llm_api_simple(messages: list[dict]) -> dict:
    response = httpx.post(
        url="http://ai-service.tal.com/openai-compatible/v1/chat/completions",
        headers={
            "api-key": "300000476:a289413b081f5f6f0bd9b4e9f85f8205"
        },
        json={
            "model": "deepseek-v4-flash",
            "messages": messages,
            "stream": False
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


try:
    result = call_llm_api_v2([
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "番茄炒蛋如何制作？精简回答"}
    ], 3)
    print(f"result: {result['choices'][0]['message']['content']}")

    result2 = call_llm_api_simple([
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "番茄炒蛋如何制作？精简回答"}
    ])
    print(f"result2: {result['choices'][0]['message']['content']}")
except Exception as e:
    print(f"执行异常：{e}")

# 常见重定向组合
# # 只重定向 stdout（默认）
# command > output.log           # 等价于 command 1>output.log
#
# # 只重定向 stderr
# command 2> error.log           # 错误写文件，正常输出还在终端
#
# # stdout 和 stderr 分开写
# command > out.log 2> err.log   # 各写各的
#
# # 合并到一个文件
# command > all.log 2>&1         # 先让 stdout 指向文件，再把 stderr 指向 stdout
# command &> all.log             # bash/zsh 简写，同上
#
# # 追加模式
# command >> out.log 2>> err.log
# command &>> all.log            # 合并追加
#
# # 丢弃输出
# command > /dev/null 2>&1       # 啥都不要