import asyncio
import math
import random
import time


# async def：定义一个协程函数，调用它不会立即执行，而是返回一个协程对象
# await：暂停当前协程，等待另一个协程完成。在等待期间，事件循环可以去执行其他协程
async def boil_water():
    print("开始烧水...")
    await asyncio.sleep(5)
    print("水烧好了")

async def wash_cup():
    print("开始洗茶杯...")
    await asyncio.sleep(1)
    print("茶杯洗好了")

async def add_tea():
    print("开始加茶叶...")
    await asyncio.sleep(0.5)
    print("茶叶放好了")

async def main():
    print("===== 异步测试 ======")
    start = time.time()
    await asyncio.gather(
        boil_water(),
        wash_cup(),
        add_tea()
    )
    print(f"总耗时: {time.time()-start:.2f}")

asyncio.run(main())

async def sum(a: int, b: int) -> int:
    return a+b

# 协程的三种运行方式：
# 1. asyncio.run() 程序的入口点
sum_result = asyncio.run(sum(3,6))
print(f"sum_result: {sum_result}")
# 2. await 在已有的协程中调用另一个协程
async def main_v2():
    result = await sum(1,3)
    print(f"result: {result}")
asyncio.run(main_v2())
# 3. asyncio.gather
async def main_v3():
    result = await asyncio.gather(sum(1, 2), sum(2, 3))
    print(f"main_v3 result: {result}, type: {type(result)}")
asyncio.run(main_v3())

# 用 asyncio.create_task() 实现"发出但不等"
async def download_pdf(filename: str, duration: int):
    print(f"[开始] 下载 {filename}...")
    await asyncio.sleep(duration)
    print(f"[完成] 下载 {filename}...")
    return f"{filename}的内容"

async def down_v1():
    start = time.time()

    task_1 = asyncio.create_task(download_pdf("file1.pdf", 3))
    task_2 = asyncio.create_task(download_pdf("file2.pdf", 2))
    task_3 = asyncio.create_task(download_pdf("file3.pdf", 1))

    await asyncio.sleep(1)

    content_1 = await task_1
    content_2 = await task_2
    content_3 = await task_3

    print(f"content_1: {content_1}")
    print(f"content_2: {content_2}")
    print(f"content_3: {content_3}")

    print(f"全部完成，花费 {time.time() - start}")

asyncio.run(down_v1())

# gather() VS create_task() VS as_completed()
# 必须拿到所有结果才能继续：gather
# 希望有一个算一个，先完成的先处理：as_completed
# 希望发出任务后干点别的，稍后再收结果：create_task
async def asy_completed():
    task = [
        download_pdf("file1.pdf", 3),
        download_pdf("file2.pdf", 1),
        download_pdf("file3.pdf", 2),
    ]
    for sub in asyncio.as_completed(task, timeout=5):
        res = await sub
        print(f"asy_completed res: {res}")

asyncio.run(asy_completed())

# 注意：await不能用于普通函数
# 可以await：1.协程对象（async def函数的返回值），2.create_task后的task，3.Future对象，asyncio.Future

# 事件循环：挑一个协程运行，直到它遇到await -> 切换下一个协程 -> 循环往复
# 事件循环就是一个无限循环的调度器——不断地检查"谁准备好了"，然后执行它。


# async.wait_for(func(), timeout) 给协程加超时
#    async def local_wait_for():
#     result = await asyncio.wait_for(download_pdf("file_name1.pdf", 3), timeout=3)


# httpx 同步 VS 异步 方式
import httpx

# 同步方式
def sync_call_apis():
    urls = [
        "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
    ]
    start = time.time()
    with httpx.Client(timeout=10) as client:
        for url in urls:
            resp = client.get(url)
            print(f"resp: {resp}")
    cost = time.time() - start
    print(f"总耗时: {cost}")

async def async_call_apis():
    urls = [
        "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
        # "https://httpbin.org/delay/1",
    ]
    start = time.time()
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [client.get(url) for url in urls]
        # # ❌ 不能直接打印展开结果（必须在容器或函数调用里）
        #   print(*[1,2,3])     # ✅ 这个可以，因为 print 是函数调用
        #   x = *[1,2,3]        # ❌ 语法错误，必须配合赋值解包
        resp = await asyncio.gather(*tasks)
        print(f"resp: {resp}")
    cost = time.time() - start
    print(f"总耗时: {cost}")

# if __name__ == '__main__':
#     sync_call_apis()
# asyncio.run(async_call_apis())


# 实战，并发调用多次大模型，以最先返回的为准
async def call_model(model: str, client: httpx.AsyncClient, query: str, api_key: str):
    # response = httpx.post(
    #     url="http://ai-service.tal.com/openai-compatible/v1/chat/completions",
    #     headers={
    #         "api-key": "300000476:a289413b081f5f6f0bd9b4e9f85f8205"
    #     },
    #     json={
    #         "model": model,
    #         "messages": messages,
    #         "stream": False
    #     },
    #     timeout=30,
    # )
    start = time.time()
    response = await client.post(
        url="http://ai-service.tal.com/openai-compatible/v1/chat/completions",
        headers={"api-key": "300000476:a289413b081f5f6f0bd9b4e9f85f8205"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": query}]
        },
        timeout=30
    )
    cost = time.time() - start
    # 如果 HTTP 响应状态码是 4xx 或 5xx，则抛出异常；否则什么都不做
    if response.status_code == 200:
        data = response.json()
        return {
            "model": model,
            "content": data['choices'][0]['message']['content'],
            "cost": cost
        }
    else:
        return {"model": model, "error": response.status_code, "cost": cost}

async def call_multi_model():
    model_list = ["deepseek-v4-flash", "kimi-k2.6", "doubao-seed-1.8"]

    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [
            asyncio.create_task(call_model(model, client, "番茄炒蛋如何制作？", "300000476:a289413b081f5f6f0bd9b4e9f85f8205")) for model in model_list
        ]
        for sub in asyncio.as_completed(tasks):
            fast_result = await sub
            if "error" not in fast_result:
                print(f"最快的模型是：{fast_result['model']}，结果是：{fast_result['content']}，耗时：{fast_result['cost']}")
                # 取消其他正在跑的请求
                for task in tasks:
                    task.cancel()
                return fast_result
            else:
                print(f"执行出错：{fast_result['model']}，错误是：{fast_result['error']}")

        return {"error": "所有模型都执行失败了"}

asyncio.run(call_multi_model())

print(f"\n================================")

async def call_model_with_retry(model: str, client: httpx.AsyncClient, query: str, api_key: str, max_retries: int):
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.post(
                    url="http://ai-service.tal.com/openai-compatible/v1/chat/completions",
                    headers={"api-key": api_key},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": query}]
                    }
                ),
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except TimeoutError as te:
            if attempt == max_retries-1:
                raise TimeoutError(f"超时异常：{te}") from te
            wait = 2 ** attempt
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as hs:
            match hs.response.status_code:
                case 429:
                    print(f"限流，等待重试...")
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                case 504 | 503 | 502:
                    print(f"服务器错误，退出...")
                    raise
    return {"error": "全部调用失败"}

async def call_multi_model_with_retry():
    model_list = ["deepseek-v4-flash"]

    async with httpx.AsyncClient(timeout=60) as client:
        # 此处使用asyncio.create_task的作用是可以cancel
        tasks = [
            asyncio.create_task(call_model_with_retry(model, client, "番茄炒蛋如何制作？精简回答", "300000476:a289413b081f5f6f0bd9b4e9f85f8205", 3)) for model in model_list
        ]
        for sub in asyncio.as_completed(tasks):
            fast_result = await sub
            if "error" not in fast_result:
                print(f"结果是：{fast_result}")
                # 取消其他正在跑的请求
                for task in tasks:
                    task.cancel()
                return fast_result
            else:
                print(f"执行出错：{fast_result}")

        return {"error": "所有模型都执行失败了"}

asyncio.run(call_multi_model_with_retry())

# 信号量，限制并发数
async def call_api_limit(client: httpx.AsyncClient, sem: asyncio.Semaphore, item_id: int):
    async with sem:
        print(f"[{item_id}] 开始请求...")
        await asyncio.sleep(1)
        print(f"[{item_id}] 完成")
        return {"item_id": item_id, "status": "ok"}

async def batch_with_limit(total: int = 20, max_concurrent: int = 5):
    sem = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient(timeout=5) as client:
        tasks = [
            call_api_limit(client, sem, i) for i in range(total)
        ]
        result = await asyncio.gather(*tasks)
        print(f"result: {result}")
    return result

asyncio.run(batch_with_limit(20, 5))

async def sub_print_num(num: int, sem: asyncio.Semaphore):
    async with sem:
        await asyncio.sleep(random.randint(1, 5))
        print(f"result: {num}")
        return num

async def async_print_num(total: int, concurrent: int):
    sem = asyncio.Semaphore(concurrent)
    tasks = [
        sub_print_num(num, sem) for num in range(total)
    ]
    for sub_task in asyncio.as_completed(tasks):
        sub_task_result = await sub_task
        print(f"sub_task_result: {sub_task_result}")

asyncio.run(async_print_num(20, 5))