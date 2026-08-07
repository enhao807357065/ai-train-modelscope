import uvicorn
import time

from fastapi import FastAPI, Query, APIRouter, Header, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from py_pydantic import MessageResponse

# uvicorn py.py_fastapi:app --port 9999 --workers 4
#           模块名        实例名

app = FastAPI()

# 也支持多个路由拆分模块
router = APIRouter(prefix="/hello", tags=["hello"])

class HelloRequest(BaseModel):
    name: str
    times: int = 1

# 也支持路径参数
@router.post("/hello/{agent_id}")
async def hello(req: HelloRequest, agent_id: str):
    return {
        "message": f"hello: {req.name}"
    }

# get 如果不给默认值，则说明是必填参数。也可以使用Query添加约束，类似于Pydantic
@router.get("/hello_query")
async def hello_query(
        name: str,
        times: int = 1,
        page: int = Query(default=1, ge=1, le=20, description="页数"),
        page_size: int = Query(default=20, description="页数")
):
    return f"你好：{name}"

# uvicorn.run(app, port=9999)

router_task = APIRouter(prefix="/task", tags=["task"])

@router_task.get("/tasks")
async def get_tasks(name: str = "test"):
    return f"task is {name}"

app.include_router(router)
app.include_router(router_task)

# 依赖注入Depends，抽成可复用的处理逻辑
# 三种常见用途：1.认证：类似中间件 2.数据库链接：封装公共的获取数据库链接对象 3.配置/设置：只读一次配置文件
# Depends 也支持嵌套
async def verify_api_key(api_key: str = Header(...)):
    """验证API KEY，验证通过返回它"""
    valid_keys = {"sk-prod", "sk-test"}
    if api_key not in valid_keys:
        raise HTTPException("无效的apikey")
    return api_key

agent_router = APIRouter(prefix="/agent", tags=["agent"])

# 也可以用model_response限定返回值
@agent_router.post("/list", response_model=MessageResponse)
async def get_agent_list(api_key: str = Depends(verify_api_key)):
    # return MessageResponse()
    pass


# 这个是FastAPI的中间件
# 适合处理日志、CORS、性能监控等
@app.middleware("http")                                         # 注册为http的中间件，拦截所有请求
async def add_process_time_header(req: Request, call_next):
    """给每个响应加一个X-Process-Time头"""
    start = time.time()                                         # 请求进来时记录事件
    response = await call_next(req)                 # 放行，等待业务逻辑执行完
    response.headers["X-Process-Time"] = "test"     # 拿到响应后，往头部写数据
    return response                                 # 返回修改后的响应

# 异常处理
# 1.在路由中抛出HTTPException
@app.post("/test_exp")
async def exp_test(id: str = ""):
    if id == "":
        raise HTTPException(
            status_code=404,
            detail="说明文本",
            headers={"X-Error-Code": "NOT FOUND"}
        )
    return "success"