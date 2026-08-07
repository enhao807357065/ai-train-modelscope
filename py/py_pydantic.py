from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=20, description="用户名字")
    email: str
    is_active: bool = True
    age: Optional[int | None] = None
    prompt: str = Field(default="hi", description="提示词信息")
    # 错误做法，所有新实例将会共享同一个list
    # items: list[str] = []
    # 正确做法：default_factory
    # 如果默认值是可变的（list、dict、set、自定义对象），永远用default_factory
    # 当默认值需要"每次新建一个"而不是"大家共用一个"时，用 default_factory
    items: list[str] = Field(default_factory=list)

    # Field 的约束不够用？也提供了field_validator-单个字段的验证
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("name不能为空")
        return v.strip()

    # 当需要验证多个字段的时候，可以用model_validator
    # before：在Pydantic字段校验之前运行（收到的还是原始数据，cls+dict，可能不一定是dict）
    # after：在Pydantic字段校验之后运行（收到的已经是类型安全的模型，完整的self）
    # wrap：包裹整个验证过程（原始输入 + handler 函数，cls+data+handler），条件性跳过验证、缓存、自定义错误处理
    @model_validator(mode="after")
    def check_params(self):
        if self.name == "" or self.prompt == "":
            raise ValueError("名字和提示词不能为空！")
        return self

    model_config = ConfigDict(
        extra="forbid"  # 是否允许额外字段，forbid=拒绝（多字段会报错，这样也最安全），allow=允许（灵活但不推荐），ignore=忽略（多余字段静默忽略，兼容旧数据）
    )

user = User(id=1, name="张飞", email="test@qq.com")
print(f"user: {user}")

# 序列化与反序列化
# 模型 -> 字典
data = user.model_dump()
print(f"data: {data}")
# 模型 -> json
data_v1 = user.model_dump_json()
print(f"data_v1: {data_v1}")
# 完整的信息，包含description、title这些内容
# 意义在于FC的工具定义 -- FC就是用JSON Schema描述每个工具的参数形式
# FastAPI 自动文档
data_v2 = user.model_json_schema()
print(f"data_v2: {data_v2}")
# 字典 -> 模型
user1 = User.model_validate(data)
print(f"user1: {user1}")
# json -> 模型
user2 = User.model_validate_json(data_v1)
print(f"user2: {user2}")

# 实战
from enum import Enum

# 枚举类型
class AgentStatus(str, Enum):
    """Agent运行状态"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    ERROR = "error"

class MessageRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"

class CreateSessionRequest(BaseModel):
    """创建Agent对话请求"""
    model_config = ConfigDict(extra="forbid")

    user_message: str = Field(..., min_length=1, max_length=10000, description="用户的初始消息")
    agent_name: str = Field(..., min_length=1, max_length=50, description="Agent名字")
    model: str = Field(default="deepseek-v4-flash", description="使用的模型")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="创意温度")
    system_prompt: Optional[str] = Field(default=None, max_length=8000, description="自定义系统提示词")

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str):
        if v.strip() == "" or not v:
            raise ValueError(f"user_message 内容错误：{v}")
        return v

class SendMessageRequest(BaseModel):
    """发送消息到已有对话"""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="会话id")
    message: str = Field(..., description="消息内容")

class MessageResponse(BaseModel):
    """单条消息响应"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ToolCallResponse(BaseModel):
    """工具调用记录"""
    tool_name: str
    params: dict = Field(default_factory=dict)
    result: Optional[str] = None
    success: bool = False
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionResponse(BaseModel):
    """会话详情响应"""
    session_id: str
    agent_name: str
    model: str
    status: AgentStatus
    messages: list[MessageResponse] = Field(default_factory=list)
    tool_calls: list[ToolCallResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime = Field(default_factory=datetime.now)
    token_usage: int = Field(default=0, description="token累计消耗")

class SessionListResponse(BaseModel):
    """会话列表响应"""
    total: int
    sessions: list[SessionResponse]
    page: int = 1
    page_size: int = 10

class ErrorResponse(BaseModel):
    """统一错误响应"""
    error_code: str
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)