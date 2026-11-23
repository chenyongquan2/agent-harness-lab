"""
Web界面路由

处理前端页面渲染和聊天功能
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from api.chat_handler import chat_stream, resolve_session_id
from executor.local import FAILED_REPLY
from harness.runtime.agent_loop import DELTA_PREFIX, TOOL_PREFIX
import json
import logging

# 创建logger实例
logger = logging.getLogger(__name__)
# 模板配置
templates = Jinja2Templates(directory="web/templates")

# Web路由器
router = APIRouter(tags=["Web界面"])

class ChatRequest(BaseModel):
    message: str
    state: str | None = None
    session_id: str | None = None  # 缺省时服务端生成并经 X-Session-Id 回传

@router.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    """渲染主页聊天界面"""
    return templates.TemplateResponse("index.html", {"request": request})

# ── /chat/stream 的 SSE 协议（change: web-sse-streaming）───────────────────────
# 每个事件是「可选的 event 行 + 一行 data（JSON）+ 空行」，前端按空行切分：
#   data: {"delta": "..."}                         最终回复的一段文字，收到就追加显示
#   event: tool / data: {"name": ..., "label": ...} 本轮在调工具，只给提示文案，不给参数
#   event: done / data: {"reply": "..."}            结束；reply 是写进历史的完整回复，
#                                                   前端用它替换流式拼接出的文字（排队满、
#                                                   超时、护栏拦停的兜底文案只在这里出现）
#   event: error / data: {"message": "..."}         服务端异常，此后不再有事件
_REPLY_PREFIX = "[REPLY]"

# 工具名对应的提示文案。主 Agent 在预约域只调 delegate，所以按派给哪个子 Agent 细分。
# 查不到的工具一律用缺省文案——宁可笼统，也不把工具参数原样推给用户。
_DELEGATE_LABELS = {
    "appointment": "正在查档期、处理预约…",
    "consultant": "正在查服务和价格信息…",
    "user_behavior": "正在查看你的历史偏好…",
}
_DEFAULT_TOOL_LABEL = "正在处理…"


def _sse(data: dict, event: str | None = None) -> str:
    """把一个事件编码成 SSE 文本块。JSON 不含裸换行，所以一行 data 就够。"""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _tool_event(payload: str) -> dict:
    """从 [TOOL] 的 JSON 里只取工具名和提示文案；参数（可能含手机号等）不外推。"""
    try:
        info = json.loads(payload)
    except ValueError:
        info = {}
    name = info.get("name", "")
    args = info.get("args") or {}
    if name == "delegate":
        label = _DELEGATE_LABELS.get(args.get("subagent"), _DEFAULT_TOOL_LABEL)
    else:
        label = _DEFAULT_TOOL_LABEL
    return {"name": name, "label": label}


@router.post("/chat/stream", summary="流式聊天（SSE）")
async def chat_stream_endpoint(chat: ChatRequest, request: Request):
    """逐 token 的 SSE 流式聊天（按 session_id 隔离会话；响应头回传 X-Session-Id）。

    仍经 executor 的同步内联模式执行（`chat_stream` → `execute_inline`），与飞书路径共用
    同一套并发限制（同会话串行锁、全局并发上限）；只是让 AgentLoop 走流式模式，把内部 token 转成 SSE 事件。

    客户端断开时停止生成，靠两层：
    - 每推一个事件前查一次 `request.is_disconnected()`，断了就退出；
    - 长时间没有事件可推（工具在跑、模型在想）时，Starlette 收到断开消息会取消本生成器。
    两条路都会走到 finally 里的 `aclose()`，一路关到 AgentLoop 底层的模型 HTTP 流，模型侧
    随之停止生成；编排层同时补写兜底 assistant 回合，会话历史不会留下孤立的 user 回合。
    """
    session_id = resolve_session_id(chat.session_id)

    async def event_generator():
        tokens = chat_stream(chat.message, session_id=session_id, stream=True)
        reply = ""
        try:
            async for token in tokens:
                if await request.is_disconnected():
                    logger.info("客户端已断开，停止生成", extra={"session_id": session_id})
                    return
                if token.startswith(DELTA_PREFIX):
                    yield _sse({"delta": token[len(DELTA_PREFIX):]})
                elif token.startswith(TOOL_PREFIX):
                    yield _sse(_tool_event(token[len(TOOL_PREFIX):]), event="tool")
                elif token.startswith(_REPLY_PREFIX):
                    reply = token[len(_REPLY_PREFIX):]  # 多条时以最后一条为准，与编排层一致
            yield _sse({"reply": reply}, event="done")
        except Exception:  # noqa: BLE001 —— 流已开始，HTTP 状态码改不了，只能用事件告诉前端
            logger.exception("流式聊天失败", extra={"session_id": session_id})
            yield _sse({"message": FAILED_REPLY}, event="error")
        finally:
            await tokens.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
            # 关掉 Nginx 的响应缓冲，否则反向代理会攒满一块再发，前端看到的还是一整段。
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/chat", summary="兼容性聊天接口")
async def chat_endpoint(chat: ChatRequest):
    """兼容性聊天接口，建议使用/chat/stream"""
    session_id = resolve_session_id(chat.session_id)

    async def token_generator():
        async for token in chat_stream(chat.message, session_id=session_id):
            yield token
    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )

@router.get("/technician", response_class=HTMLResponse, summary="技师状态页面")
async def technician_page(request: Request):
    """技师状态页面"""
    # 通过API层获取技师数据
    try:
        from api.technician import get_all_technicians
        
        # 调用API层函数获取数据
        technicians = await get_all_technicians()
        
        return templates.TemplateResponse("technician.html", {
            "request": request,
            "technicians": technicians
        })
    except Exception as e:
        return templates.TemplateResponse("technician.html", {
            "request": request,
            "technicians": [],
            "error": str(e)
        })

@router.get("/technician_schedule", response_class=HTMLResponse, summary="技师排班页面")
async def technician_schedule_page(request: Request):
    """技师排班页面"""
    try:
        from api.technician import get_all_technicians_schedule_today
        from config.time_config import time_config
        
        # 获取当前日期
        current_date = time_config.current_date_str()
        
        # 通过API层获取所有技师的排班数据
        schedules_data = await get_all_technicians_schedule_today()
        
        # 构建排班数据格式 - 直接使用API返回的数据
        schedule = []
        for schedule_item in schedules_data:
            schedule.append({
                "id": schedule_item["technician_id"],
                "name": schedule_item["technician_name"],
                "busy_periods": schedule_item["busy_periods"]
            })
        
        return templates.TemplateResponse("technician_schedule.html", {
            "request": request,
            "schedule": schedule,
            "current_date": current_date
        })
    except Exception as e:
        logger.error(f"加载技师排班数据失败: {str(e)}")
        return templates.TemplateResponse("technician_schedule.html", {
            "request": request,
            "schedule": [],
            "error": str(e)
        })

@router.get("/admin", response_class=HTMLResponse, summary="系统管理页面")
async def admin_dashboard(request: Request):
    """系统管理仪表板"""
    try:
        # 通过API层获取系统状态信息
        from api.technician import get_all_technicians

        # 获取技师数据
        technicians = await get_all_technicians()

        # 数据库信息（知识库已随本地 RAG 移除，见 change: remove-local-rag）
        db_info = {
            "technicians_count": len(technicians),
        }
        
        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "db_info": db_info,
            "technicians": technicians[:5]  # 只显示前5个技师
        })
    except Exception as e:
        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "db_info": {},
            "technicians": [],
            "error": str(e)
        })

@router.get("/admin/database", response_class=HTMLResponse, summary="数据库管理页面")
async def database_admin_page(request: Request):
    """数据库管理页面"""
    try:
        # 通过API层获取数据库统计信息
        from api.technician import get_all_technicians

        # 获取技师数据
        technicians = await get_all_technicians()

        stats = {
            "technicians": len(technicians),
            "appointments": 0  # TODO: 通过API获取预约数量
        }
        
        return templates.TemplateResponse("database_admin.html", {
            "request": request,
            "stats": stats
        })
    except Exception as e:
        return templates.TemplateResponse("database_admin.html", {
            "request": request,
            "stats": {},
            "error": str(e)
        })
