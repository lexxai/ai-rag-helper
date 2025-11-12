import time

from starlette.websockets import WebSocket

from depends.depends import get_model_manager
from handlers.models import handler_unload_model, handler_load_model
from logger_config import get_logger
from model_manager import ModelManager
from routers.ws import router_ws

logger = get_logger(__name__)

# router = APIRouter(prefix="/ws/models", tags=["WebSocket Models"])


# WebSocket models
@router_ws.websocket("/models")
async def ws_models(ws: WebSocket):
    await ws.accept()
    manager = await get_model_manager(ws.app)
    queue = await manager.bus.subscribe()

    async def send_state():
        await ws.send_json({"event": "state", "time": time.time(), "models": await manager.list_loaded()})

    try:
        await send_state()
        while True:
            event = await queue.get()
            await send_state()
    except:
        pass
    finally:
        await manager.bus.unsubscribe(queue)
        await ws.close()


@router_ws.post("/load")
async def load_model(ws: WebSocket):
    await ws.accept()
    manager: ModelManager = await get_model_manager(ws.app)
    queue = await manager.bus.subscribe()
    data = await ws.receive()
    model_name = data.get("model_name")

    if not model_name:
        await ws.send_json(
            {"event": "state", "time": time.time(), "status": "error", "message": f"model_name undefined"}
        )
        await manager.bus.unsubscribe(queue)
        await ws.close()

    async def send_state():
        await handler_load_model(model_name, manager)
        await ws.send_json({"event": "state", "time": time.time(), "status": "ok", "message": f"Loaded {model_name}"})

    try:
        await send_state()
        while True:
            event = await queue.get()
            await send_state()
    except:
        pass
    finally:
        await manager.bus.unsubscribe(queue)
        await ws.close()


@router_ws.post("/unload")
async def load_model(ws: WebSocket):
    await ws.accept()
    manager: ModelManager = await get_model_manager(ws.app)
    queue = await manager.bus.subscribe()
    data = await ws.receive()
    model_name = data.get("model_name")

    if not model_name:
        await ws.send_json(
            {"event": "state", "time": time.time(), "status": "error", "message": f"model_name undefined"}
        )
        await manager.bus.unsubscribe(queue)
        await ws.close()

    async def send_state():
        await handler_unload_model(model_name, manager)
        await ws.send_json({"event": "state", "time": time.time(), "status": "ok", "message": f"UnLoaded {model_name}"})

    try:
        await send_state()
        while True:
            event = await queue.get()
            await send_state()
    except:
        pass
    finally:
        await manager.bus.unsubscribe(queue)
        await ws.close()
