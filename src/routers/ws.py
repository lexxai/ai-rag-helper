from asyncio import sleep

from fastapi import APIRouter
from starlette.websockets import WebSocket

from logger_config import get_logger

logger = get_logger(__name__)

router_ws = APIRouter(prefix="/ws", tags=["WebSocket"])


# WebSocket dashboard
@router_ws.websocket("/")
async def ws_models(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await sleep(10)
    except:
        ...
    finally:

        await ws.close()
