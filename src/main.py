try:
    import orjson
    from fastapi.responses import ORJSONResponse  # noqa

    RESPONSE_CLASS = ORJSONResponse

except ImportError:
    import json
    from fastapi.responses import JSONResponse

    RESPONSE_CLASS = JSONResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from config.swagger import swagger_params
from lifespan import lifespan
from logger_config import setup_root_logger, get_logger
from routers import cache, models

# Setup root logger for the application
setup_root_logger(level=settings.log_level)

# Get logger for this module
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    default_response_class=RESPONSE_CLASS,
    swagger_ui_parameters=swagger_params,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with api_prefix from settings
app.include_router(cache.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
# app.include_router(ws.router_ws, prefix=settings.api_prefix)
