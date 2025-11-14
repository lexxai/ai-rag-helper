from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette import status
from starlette.requests import Request

try:
    import orjson  # noqa F401
    from fastapi.responses import ORJSONResponse  # noqa F401

    RESPONSE_CLASS = ORJSONResponse

except ImportError:
    import json  # noqa F401
    from fastapi.responses import JSONResponse  # noqa F401

    RESPONSE_CLASS = JSONResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from config.swagger import swagger_params
from lifespan import lifespan
from logger_config import setup_root_logger, get_logger
from routers import cache, models, embedding, rerank

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


class StarletteHTTPException:
    pass


# Unified handler (works for all three types)
async def unified_validation_handler(request: Request, exc):
    if isinstance(exc, RequestValidationError):
        details = [
            {
                "field": " → ".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
                "input": err.get("input"),
            }
            for err in exc.errors()
        ]
        message = "Validation failed"
    else:
        details = None
        message = str(exc) or "Invalid request"
    logger.error(f"{message}: {details}")
    return RESPONSE_CLASS(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": message,
            "details": details,
        },
    )


# Register each exception individually
app.add_exception_handler(RequestValidationError, unified_validation_handler)
app.add_exception_handler(ValueError, unified_validation_handler)
app.add_exception_handler(ValidationError, unified_validation_handler)


# Include routers with api_prefix from settings
app.include_router(cache.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
app.include_router(embedding.router, prefix=settings.api_prefix)
app.include_router(rerank.router, prefix=settings.api_prefix)
