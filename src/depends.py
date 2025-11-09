from fastapi import Request


async def get_redis_session(request: Request):
    """Get Redis session from app state via request object."""
    yield request.app.state.redis
