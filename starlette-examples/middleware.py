from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware


async def service_info(request):
    print(request.headers)
    return JSONResponse({
        "version": "1.0",
    })


class CustomHeaderMiddleware(BaseHTTPMiddleware):
    # Adapted from https://github.com/fastapi/fastapi/issues/3027

    async def __call__(self, scope, receive, send):
        scope["headers"].append((b"my-header", b"12345"))
        await super().__call__(scope, receive, send)

    async def dispatch(self, request, call_next):
        return await call_next(request)


routes = [
    Route("/service-info", service_info),
]

middleware = [
    Middleware(CustomHeaderMiddleware),
]

app = Starlette(routes=routes, middleware=middleware)