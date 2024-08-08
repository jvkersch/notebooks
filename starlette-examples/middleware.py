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


class CustomHeaderMiddleware:

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope["headers"].append((b"my-header", b"12345"))
        await self.app(scope, receive, send)


routes = [
    Route("/service-info", service_info),
]

middleware = [
    Middleware(CustomHeaderMiddleware),
]

app = Starlette(routes=routes, middleware=middleware)