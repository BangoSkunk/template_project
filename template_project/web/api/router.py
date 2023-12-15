from fastapi.routing import APIRouter

from template_project.web.api import monitoring

api_router = APIRouter()
api_router.include_router(monitoring.router)
