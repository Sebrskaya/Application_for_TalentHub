from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory='/app/frontend/templates')


@router.get('/login')
async def get_students_html(request: Request):
    return templates.TemplateResponse(name='login.html', context={'request': request})

@router.get('/register')
async def get_students_html(request: Request):
    return templates.TemplateResponse(name='register.html', context={'request': request})

@router.get('/upload')
async def get_students_html(request: Request):
    return templates.TemplateResponse(name='upload.html', context={'request': request})

@router.get('/analytics')
async def get_students_html(request: Request):
    return templates.TemplateResponse(name='analytics.html', context={'request': request})

@router.get('/admin')
async def get_students_html(request: Request):
    return templates.TemplateResponse(name='admin.html', context={'request': request})