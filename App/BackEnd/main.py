from dotenv import load_dotenv
import os


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from routers import analytics_router, auth_router, upload_router, users_router, reports_router, pages_router, logs_router, website_router
import uvicorn
from core.database import init_db



app = FastAPI(title="МонСвар API", description="API для анализа сварочного шва", version="1.0")

# Подключение маршрутов
app.include_router(pages_router.router, prefix='/pages', tags=['Фронтенд'])
app.include_router(auth_router.router, prefix="/auth", tags=["Авторизация"])
app.include_router(upload_router.router, prefix="/upload", tags=["Загрузка видео"])
app.include_router(analytics_router.router, prefix="/analytics", tags=["Аналитика"])
app.include_router(users_router.router, prefix="/users", tags=["Пользователи"])
app.include_router(reports_router.router, prefix="/reports", tags=["Отчеты"])
app.include_router(logs_router.router, prefix="/logs", tags=["Логи"])
app.include_router(website_router.router, prefix="/website", tags=["Сайт"])

app.mount('/static', StaticFiles(directory='/app/frontend/static'), 'static')

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API МонСвар!"}

# Запуск приложения
if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8081)