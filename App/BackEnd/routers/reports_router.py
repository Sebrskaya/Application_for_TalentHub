from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from typing import Annotated
from sqlalchemy.orm import Session
from core.database import get_session
from dependencies import get_current_admin_user, get_current_user
from models import User, ActionLog, AnalyticsReport
from services.report_service import ReportService
import os
from services.db_service import DbService
from datetime import datetime, timezone

router = APIRouter()

# Получение отчёта по пользователям
@router.get("/users")
def get_users_report(current_user: Annotated[User, Depends(get_current_admin_user)],
                     session: Annotated[Session, Depends(get_session)], 
                     format: str):
    try:
        # Генерация отчета
        file_path = ReportService(session).generate_users_report(format)
        # Логирование
        log_data = {"datetime": datetime.now(timezone.utc), "user_id": current_user.id, "action_log_type": 3}
        DbService(session).insert_row_in_table(ActionLog, log_data)
        # Проверка, существует ли файл
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Возвращениие файл для скачивания
        return FileResponse(file_path, media_type="application/octet-stream", filename=os.path.basename(file_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Получение отчёта по логам всех пользователей
@router.get("/users-logs")
def get_all_user_logs_report(current_user: Annotated[User, Depends(get_current_admin_user)],
                              session: Annotated[Session, Depends(get_session)],
                              format: str):
    try:
        # Генерация отчета по всем логам
        file_path = ReportService(session).generate_all_user_logs_report(format)
        print("get_all_user_logs_report")
        # Логирование
        log_data = {"datetime": datetime.now(timezone.utc), "user_id": current_user.id, "action_log_type": 3}
        DbService(session).insert_row_in_table(ActionLog, log_data)
        print("log_data get_all_user_logs_report")
        # Проверка, существует ли файл
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        print("file_path get_all_user_logs_report")
        # Возвращаем файл для скачивания
        return FileResponse(file_path, media_type="application/octet-stream", filename=os.path.basename(file_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Получение отчёта по логам конкретного пользователя
@router.get("/users-logs/{user_id}")
def get_user_logs_report(user_id: int,
                         current_user: Annotated[User, Depends(get_current_admin_user)],
                         session: Annotated[Session, Depends(get_session)],
                         format: str):
    try:
        # Генерация отчета по логам конкретного пользователя
        file_path = ReportService(session).generate_user_logs_report(user_id, format)
        # Логирование
        log_data = {"datetime": datetime.now(timezone.utc), "user_id": current_user.id, "action_log_type": 3}
        DbService(session).insert_row_in_table(ActionLog, log_data)
        # Проверка, существует ли файл
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Возвращаем файл для скачивания
        return FileResponse(file_path, media_type="application/octet-stream", filename=os.path.basename(file_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Получение отчёта по аналитике сайта за всё время
@router.get("/website")
def get_website_analytics_report(current_user: Annotated[User, Depends(get_current_admin_user)],
                                 session: Annotated[Session, Depends(get_session)],
                                 format: str = "xlsx"):
    try:
        # Генерация отчета по общей аналитике сайта
        file_path = ReportService(session).generate_website_analytics_report(format)
        # Логирование
        log_data = {"datetime": datetime.now(timezone.utc), "user_id": current_user.id, "action_log_type": 3}
        DbService(session).insert_row_in_table(ActionLog, log_data)
        # Проверка, существует ли файл
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Возвращаем файл для скачивания
        return FileResponse(file_path, media_type="application/octet-stream", filename=os.path.basename(file_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Получение отчёта по видео аналитике
@router.get("/video-analytics/{report_id}")
def get_video_analytics_report(report_id: int,
                               current_user: Annotated[User, Depends(get_current_user)],
                               session: Annotated[Session, Depends(get_session)]):
    try:
        # Получаем отчеты из базы данных
        reports = DbService(session).select_table_rows(AnalyticsReport, **{"id": report_id})
        if not reports:
            raise HTTPException(status_code=404, detail="Отчет не найден")

        # Извлекаем первый отчет из списка
        report = reports[0]
        file_path = report.file_path

        # Проверка, что file_path — строка
        if not isinstance(file_path, (str, bytes, os.PathLike)):
            raise HTTPException(status_code=500, detail="Некорректный тип file_path")

        # Проверка, существует ли файл
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Файл отчета не найден")

        # Логирование
        log_data = {"datetime": datetime.now(timezone.utc), "user_id": current_user.id, "action_log_type": 3}
        DbService(session).insert_row_in_table(ActionLog, log_data)

        # Возвращаем файл для скачивания
        return FileResponse(file_path, media_type="application/octet-stream", filename=os.path.basename(file_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))