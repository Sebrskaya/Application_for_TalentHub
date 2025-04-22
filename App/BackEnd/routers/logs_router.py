from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from core.database import get_session
from dependencies import get_current_admin_user
from models import User
from services.db_service import DbService
from schemas import ListUserLogsResponse

router = APIRouter()

# Получение отчёта по логам всех пользователей
@router.get("/users", response_model=ListUserLogsResponse)
def get_all_user_logs(current_user: Annotated[User, Depends(get_current_admin_user)],
                              session: Annotated[Session, Depends(get_session)]):
    try:
        # Получение логов
        users_logs = DbService(session).select_user_logs_with_details()
        # Проверка, есть ли логи
        if users_logs is None:
            raise HTTPException(status_code=404, detail="Users logs not found")
        
        # Преобразование логов в список словарей
        data = []
        for log_tuple in users_logs:
            action_log, action_log_type, role, user = log_tuple

            # Формирование строк
            row = {
                "user_id": user.id,
                "datetime": action_log.datetime,
                "action_type": action_log_type.type,
                "username": user.username,
                "rolename": role.rolename,
            }
            data.append(row)
        return {"logs": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))