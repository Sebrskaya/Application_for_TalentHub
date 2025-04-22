from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from core.database import get_session
from dependencies import get_current_admin_user
from models import User
from services.db_service import DbService
from schemas import ListWebsiteAnalyticsResponse

router = APIRouter()

# Получение отчёта по логам всех пользователей
@router.get("/analytics", response_model=ListWebsiteAnalyticsResponse)
def get_website_analytics(current_user: Annotated[User, Depends(get_current_admin_user)],
                              session: Annotated[Session, Depends(get_session)]):
    try:
        # Получение аналитики
        analytics = DbService(session).select_website_analytics_report()
        # Проверка, есть ли аналитика
        if analytics is None:
            raise HTTPException(status_code=404, detail="Website analytics not found")
        
        # Преобразование аналитики в список словарей
        data = []
        for analytic in analytics:
            row = {
                "datetime": analytic.date,
                "loginsCount": analytic.loginsCount,
                "uploadedCount": analytic.uploadedCount,
                "reportsCount": analytic.reportsCount
            }
            data.append(row)
        return {"analytics": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))