from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Annotated
from models import User, Video
from core.database import get_session
from services.db_service import DbService
from schemas import UserResponse, VideoResponse, VideoAnalyticsResponse, ListVideoResponse, VideoAnalyticsReportResponse
from dependencies import get_current_user
from services.analytics_service import AnalyticsServis
from services.report_service import ReportService

router = APIRouter()

# Чтение всех видео, загруженных определенным пользователем
@router.get("/videos", response_model=ListVideoResponse)
def read_all_videos(current_user: Annotated[User, Depends(get_current_user)],
                    session: Annotated[Session, Depends(get_session)]):
    videos = DbService(session).select_table_rows(Video, **{"user_id": current_user.id})
    if videos is None:
        return {"message": "You don't have any videos yet"}
    return {"videos": videos}

# Чтение видео
@router.get("/videos/{video_id}", response_model=VideoResponse)
def read_video(video_id: int,
               current_user: Annotated[User, Depends(get_current_user)],
               session: Annotated[Session, Depends(get_session)]):
    video = DbService(session).select_table_rows(Video, **{"id": video_id})
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video[0]

# Получение аналитики по видео
@router.get("/videos/{video_id}/analytics", response_model=VideoAnalyticsResponse)
def get_video_analytics(video_id: int,
                        current_user: Annotated[User, Depends(get_current_user)],
                        session: Annotated[Session, Depends(get_session)]):
    try:
        analytics = AnalyticsServis(session).get_analytics_by_video_id(video_id)
        return analytics
    except ValueError as e:  # Если видео не найдено
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Получение отчета аналитики по видео
@router.get("/videos/{video_id}/analytics/report", response_model=VideoAnalyticsReportResponse)
def get_video_analytics_report(
    video_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    format: str = Query(..., description="Формат отчета (pdf или xlsx)")
):
    try:
        analytics = AnalyticsServis(session).get_analytics_by_video_id(video_id)
        report = ReportService(session).get_report_by_analytics_id(analytics.id, format)
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))