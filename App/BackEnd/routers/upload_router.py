from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from dependencies import get_current_user
from typing import Annotated
from core.database import get_session
from sqlalchemy.orm import Session
from services.video_service import VideoService
from models import User
from schemas import VideoResponse

router = APIRouter()

@router.post("/", response_model=VideoResponse)
def upload_video(current_user: Annotated[User, Depends(get_current_user)],
                 session: Annotated[Session, Depends(get_session)],
                 file: Annotated[UploadFile, File(...)]):
    try:
        result = VideoService(session).upload_file(file, current_user)
        return result
    except ValueError as e:  # Если видео не найдено
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:  # Если нет доступа
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:  # Остальные ошибки
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
def delete_video(video_id: int,
                 current_user: Annotated[User, Depends(get_current_user)],
                 session: Annotated[Session, Depends(get_session)]):
    try:
        result = VideoService(session).delete_file(video_id, current_user)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))