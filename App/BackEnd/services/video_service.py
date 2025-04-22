import os
from fastapi import UploadFile
import shutil
from services.db_service import DbService
from sqlalchemy.orm import Session
from models import Video, User, VideoAnalytics, AnalyticsReport, ActionLog
from datetime import datetime, timezone

UPLOAD_DIR = "/app/frontend/static/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class VideoService:
    def __init__(self, session: Session):
        self.session = session

    def upload_file(self, file: UploadFile, current_user: User):
        try:
            filename = file.filename
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            

            physical_path = os.path.join(UPLOAD_DIR, filename)
            with open(physical_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"[UPLOAD] Файл сохраняется в: {physical_path}")
            print(f"[UPLOAD] Существует ли папка? {os.path.exists(UPLOAD_DIR)}")

            relative_path = f"/static/videos/{filename}"

            video_data = {
                "file_path": relative_path,
                "user_id": current_user.id
            }

            result = DbService(self.session).insert_row_in_table(Video, video_data)

            # Лог
            log_data = {
                "datetime": datetime.now(timezone.utc),
                "user_id": current_user.id,
                "action_log_type": 2
            }
            DbService(self.session).insert_row_in_table(ActionLog, log_data)

            return result

        except Exception as e:
            raise Exception(f"[video_service] Ошибка при загрузке видео: {str(e)}")

    def delete_file(self, video_id: int):
        db_service = DbService(self.session)
        video = db_service.select_table_rows(Video, id=video_id)
        if video is None:
            raise ValueError("Видео не найдено")
        video = video[0]

        try:
            # Удаление связанных аналитик
            analytics_list = db_service.select_table_rows(VideoAnalytics, video_id=video.id)
            for analytics in analytics_list:
                # Удаление отчетов
                reports = db_service.select_table_rows(AnalyticsReport, video_analytics_id=analytics.id)
                for report in reports:
                    if os.path.isfile(report.file_path):
                        os.remove(report.file_path)
                    # Удаление записи об отчёте
                    db_service.delete(AnalyticsReport, id=report.id)
                # Удаление записи об аналитике
                db_service.delete(VideoAnalytics, id=analytics.id)

            # Удаление видеофайла
            if os.path.isfile(video.file_path):
                os.remove(video.file_path)

            # Удаление записи о видео
            db_service.delete(Video, id=video.id)

            return {"message": f"Видео с ID {video_id} и все связанные ресурсы успешно удалены."}
        except Exception as e:
            if not os.path.isfile(video.file_path):  # Если файлы удален, но в бд произошла ошибка
                raise Exception(f"Файлы удалён, но произошёл откат в базе данных: {str(e)}")
            raise Exception(f"Ошибка при удалении видео: {str(e)}")