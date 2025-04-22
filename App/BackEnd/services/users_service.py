import os
from sqlalchemy.orm import Session
from dependencies import get_current_admin_user
from core.database import get_session
from services.db_service import DbService
from models import User, Video, VideoAnalytics, AnalyticsReport
from schemas import UserEdit
from services.auth_service import AuthService


class UsersService:
    def __init__(self, session: Session = None):
        if session is not None:
            self.session = session

    def edit_user(self, user_id: int, user_data: UserEdit):
        try:
            # Т.к. в бд поле называется password_hash, 
            # то надо передать словарь с ключом не 'password', а 'passsword_hash' 
            user_dict = user_data.model_dump(exclude_none=True)
            if user_data.password is not None:
                user_dict["password_hash"] = AuthService().get_password_hash(user_dict.pop("password"))

            updated_user = DbService(self.session).update_row_in_table(User, {"id": user_id}, user_dict)
            return updated_user
        except Exception as e:
            raise Exception(f"Ошибка при редактировании пользователя: {str(e)}")

    def delete_user(self, user_id: int):
        db_service = DbService(self.session)

        try:
            # Получение всех видео пользователя
            videos = db_service.select_table_rows(Video, user_id=user_id)
            
            # Добавляем проверку на None
            if videos is None:
                videos = []  # Если видео нет, создаем пустой список
            
            for video in videos:
                # Удаление связанных аналитик
                analytics_list = db_service.select_table_rows(VideoAnalytics, video_id=video.id) or []
                
                for analytics in analytics_list:
                    # Удаление отчетов
                    reports = db_service.select_table_rows(AnalyticsReport, video_analytics_id=analytics.id) or []
                    
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

            # Удаление пользователя
            db_service.delete(User, id=user_id)

            return {"message": f"Пользователь с ID {user_id} и все связанные ресурсы успешно удалены."}

        except Exception as e:
            self.session.rollback()
            raise Exception(f"Ошибка при удалении пользователя: {str(e)}")
