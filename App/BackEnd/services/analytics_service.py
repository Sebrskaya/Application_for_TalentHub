from services.db_service import DbService
from sqlalchemy.orm import Session
from models import VideoAnalytics, Video
from services.video_service import UPLOAD_DIR
import random
from datetime import datetime
import time

class AnalyticsServis:
    def __init__(self, session: Session):
        self.session = session

    def get_analytics_by_video_id(self, video_id: int):
        """Основной метод получения аналитики по видео"""
        # Проверка существования видео
        video = DbService(self.session).select_table_rows(Video, **{"id": video_id})
        if not video:
            raise ValueError("Видео не найдено")
        
        # Проверка существующей аналитики
        analytics = DbService(self.session).select_table_rows(VideoAnalytics, **{"video_id": video_id})
        if analytics:
            return analytics[0]
        
        # Если аналитики нет - делаем "запрос к нейронке"
        return self.make_analytics(video_id)

    def make_analytics(self, video_id: int):
        """Заглушка запроса к нейронке с фиктивными данными"""
        try:
            # Имитация запроса к нейронке (задержка 1-3 секунды)
            processing_time = random.uniform(1.0, 3.0)
            time.sleep(processing_time)
            
            
            # Фиктивные результаты "анализа" нейронки
            analytics_data = {
                "video_id": video_id,
                "anomaly_count": random.randint(1, 15),
                "frequency": random.randint(1, 5),
                "size": random.randint(5, 50),
                "intensity": random.randint(1, 10),
                "avg_distribution": random.choice(["равномерно", "локально", "кластерами"])
            }
            
            # Правильная вставка данных
            result = DbService(self.session).insert_row_in_table(
                model=VideoAnalytics,
                data=analytics_data  # Передаем весь словарь как один аргумент
            )
            
            # Логирование "успешного анализа"
            #print(f"Фейковый анализ завершен для видео {video_id}:")
            #print(f"Путь к видео: {video_path}")
            #print(f"Результаты: {analytics_data}")
            
            return result
            
        except Exception as e:
            raise Exception(f"Ошибка при анализе видео: {str(e)}")