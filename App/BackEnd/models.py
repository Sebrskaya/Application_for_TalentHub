from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase): pass

# Определение модели ролей
class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    rolename = Column(String(45), unique=True, nullable=False)

# Определение модели пользователя
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)  # Добавьте autoincrement
    username = Column(String(45), unique=True, index=True, nullable=False)
    password_hash = Column(String(100), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)

# Логи действий пользователя
class ActionLog(Base):
    __tablename__ = "action_logs"
    id = Column(Integer, primary_key=True, index=True)
    datetime = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action_log_type = Column(Integer, ForeignKey("action_log_types.id"), nullable=False)

# Типы действий пользователя
class ActionLogType(Base):
    __tablename__ = "action_log_types"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(45), nullable=False)

# Определение модели видео
class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(255), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

# Определение модели аналитики видео
class VideoAnalytics(Base):
    __tablename__ = "video_analytics"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    anomaly_count = Column(Integer, nullable=False)
    frequency = Column(Float, nullable=False)
    size = Column(Integer)
    intensity = Column(Float, nullable=False)
    avg_distribution = Column(String(45))

# Определение модели отчета аналитики
class AnalyticsReport(Base):
    __tablename__ = "analytics_reports"
    id = Column(Integer, primary_key=True, index=True)
    video_analytics_id = Column(Integer, ForeignKey("video_analytics.id"))
    file_path = Column(String(255))