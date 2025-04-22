from pydantic import BaseModel
from typing import List
from datetime import datetime

# Схемы для регистрации и логина пользователя
class UserBase(BaseModel):
    username: str
    role_id: int

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    class Config:
        from_attributes = True
    password_hash: str


class ListUserResponse(BaseModel):
    users: List[UserResponse]     

class UserLogin(BaseModel):
    username: str
    password: str

class UserEdit(BaseModel):
    password: str | None = None
    role_id: int | None = None

# Схемы ролей
class RoleBase(BaseModel):
    rolename: str

class RoleResponse(RoleBase):
    id: int
    class Config:
        from_attributes = True

# Схемы видео
class VideoBase(BaseModel):
    file_path: str
    user_id: int

class VideoCreate(VideoBase):
    pass

class VideoResponse(VideoBase):
    id: int
    class Config:
        from_attributes = True

class ListVideoResponse(BaseModel):
    videos: List[VideoResponse] | None = None

# Схемы аналитики видео
class VideoAnalyticsBase(BaseModel):
    video_id: int
    anomaly_count: int
    frequency: int
    size: int
    intensity: int
    avg_distribution: str

class VideoAnalyticsResponse(VideoAnalyticsBase):
    id: int
    class Config:
        from_attributes = True

# Схемы отчетов аналитики видео
class VideoAnalyticsReportResponse(BaseModel):
    id: int
    video_analytics_id: int
    file_path: str


# Схемы для логов
class ActionLogTypeResponse(BaseModel):
    id: int
    type: str

class ActionLogResponse(BaseModel):
    user_id: int
    datetime: datetime
    action_type: str
    username: str
    rolename: str

class ListUserLogsResponse(BaseModel):
    logs: List[ActionLogResponse]


# Схемы для аналитики по сайту
class WebsiteAnalyticsResponse(BaseModel):
    datetime: datetime
    loginsCount: int
    uploadedCount: int
    reportsCount: int

class ListWebsiteAnalyticsResponse(BaseModel):
    analytics: List[WebsiteAnalyticsResponse]