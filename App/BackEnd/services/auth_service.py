from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from core.config import get_auth_data
import jwt
from services.db_service import DbService
from models import User, ActionLog
from schemas import UserLogin

class AuthService:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def __init__(self, session: Session = None):
        if session is not None:
            self.session = session

    # Проверка пароля
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    # Получить хэш пароля
    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    # Создание access токена
    def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=90)
        auth_data = get_auth_data()
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, auth_data["secret_key"], algorithm=auth_data["algorithm"])
        return encoded_jwt
    
    # Аутентификация пользователя
    def authenticate_user(self, user_data: UserLogin):
        users = DbService(self.session).select_table_rows(User, **{"username": user_data.username})
        if not users:
            return False
        user = users[0]
        if not self.verify_password(user_data.password, user.password_hash):
            return False
        
        # Логирование
        log_data = {"datetime": datetime.now(timezone.utc), "user_id": user.id, "action_log_type": 1}
        DbService(self.session).insert_row_in_table(ActionLog, log_data)
        return user
    
    # Регистрация пользователя
    def register_user(self, user_data: UserLogin):
        user = DbService(self.session).select_table_rows(User, **{"username": user_data.username})
        if user:
            raise Exception("Пользователь уже существует")
        
        user_dict = {}
        user_dict["username"] = user_data.username
        user_dict["password_hash"] = self.get_password_hash(user_data.password)
        user_dict["role_id"] = 2
        result = DbService(self.session).insert_row_in_table(User, user_dict)
        return result