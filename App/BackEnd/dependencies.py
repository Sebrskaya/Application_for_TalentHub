from fastapi import Request, HTTPException, status, Depends
from datetime import datetime, timezone
import jwt
from core.config import get_auth_data
from typing import Annotated
from jwt.exceptions import InvalidTokenError
from services.db_service import DbService
from models import User
from sqlalchemy.orm import Session
from core.database import get_session


# Получение токена из куки
def get_token(request: Request):
    token = request.cookies.get('users_access_token')
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail='Token not found')
    return token


# Получение текущего пользователя
def get_current_user(session: Annotated[Session, Depends(get_session)], token: Annotated[str, Depends(get_token)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Получение данных с токена
    try:
        auth_data = get_auth_data()
        payload = jwt.decode(token, auth_data["secret_key"], algorithms=[auth_data['algorithm']])
    except InvalidTokenError:
        raise credentials_exception

    # Проверка не истек ли токен
    expire = payload.get('exp')
    expire_time = datetime.fromtimestamp(int(expire), tz=timezone.utc)
    if (not expire) or (expire_time < datetime.now(timezone.utc)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token expired')

    # Получение id пользователя
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User id not found')

    # Получение пользователя
    user = DbService(session).select_table_rows(User, **{"id": int(user_id)})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found')

    return user[0]


# Проверка является ли текущий пользователь админом
async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role_id == 1:
        return current_user
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Недостаточно прав!')