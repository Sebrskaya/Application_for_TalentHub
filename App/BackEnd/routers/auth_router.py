from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from core.database import get_session
from schemas import UserLogin
from services.auth_service import AuthService
from dependencies import get_current_user
from models import User

router = APIRouter()

# Аутентификация с получением JWT токена в куки
@router.post("/login/")
def login(response: Response, user_data: UserLogin, session: Session = Depends(get_session)):
    user = AuthService(session).authenticate_user(user_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid login or password")
    
    access_token = AuthService.create_access_token(
        data={"sub": str(user.id)}
    )
    # Если сайт будет использовать https, то надо добавить еще: secure=True - 
    # если имеет значение True, то куки будут посылаться на сервер только в запросе по протоколу https
    response.set_cookie(key="users_access_token", value=access_token, httponly=True)
    return {'access_token': access_token, 'refresh_token': None}

# Выход из системы
@router.post("/logout/")
async def logout_user(response: Response):
    response.delete_cookie(key="users_access_token")
    return {'message': 'Successfully logged out!'}

# Регистрация
@router.post("/register/")
def register_user(user_data: UserLogin, session: Session = Depends(get_session)):
    try:
        AuthService(session).register_user(user_data)
        return {'message': 'Вы успешно зарегистрированы!'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Проверка пользователя
@router.get("/me/")

async def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role_id": current_user.role_id
    }