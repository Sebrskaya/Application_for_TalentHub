from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from core.database import get_session
from schemas import UserLogin, UserResponse, UserEdit, ListUserResponse
from dependencies import get_current_admin_user
from services.db_service import DbService
from models import User
from services.auth_service import AuthService
from services.users_service import UsersService

router = APIRouter()

# Чтение всех пользователей
@router.get("/", response_model=ListUserResponse)
def read_all_users(current_user: Annotated[User, Depends(get_current_admin_user)],
              session: Annotated[Session, Depends(get_session)]):
    users = DbService(session).select_table_rows(User)
    if users is None:
        raise HTTPException(status_code=404, detail="Users not found")
    
    return {"users": users}

# Чтение пользователя по id
@router.get("/{user_id}", response_model=UserResponse)
def read_user(user_id: int, 
              current_user: Annotated[User, Depends(get_current_admin_user)],
              session: Annotated[Session, Depends(get_session)]):
    user = DbService(session).select_table_rows(User, **{"id": user_id})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user[0]



# Добавление пользователя
@router.post("/")
def add_user(user_data: UserLogin,
          current_user: Annotated[User, Depends(get_current_admin_user)],
          session: Session = Depends(get_session)):
    try:
        user = AuthService(session).register_user(user_data)
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Изменение пользователя
@router.patch("/{user_id}")
def edit_user(user_id: int, 
              user_data: UserEdit,
              current_user: Annotated[User, Depends(get_current_admin_user)],
              session: Session = Depends(get_session)):
    try:
        return UsersService(session).edit_user(user_id, user_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Удаление пользователя и связанных с ним атрибутов (видео, аналитика и т.д.)
@router.delete("/{user_id}")
def delete_user(user_id: int,
                current_user: Annotated[User, Depends(get_current_admin_user)],
                session: Session = Depends(get_session)):
    try:
        return UsersService(session).delete_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))