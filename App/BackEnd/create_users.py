from sqlalchemy.orm import Session
from models import User
from core.database import get_session
from services.auth_service import AuthService

def create_users():
    db: Session = next(get_session())

    users = [
        {
            "id": "1",
            "username": "admin",
            "password": "123456",
            "role_id": "1",
        },
    ]

    err_count = 0
    for user in users:
        user_db = db.query(User).filter(User.username == user["username"]).first()
        if user_db:
            print(f"Пользователь с username {user['username']} уже существует!")
            err_count += 1
            continue

        hashed_password = AuthService(db).get_password_hash(user["password"])
        db_user = User(
            id=user["id"],
            username=user["username"],
            password_hash=hashed_password,
            role_id=user["role_id"],
        )
        db.add(db_user)

    if err_count == 0:
        db.commit()
        print("Пользователи добавлены!")

if __name__ == "__main__":
    create_users()