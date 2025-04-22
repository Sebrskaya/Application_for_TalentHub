from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import delete, insert, update
from sqlalchemy.exc import SQLAlchemyError
from models import ActionLog, ActionLogType, Role, User
from sqlalchemy import func


class DbService:
    def __init__(self, session: Session):
        self.session = session

    # Возращает все записи с указанной таблицы (можно указывать фильтры для таблицы - аналог where)
    # Фильтр работает только для проверки равенства значений полей
    def select_table_rows(self, model, **filter_by):
        query = select(model).filter_by(**filter_by)
        result = self.session.scalars(query).all()
        return result if result else None

    # Добавление записи в таблицу
    def insert_row_in_table(self, model, data: dict):
        try:
            query = insert(model).values(**data).returning(model)
            result = self.session.execute(query)
            self.session.commit()

            return result.scalars().first()  # Возвращение созданного объекта
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Ошибка при добавлении записи: {str(e)}")

    # Удаление всех записей в таблице (можно указывать фильтры для таблицы - аналог where)
    # Фильтр работает только для проверки равенства значений полей
    def delete(self, model, **filter_by):
        try:
            stmt = delete(model).filter_by(**filter_by)
            result = self.session.execute(stmt)
            # Фиксирование изменений
            self.session.commit()

            return {f"Удаленное количество строк: {result.rowcount}"}
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Ошибка при удалении записи: {str(e)}")

    # Обновление записи в таблице
    def update_row_in_table(self, model, filter_by: dict, values: dict):
        try:
            # Запрос на обновление с учетом фильтров
            stmt = (
                update(model)
                .where(*[getattr(model, key) == value for key, value in filter_by.items()])
                .values(**values)
                .returning(model)
            )
            result = self.session.execute(stmt)
            self.session.commit()

            return result.scalars().all()  # Возврат обновленных записей
        except SQLAlchemyError as e:
            self.session.rollback()
            raise RuntimeError(f"Ошибка при обновлении записи: {str(e)}")

    # Все логи действий пользователей с деталями: тип действия, роль и пользователь
    def select_user_logs_with_details(self, user_id: int = None):
        stmt = (
            select(ActionLog, ActionLogType, Role, User)
            .join(ActionLogType, ActionLog.action_log_type == ActionLogType.id)
            .join(User, ActionLog.user_id == User.id)
            .join(Role, User.role_id == Role.id)
            .order_by(ActionLog.datetime)
        )

        # Если передан user_id, производится фильтрацию по пользователю
        if user_id:
            stmt = stmt.filter(ActionLog.user_id == user_id)

        # Выполняем запрос
        logs = self.session.execute(stmt).fetchall()
        return logs if logs else None

    # Аналитика по сайту
    def select_website_analytics_report(self):
        stmt = (
            select(
                func.date(ActionLog.datetime).label('date'),  # Преобразование datetime в date
                func.count().filter(ActionLogType.type == "login").label("loginsCount"),
                func.count().filter(ActionLogType.type == "upload").label("uploadedCount"),
                func.count().filter(ActionLogType.type == "report").label("reportsCount")
            )
            .join(ActionLogType, ActionLog.action_log_type == ActionLogType.id)  # Объединение с типами действий
            .group_by(func.date(ActionLog.datetime))  # Группировка по дате
            .order_by(func.date(ActionLog.datetime))  # Сортировка по дате
        )

        result = self.session.execute(stmt).fetchall()
        return result if result else None

    # Пользователи с ролями
    def select_users_with_roles(self):
        stmt = (
            select(User, Role)
            .join(Role, User.role_id == Role.id)
            .order_by(User.id)
        )

        # Выполняем запрос и возвращаем кортежи (User, Role)
        result = self.session.execute(stmt).fetchall()
        return result if result else None