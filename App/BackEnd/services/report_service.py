from services.db_service import DbService
import os
from sqlalchemy.orm import Session
from models import AnalyticsReport, VideoAnalytics, User
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


class ReportService:
    def __init__(self, session: Session):
        self.session = session

    # -----------------------------------------------------ДЛЯ АНАЛИТИКИ--------------------------------------------
    def get_report_by_analytics_id(self, analytics_id: int, format: str):
        reports = DbService(self.session).select_table_rows(AnalyticsReport, **{"video_analytics_id": analytics_id})
        # Поиск отчета с нужным расширением
        if reports is not None:
            for report in reports:
                # Извлекаем расширение без точки
                extension = os.path.splitext(report.file_path)[1].lower().strip('.')
                if extension == format.lower():
                    return report

        return self.generate_analytics_report(analytics_id, format)

    def generate_analytics_report(self, analytics_id: int, format: str):
        analytics = DbService(self.session).select_table_rows(VideoAnalytics, **{"id": analytics_id})[0]
        filename = f"analytics_{analytics.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        file_path = os.path.join(REPORT_DIR, filename)

        if format == "xlsx":
            self._generate_analytics_xlsx_report(analytics, file_path)
        elif format == "pdf":
            self._generate_analytics_pdf_report(analytics, file_path)
        else:
            raise ValueError("Неподдерживамый формат файла!")

        report_data = {
            "video_analytics_id": analytics_id,
            "file_path": file_path
        }
        return DbService(self.session).insert_row_in_table(AnalyticsReport, report_data)

    # Создание XLSX отчета
    def _generate_analytics_xlsx_report(self, analytics: VideoAnalytics, file_path: str):
        data = {
            "Метрика": ["Число аномалий", "Частота", "Размер", "Интенсивность", "Среднее распределение"],
            "Значение": [analytics.anomaly_count, analytics.frequency, analytics.size, analytics.intensity,
                         analytics.avg_distribution]
        }
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    # Создание PDF отчета
    def _generate_analytics_pdf_report(self, analytics: VideoAnalytics, file_path: str):
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4

        # Заголовок
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, height - 2 * cm, "Video Analytics")

        # Отступ от заголовка
        y_position = height - 3 * cm

        # Данные аналитики
        metrics = [
            ("Anomaly count", analytics.anomaly_count),
            ("Frequency", analytics.frequency),
            #("Size", analytics.size),
            ("Intensity", analytics.intensity),
            #("Avg. distribution", analytics.avg_distribution)
        ]

        c.setFont("Helvetica", 12)
        for metric, value in metrics:
            c.drawString(2 * cm, y_position, f"{metric}: {value}")
            y_position -= 1 * cm  # Отступ между строками

        c.save()

    # -----------------------------------------------------ДЛЯ ПОЛЬЗОВАТЕЛЕЙ--------------------------------------------
    def generate_users_report(self, format: str):
        users_with_roles = DbService(self.session).select_users_with_roles()
        if not users_with_roles:
            raise ValueError("Пользователи не найдены")

        filename = f"report_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        file_path = os.path.join(REPORT_DIR, filename)

        if format == "xlsx":
            self._generate_users_report_xlsx(file_path, users_with_roles)
        elif format == "pdf":
            self._generate_users_report_pdf(file_path, users_with_roles)
        else:
            raise ValueError("Неподдерживаемый формат файла!")

        return file_path

    # Создание XLSX отчета
    def _generate_users_report_xlsx(self, file_path: str, users_with_roles):
        # Преобразование объектов SQLAlchemy в словари
        users_data = [
            {
                "ID": user.id,
                "Rolename": role.rolename,
                "Username": user.username,
                "Password": user.password_hash
            }
            for user, role in users_with_roles
        ]

        df = pd.DataFrame(users_data)
        df.to_excel(file_path, index=False)

    # Создание PDF отчета
    def _generate_users_report_pdf(self, file_path: str, users):
        """
        Генерирует PDF отчет для списка пользователей.
        :param file_path: Путь для сохранения PDF файла.
        :param users: Список кортежей (user, role), где user — объект пользователя, role — объект роли.
        """
        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4
            x_margin = 2 * cm
            y_position = height - 2 * cm

            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(width / 2, y_position, "Users Report")
            y_position -= 1.5 * cm

            c.setFont("Helvetica", 10)
            for user, role in users:  # Используем кортеж (user, role)
                if y_position < 3 * cm:
                    c.showPage()  # Создаем новую страницу, если закончилось место
                    y_position = height - 2 * cm
                    c.setFont("Helvetica", 10)

                # Выводим данные пользователя
                c.drawString(x_margin, y_position, f"ID: {user.id}")
                y_position -= 0.5 * cm
                c.drawString(x_margin, y_position, f"Rolename: {role.rolename}")  # Используем role.rolename
                y_position -= 0.5 * cm
                c.drawString(x_margin, y_position, f"Username: {user.username}")
                y_position -= 0.5 * cm

                # Разделение пароля на строки по 90 символов
                if hasattr(user, 'password_hash'):
                    hash_lines = [user.password_hash[i:i + 90] for i in range(0, len(user.password_hash), 90)]
                    c.drawString(x_margin, y_position, "Password hash:")
                    y_position -= 0.5 * cm
                    for line in hash_lines:
                        c.drawString(x_margin + 1 * cm, y_position, line)
                        y_position -= 0.4 * cm

                y_position -= 0.8 * cm  # Отступ между пользователями

            c.save()
            print(f"PDF отчет успешно создан: {file_path}")
            return file_path
        except Exception as e:
            print(f"Ошибка при создании PDF отчета: {e}")
            raise ValueError(f"Ошибка при создании PDF отчета: {e}")


    # -----------------------------------------------------ДЛЯ ЛОГОВ ПОЛЬЗОВАТЕЛЕЙ--------------------------------------------
    # Отчет по всем логам пользователей
    def generate_all_user_logs_report(self, format: str = "xlsx"):
        # Получаем все логи из базы данных
        logs = DbService(self.session).select_user_logs_with_details()
        # Генерация отчета
        filename = f"all_user_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        file_path = os.path.join(REPORT_DIR, filename)

        if format == "xlsx":
            self._generate_all_user_logs_xlsx_report(logs, file_path)
        elif format == "pdf":
            self._generate_all_user_logs_pdf_report(logs, file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла!")

        return file_path

    # Отчет по логам конкретного пользователя
    def generate_user_logs_report(self, user_id: int, format: str = "xlsx"):
        # Получаем логи пользователя из базы данных
        logs = DbService(self.session).select_user_logs_with_details(user_id)
        # Генерация отчета
        filename = f"user_logs_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        file_path = os.path.join(REPORT_DIR, filename)

        if format == "xlsx":
            self._generate_user_logs_xlsx_report(logs, file_path)
        elif format == "pdf":
            self._generate_user_logs_pdf_report(logs, file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла!")

        return file_path

    # Отчет по аналитике сайта за всё время
    def generate_website_analytics_report(self, format: str = "xlsx"):
        # Получаем аналитику за всё время
        analytics = DbService(self.session).select_website_analytics_report()
        # Генерация отчета
        filename = f"website_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        file_path = os.path.join(REPORT_DIR, filename)

        if format == "xlsx":
            self._generate_website_analytics_xlsx_report(analytics, file_path)
        elif format == "pdf":
            self._generate_website_analytics_pdf_report(analytics, file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла!")

        return file_path

    # Приватные методы для генерации отчетов в формате xlsx
    def _generate_all_user_logs_xlsx_report(self, logs, file_path):
        # Преобразование логов в список словарей
        data = []
        for log_tuple in logs:
            action_log, action_log_type, role, user = log_tuple

            # Формирование строки отчета
            row = {
                "datetime": action_log.datetime,
                "action_type": action_log_type.type,
                "username": user.username,
                "rolename": role.rolename,
            }
            data.append(row)

        # Создаем DataFrame и сохраняем в Excel
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    def _generate_user_logs_xlsx_report(self, logs, file_path):
        data = []
        for log_tuple in logs:
            action_log, action_log_type, role, user = log_tuple
            row = {
                "id": action_log.id,
                "datetime": action_log.datetime,
                "username": user.username,
                "role": role.rolename,
                "action_type": action_log_type.type
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    def _generate_website_analytics_xlsx_report(self, analytics, file_path):
        data = []
        for analytic in analytics:
            row = {
                "date": analytic.date,
                "loginsCount": analytic.loginsCount,
                "uploadedCount": analytic.uploadedCount,
                "reportsCount": analytic.reportsCount
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    def _generate_all_user_logs_pdf_report(self, logs, file_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from datetime import datetime

        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4
            x_margin = 2 * cm
            y_position = height - 2 * cm

            # Заголовок отчета
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(width / 2, y_position, "Complete User Activity Log")
            y_position -= 1 * cm

            # Дата генерации отчета
            c.setFont("Helvetica-Oblique", 10)
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.drawCentredString(width / 2, y_position, f"Generated on: {report_date}")
            y_position -= 1.5 * cm

            # Статистика по логам
            unique_users = len({log[3].username for log in logs})
            first_date = min(log[0].datetime for log in logs).strftime("%Y-%m-%d")
            last_date = max(log[0].datetime for log in logs).strftime("%Y-%m-%d")

            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_margin, y_position, "Summary:")
            y_position -= 0.7 * cm

            c.setFont("Helvetica", 10)
            c.drawString(x_margin + 0.5 * cm, y_position, f"Total Actions: {len(logs)}")
            y_position -= 0.5 * cm
            c.drawString(x_margin + 0.5 * cm, y_position, f"Unique Users: {unique_users}")
            y_position -= 0.5 * cm
            c.drawString(x_margin + 0.5 * cm, y_position, f"Period: {first_date} to {last_date}")
            y_position -= 1 * cm

            # Детальные записи логов
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_margin, y_position, "Activity Details:")
            y_position -= 0.7 * cm

            c.setFont("Helvetica", 10)
            for log_tuple in logs:
                if y_position < 3 * cm:
                    c.showPage()
                    y_position = height - 2 * cm
                    c.setFont("Helvetica", 10)

                action_log, action_log_type, role, user = log_tuple
                
                c.drawString(x_margin, y_position, f"ID: {action_log.id}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Date: {action_log.datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"User: {user.username} ({role.rolename})")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Action: {action_log_type.type}")
                y_position -= 1 * cm

            # Подпись
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(x_margin, 1 * cm, "Automatically generated by System Audit Module")

            c.save()
            print(f"Complete user logs PDF generated: {file_path}")
            return file_path

        except Exception as e:
            print(f"Error generating complete user logs PDF: {e}")
            raise ValueError(f"Error generating complete user logs PDF: {e}")

    def _generate_user_logs_pdf_report(self, logs, file_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from datetime import datetime

        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4
            x_margin = 2 * cm
            y_position = height - 2 * cm

            # Заголовок отчета
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(width / 2, y_position, "User Activity Report")
            y_position -= 1 * cm

            # Информация о пользователе (берем из первой записи)
            if logs:
                _, _, role, user = logs[0]
                c.setFont("Helvetica", 10)
                c.drawString(x_margin, y_position, f"User: {user.username}")
                y_position -= 0.5 * cm
                c.drawString(x_margin, y_position, f"Role: {role.rolename}")
                y_position -= 0.5 * cm

            # Дата генерации отчета
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.drawString(x_margin, y_position, f"Report date: {report_date}")
            y_position -= 1.5 * cm

            # Статистика
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_margin, y_position, "Activity Summary:")
            y_position -= 0.7 * cm

            c.setFont("Helvetica", 10)
            c.drawString(x_margin + 0.5 * cm, y_position, f"Total Actions: {len(logs)}")
            y_position -= 0.5 * cm
            
            if logs:
                first_date = min(log[0].datetime for log in logs).strftime("%Y-%m-%d")
                last_date = max(log[0].datetime for log in logs).strftime("%Y-%m-%d")
                c.drawString(x_margin + 0.5 * cm, y_position, f"Period: {first_date} to {last_date}")
                y_position -= 1 * cm

            # Детальные записи
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_margin, y_position, "Action Details:")
            y_position -= 0.7 * cm

            c.setFont("Helvetica", 10)
            for log_tuple in logs:
                if y_position < 3 * cm:
                    c.showPage()
                    y_position = height - 2 * cm
                    c.setFont("Helvetica", 10)

                action_log, action_log_type, _, _ = log_tuple
                
                c.drawString(x_margin, y_position, f"ID: {action_log.id}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Date: {action_log.datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Action Type: {action_log_type.type}")
                y_position -= 0.5 * cm
                if hasattr(action_log, 'description') and action_log.description:
                    c.drawString(x_margin + 0.5 * cm, y_position, f"Details: {action_log.description[:100]}{'...' if len(action_log.description) > 100 else ''}")
                    y_position -= 0.5 * cm
                y_position -= 0.5 * cm  # Дополнительный отступ

            # Подпись
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(x_margin, 1 * cm, "Automatically generated by User Activity Module")

            c.save()
            print(f"User activity PDF report generated: {file_path}")
            return file_path

        except Exception as e:
            print(f"Error generating user activity PDF: {e}")
            raise ValueError(f"Error generating user activity PDF: {e}")


    def _generate_website_analytics_pdf_report(self, analytics, file_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from datetime import datetime

        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4
            x_margin = 2 * cm
            y_position = height - 2 * cm

            # Заголовок отчета
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(width / 2, y_position, "Website Analytics Report")
            y_position -= 1 * cm

            # Дата генерации отчета
            c.setFont("Helvetica-Oblique", 10)
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.drawCentredString(width / 2, y_position, f"Generated on: {report_date}")
            y_position -= 1.5 * cm

            # Статистика по сайту
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_margin, y_position, "Summary Statistics:")
            y_position -= 0.7 * cm

            # Вычисляем общую статистику
            total_logins = sum(a.loginsCount for a in analytics)
            total_uploads = sum(a.uploadedCount for a in analytics)
            total_reports = sum(a.reportsCount for a in analytics)

            c.setFont("Helvetica", 10)
            c.drawString(x_margin + 0.5 * cm, y_position, f"Total Logins: {total_logins}")
            y_position -= 0.5 * cm
            c.drawString(x_margin + 0.5 * cm, y_position, f"Total Uploads: {total_uploads}")
            y_position -= 0.5 * cm
            c.drawString(x_margin + 0.5 * cm, y_position, f"Total Reports: {total_reports}")
            y_position -= 1 * cm

            # Детальная статистика по дням
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_margin, y_position, "Daily Statistics:")
            y_position -= 0.7 * cm

            c.setFont("Helvetica", 10)
            for analytic in analytics:
                if y_position < 3 * cm:
                    c.showPage()  # Новая страница если закончилось место
                    y_position = height - 2 * cm
                    c.setFont("Helvetica", 10)

                # Форматируем дату для отображения
                date_str = analytic.date.strftime("%Y-%m-%d") if hasattr(analytic.date, 'strftime') else str(analytic.date)
                
                c.drawString(x_margin, y_position, f"Date: {date_str}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Logins: {analytic.loginsCount}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Uploads: {analytic.uploadedCount}")
                y_position -= 0.5 * cm
                c.drawString(x_margin + 0.5 * cm, y_position, f"Reports: {analytic.reportsCount}")
                y_position -= 1 * cm  # Отступ между записями

            # Подпись внизу страницы
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(x_margin, 1 * cm, "Generated automatically by Website Analytics System")

            c.save()
            print(f"PDF report successfully generated: {file_path}")
            return file_path

        except Exception as e:
            print(f"Error generating PDF report: {e}")
            raise ValueError(f"Error generating PDF report: {e}")
