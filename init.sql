-- init.sql
-- Создание таблицы ролей
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    rolename VARCHAR(45) UNIQUE NOT NULL
);

-- Создание таблицы пользователей
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(45) UNIQUE NOT NULL,
    password_hash VARCHAR(100) NOT NULL,
    role_id INTEGER NOT NULL,
    FOREIGN KEY (role_id) REFERENCES roles(id)
);

-- Создание таблицы типов действий
CREATE TABLE IF NOT EXISTS action_log_types (
    id SERIAL PRIMARY KEY,
    type VARCHAR(45) NOT NULL
);

-- Создание таблицы логов действий
CREATE TABLE IF NOT EXISTS action_logs (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    user_id INTEGER NOT NULL,
    action_log_type INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (action_log_type) REFERENCES action_log_types(id)
);

-- Создание таблицы видео
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(255) NOT NULL,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Создание таблицы аналитики видео
CREATE TABLE IF NOT EXISTS video_analytics (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL,
    anomaly_count INTEGER NOT NULL,
    frequency FLOAT NOT NULL,
    size INTEGER,
    intensity FLOAT NOT NULL,
    avg_distribution VARCHAR(45),
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

-- Создание таблицы отчетов аналитики
CREATE TABLE IF NOT EXISTS analytics_reports (
    id SERIAL PRIMARY KEY,
    video_analytics_id INTEGER,
    file_path VARCHAR(255),
    FOREIGN KEY (video_analytics_id) REFERENCES video_analytics(id)
);

-- Вставка начальных данных (опционально)
INSERT INTO roles (rolename) VALUES ('admin'), ('user') ON CONFLICT DO NOTHING;
INSERT INTO action_log_types (type) VALUES ('login'), ('logout'), ('video_upload'), ('video_analysis') ON CONFLICT DO NOTHING;
