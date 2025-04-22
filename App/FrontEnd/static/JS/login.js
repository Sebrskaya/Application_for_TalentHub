document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    const errorMessage = document.getElementById('error-message');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Получаем данные формы
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value.trim();

        // Валидация
        if (!username || !password) {
            errorMessage.textContent = 'Заполните все поля';
            return;
        }

        try {
            // Показываем состояние загрузки
            const submitBtn = loginForm.querySelector('button');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Вход...';
            errorMessage.textContent = '';

            // 1. Отправляем запрос на вход
            const loginResponse = await fetch('http://localhost:8081/auth/login/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include', // Для работы с куками
                body: JSON.stringify({ username, password })
            });

            if (!loginResponse.ok) {
                const errorData = await loginResponse.json();
                throw new Error(errorData.detail || 'Ошибка входа');
            }

            // 2. Получаем данные пользователя
            const meResponse = await fetch('http://localhost:8081/auth/me', {
                method: 'GET',
                credentials: 'include' // Важно для передачи куков
            });

            if (!meResponse.ok) {
                throw new Error('Ошибка получения данных пользователя');
            }

            const userData = await meResponse.json();

            // 3. Перенаправляем по роли
            if (userData.role_id === 1) { // Предполагаем, что 1 - это админ
                window.location.href = '/pages/admin';
            } else {
                window.location.href = '/pages/upload';
            }

        } catch (error) {
            console.error('Ошибка входа:', error);
            errorMessage.textContent = error.message.includes('Invalid') 
                ? 'Неверный логин или пароль' 
                : 'Ошибка при входе в систему';
        } finally {
            // Восстанавливаем кнопку
            const submitBtn = loginForm.querySelector('button');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Войти';
            }
        }
    });
});