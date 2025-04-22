document.addEventListener("DOMContentLoaded", () => {
    const registerForm = document.getElementById("register-form");
    const errorMessage = document.getElementById("error-message");

    registerForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        //errorMessage.textContent = ""; // Очищаем предыдущие ошибки

        // Получаем значения полей
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value;
        const confirmPassword = document.getElementById("confirm-password").value;

        // Валидация на клиенте
        if (!username || !password || !confirmPassword) {
            errorMessage.textContent = "Все поля обязательны для заполнения";
            return;
        }

        if (password.length < 6) {
            errorMessage.textContent = "Пароль должен содержать минимум 6 символов";
            return;
        }

        if (password !== confirmPassword) {
            errorMessage.textContent = "Пароли не совпадают";
            return;
        }

        try {
            // Отправляем запрос на сервер
            const response = await fetch("http://localhost:8081/auth/register", {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ 
                    username: username,
                    password: password 
                })
            });

            const data = await response.json();

            if (!response.ok) {
                // Обрабатываем ошибки от сервера
                throw new Error(data.detail || "Ошибка регистрации");
            }

            // Успешная регистрация
            errorMessage.style.color = "green";
            errorMessage.textContent = "Регистрация успешна! Подождите, перенаправляем вас на страницу авторизации!";
    
            // Перенаправление через 2 секунды
            setTimeout(() => {
                window.location.href = "/pages/login";
            }, 2000);
    
        } catch (error) {
            console.error("Ошибка регистрации:", error);
            errorMessage.style.color = "red";
            errorMessage.textContent = error.message;
        }
    });
});