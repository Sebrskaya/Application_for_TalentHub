document.addEventListener('DOMContentLoaded', function () {
    const toggleBtn = document.getElementById('toggle-user-edit');
    const editPanel = document.getElementById('user-edit-panel');
    const checkboxes = document.querySelectorAll('.log-checkbox');
    const userReportLinks = document.getElementById('user-report-links');
    const pdfUserLink = document.getElementById('generate-log-user-pdf');
    const excelUserLink = document.getElementById('generate-log-user-xlsx');

    // Загрузка данных при открытии страницы
    loadUsers();
    loadLogs();
    loadWebsiteAnalytics();

    // Переключение панели редактирования пользователя
    toggleBtn.addEventListener('click', function () {
        editPanel.classList.toggle('active');
        toggleBtn.textContent = editPanel.classList.contains('active') ?
            'Редактировать ▲' : 'Редактировать ▼';
    });

    // Обработчик выделения строк
    document.getElementById('logs-table').addEventListener('change', function (e) {
        if (e.target.classList.contains('log-checkbox')) {
            updateUserReportLinks();
        }
    });

    // Обновление ссылок для отчетов по пользователю
    function updateUserReportLinks() {
        const selectedUsernames = new Set();
        const selectedUserIds = new Set();
        document.querySelectorAll('.log-checkbox:checked').forEach(checkbox => {
            const username = checkbox.closest('tr').dataset.username; // Получаем username из data-атрибута
            const userId = checkbox.closest('tr').dataset.userId; // Получаем user_id из data-атрибута
            if (username && userId) {
                selectedUsernames.add(username);
                selectedUserIds.add(userId);
            }
        });
    
        const userReportLinks = document.getElementById('user-report-links');
        const pdfUserLink = document.getElementById('generate-log-user-pdf');
        const excelUserLink = document.getElementById('generate-log-user-xlsx');
        const generateLogUser = document.getElementById('generate-log-user');
    
        if (selectedUsernames.size === 1 && selectedUserIds.size === 1) {
            const username = Array.from(selectedUsernames)[0];
            const userId = Array.from(selectedUserIds)[0];
            userReportLinks.style.display = 'block'; // Показываем окно
            generateLogUser.textContent = username; // Записываем имя пользователя
            generateLogUser.dataset.userId = userId; // Сохраняем user_id
            pdfUserLink.href = `/reports/users-logs/${userId}?format=pdf`; // Обновляем ссылку на PDF
            excelUserLink.href = `/reports/users-logs/${userId}?format=xlsx`; // Обновляем ссылку на Excel
        } else {
            userReportLinks.style.display = 'none'; // Скрываем окно, если выбрано 0 или больше одного
        }
    }

    // Загрузка данных о пользователях
    async function loadUsers() {
        try {
            const response = await fetch('http://localhost:8081/users', {
                method: 'GET',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Ошибка загрузки пользователей');
            }

            const data = await response.json();
            const userTable = document.getElementById('user-table');
            userTable.innerHTML = ''; // Очистка таблицы

            data.users.forEach(user => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${user.id}</td>
                    <td>${user.role_id === 1 ? 'Админ' : 'Пользователь'}</td>
                    <td>${user.username}</td>
                    <td>${user.password_hash}</td>
                `;
                userTable.appendChild(row);
            });
        } catch (error) {
            console.error('Ошибка загрузки пользователей:', error);
        }
    }

    // Добавление нового пользователя
document.getElementById('add-user-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('new-username').value;
    const password = document.getElementById('new-password').value;

    // Проверяем, что все поля заполнены
    if (!username || !password) {
        alert('Все поля обязательны для заполнения');
        return;
    }

    try {
        const response = await fetch('http://localhost:8081/users', {
            method: 'POST', // Используем PUT для создания нового пользователя
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify({
                username,
                password
            })
        });

        if (!response.ok) {
            throw new Error('Ошибка добавления пользователя');
        }

        const data = await response.json();
        alert('Пользователь успешно добавлен');
        loadUsers(); // Обновляем таблицу
        document.getElementById('add-user-form').reset(); // Очищаем форму
    } catch (error) {
        console.error('Ошибка добавления пользователя:', error);
        alert('Ошибка при добавлении пользователя');
    }
});


    // Редактирование пользователя
document.getElementById('user-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userId = document.getElementById('user-id').value;
    const password = document.getElementById('user-password').value;
    const role = document.getElementById('user-role').value;

    // Формируем объект данных для обновления
    const updateData = {};
    if (password.trim()) updateData.password = password; // Добавляем пароль, только если он указан
    if (role) updateData.role_id = role === 'admin' ? 1 : 2; // Добавляем роль, только если она выбрана

    try {
        const response = await fetch(`http://localhost:8081/users/${userId}`, {
            method: 'PATCH', // Используем PATCH для частичного обновления
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify(updateData) // Отправляем только измененные данные
        });

        if (!response.ok) {
            throw new Error('Ошибка редактирования пользователя');
        }

        alert('Пользователь успешно изменен');
        loadUsers(); // Обновляем таблицу
    } catch (error) {
        console.error('Ошибка редактирования пользователя:', error);
        alert('Ошибка при изменении пользователя');
    }
});


    // Удаление пользователя
    document.getElementById('user-delete-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userId = document.getElementById('user-delete-id').value;

        try {
            const response = await fetch(`http://localhost:8081/users/${userId}`, {
                method: 'DELETE',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Ошибка удаления пользователя');
            }

            alert('Пользователь успешно удален');
            loadUsers(); // Обновляем таблицу
        } catch (error) {
            console.error('Ошибка удаления пользователя:', error);
            alert('Ошибка при удалении пользователя');
        }
    });

    // Загрузка логов
    async function loadLogs() {
        try {
            const response = await fetch('http://localhost:8081/logs/users', {
                method: 'GET',
                credentials: 'include'
            });
    
            if (!response.ok) {
                throw new Error('Ошибка загрузки логов');
            }
    
            const data = await response.json();
            const logsTable = document.getElementById('logs-table');
            logsTable.innerHTML = ''; // Очистка таблицы
    
            data.logs.forEach(log => {
                const row = document.createElement('tr');
                row.dataset.username = log.username;
                row.dataset.userId = log.user_id; // Добавляем user_id в dataset
                row.innerHTML = `
                    <td><input type="checkbox" class="log-checkbox"></td>
                    <td>${new Date(log.datetime).toLocaleDateString()}</td>
                    <td>${new Date(log.datetime).toLocaleTimeString()}</td>
                    <td>${log.username}</td>
                    <td>${log.rolename}</td>
                    <td>${log.action_type}</td>
                `;
                logsTable.appendChild(row);
            });
        } catch (error) {
            console.error('Ошибка загрузки логов:', error);
        }
    }

    // Загрузка аналитики сайта
    async function loadWebsiteAnalytics() {
        try {
            const response = await fetch('http://localhost:8081/website/analytics', {
                method: 'GET',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Ошибка загрузки аналитики сайта');
            }

            const data = await response.json();
            const analyticsTable = document.getElementById('analytics-table');
            analyticsTable.innerHTML = ''; // Очистка таблицы

            data.analytics.forEach(analytic => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${new Date(analytic.datetime).toLocaleDateString()}</td>
                    <td>${analytic.loginsCount}</td>
                    <td>${analytic.uploadedCount}</td>
                    <td>${analytic.reportsCount}</td>
                `;
                analyticsTable.appendChild(row);
            });
        } catch (error) {
            console.error('Ошибка загрузки аналитики сайта:', error);
        }
    }

    // Генерация отчета по всем пользователям в PDF
document.getElementById('generate-user-pdf')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const downloadUrl = `http://localhost:8081/reports/users?format=pdf`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

    // Генерация отчета по всем пользователям в Excel
document.getElementById('generate-user-xlsx')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const downloadUrl = `http://localhost:8081/reports/users?format=xlsx`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

// Генерация отчета по всем пользователям в PDF
document.getElementById('generate-analytics-pdf')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const downloadUrl = `http://localhost:8081/reports/website?format=pdf`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

    // Генерация отчета по всем пользователям в Excel
document.getElementById('generate-analytics-xlsx')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const downloadUrl = `http://localhost:8081/reports/website?format=xlsx`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

// Генерация отчета по логам всех пользователей в PDF
document.getElementById('generate-log-pdf')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const downloadUrl = `http://localhost:8081/reports/users-logs?format=pdf`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

// Генерация отчета по логам всех пользователей в Excel
document.getElementById('generate-log-xlsx')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const downloadUrl = `http://localhost:8081/reports/users-logs?format=xlsx`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

// Генерация отчета по логам конкретного пользователя в PDF
document.getElementById('generate-log-user-pdf')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const userId = document.getElementById('generate-log-user').dataset.userId; // Получаем user_id
        console.log(userId)
        const downloadUrl = `http://localhost:8081/reports/users-logs/${userId}?format=pdf`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

// Генерация отчета по логам конкретного пользователя в Excel
document.getElementById('generate-log-user-xlsx')?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        const userId = document.getElementById('generate-log-user').dataset.userId; // Получаем user_id
        const downloadUrl = `http://localhost:8081/reports/users-logs/${userId}?format=xlsx`;

        // Создаём скрытый iframe для скачивания
        const iframe = document.createElement('iframe');
        iframe.src = downloadUrl;
        iframe.style.display = 'none';
        document.body.appendChild(iframe);

        // Очистка через 5 минут (на случай если iframe не удалится автоматически)
        setTimeout(() => {
            document.body.removeChild(iframe);
        }, 300000);
    } catch (error) {
        console.error('Ошибка генерации отчета:', error);
    }
});

    // Обработка выхода
    document.querySelector('.auth-link a')?.addEventListener('click', async (e) => {
        if (e.target.textContent.trim() === 'Выйти') {
            e.preventDefault();
            try {
                await fetch('http://localhost:8081/auth/logout', {
                    method: 'POST',
                    credentials: 'include'
                });
                window.location.href = '/pages/login';
            } catch (error) {
                console.error('Ошибка при выходе:', error);
            }
        }
    });
});
