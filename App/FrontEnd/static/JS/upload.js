document.addEventListener('DOMContentLoaded', async() => {
    // Проверяем роль пользователя перед загрузкой страницы
    await checkUserRole();

    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('video-file');
    const uploadMessage = document.getElementById('upload-message');


    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        width: 100%;
        height: 5px;
        background: #e0e0e0;
        margin-top: 10px;
        border-radius: 5px;
        overflow: hidden;
    `;
    const progressFill = document.createElement('div');
    progressFill.style.cssText = `
        height: 100%;
        width: 0%;
        background: #007bff;
        transition: width 0.3s ease;
    `;
    progressBar.appendChild(progressFill);
    uploadForm.appendChild(progressBar);

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            showMessage('Выберите файл для загрузки', 'error');
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Показываем состояние загрузки
            const submitBtn = uploadForm.querySelector('button');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Загрузка...';
            progressFill.style.width = '0%';
            uploadMessage.textContent = '';

            const response = await fetch('http://localhost:8081/upload/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${getCookie('users_access_token')}`
                },
                body: formData,
                credentials: 'include'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Ошибка загрузки');
            }

            const result = await response.json();
            showMessage('Видео успешно загружено!', 'success');
            console.log('Upload result:', result);
            
            // Перенаправление на страницу аналитики через 2 секунды
            setTimeout(() => {
                window.location.href = '/pages/analytics';
            }, 2000);

        } catch (error) {
            console.error('Ошибка загрузки:', error);
            showMessage(error.message || 'Произошла ошибка при загрузке', 'error');
        } finally {
            const submitBtn = uploadForm.querySelector('button');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Загрузить';
            }
        }
    });

    // Обработчик для отображения прогресса загрузки
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            const file = fileInput.files[0];
            const maxSize = 500 * 1024 * 1024; // 500MB
            
            if (file.size > maxSize) {
                showMessage('Файл слишком большой (макс. 500MB)', 'error');
                fileInput.value = '';
            } else {
                showMessage(`Выбран файл: ${file.name} (${formatFileSize(file.size)})`, 'info');
            }
        }
    });


    async function checkUserRole() {
        try {
            const response = await fetch('http://localhost:8081/auth/me', {
                method: 'GET',
                credentials: 'include'
            });

            if (response.ok) {
                const userData = await response.json();
                if (userData.role_id === 2) { // Если пользователь (не админ)
                    const adminPanelItem = document.getElementById('admin-panel-item');
                    if (adminPanelItem) {
                        adminPanelItem.style.display = 'none';
                    }
                }
            }
        } catch (error) {
            console.error('Ошибка при проверке роли:', error);
        }
    }

    // Вспомогательные функции
    function showMessage(text, type) {
        uploadMessage.textContent = text;
        uploadMessage.style.color = type === 'error' ? 'red' : type === 'success' ? 'green' : '#007bff';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }
    
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