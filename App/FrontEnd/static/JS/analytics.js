document.addEventListener('DOMContentLoaded', async () => {
    // Проверка роли пользователя и скрытие админ-панели
  
    await checkUserRole();

    

    // Получение и отображение данных аналитики
    await loadVideoAnalytics();

    // Обработчики кнопок генерации отчетов
    document.getElementById('generate-pdf')?.addEventListener('click', (e) => {
        e.preventDefault();
        generateReport('pdf',);
    });

    document.getElementById('generate-excel')?.addEventListener('click', (e) => {
        e.preventDefault();
        generateReport('xlsx');
    });

    // Проверка роли пользователя
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

    // Загрузка аналитики видео
    async function loadVideoAnalytics() {
        try {
            const response = await fetch('http://localhost:8081/analytics/videos', {
                method: 'GET',
                credentials: 'include'
            });

            const noDataDiv = document.getElementById('no-data');
            const analyticsDiv = document.getElementById('analytics-data');
            const analyticsTable = document.getElementById('analytics-table');

            if (!response.ok) {
                throw new Error('Ошибка загрузки аналитики');
            }

            const data = await response.json();
            
            if (!data.videos || data.videos.length === 0) {
                noDataDiv.classList.remove('hidden');
                analyticsDiv.classList.add('hidden');
                return;
            }

            // Отображаем последнее загруженное видео
            
            const latestVideo = data.videos[data.videos.length - 1];
            displayVideoAnalytics(latestVideo);

            noDataDiv.classList.add('hidden');
            analyticsDiv.classList.remove('hidden');

        } catch (error) {
            console.error('Ошибка:', error);
            showMessage('Ошибка при загрузке данных аналитики', 'error');
        }
    }

    // Отображение аналитики конкретного видео
    async function displayVideoAnalytics(video) {
        try {
            const response = await fetch(`http://localhost:8081/analytics/videos/${video.id}/analytics`, {
                method: 'GET',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Аналитика для видео не найдена');
            }

            console.log('video object:', video);

            const analytics = await response.json();
            const videoPlayer = document.getElementById('video-player');
            console.log('videoPlayer:', videoPlayer);

            const analyticsTable = document.getElementById('analytics-table');

            // Установка видео
            console.log('Результат video.file_path: '+ video.file_path); 
            videoPlayer.src = video.file_path; 
            console.log('Результат video.file_path: '+ videoPlayer.src); 
            videoPlayer.load();

            // Заполнение таблицы аналитики
            analyticsTable.innerHTML = `
                <tr>
                    <td>${analytics.anomaly_count}</td>
                    <td>${analytics.frequency.toFixed(2)}</td>
                    <td>${analytics.intensity.toFixed(2)}</td>
                </tr>
            `;

        } catch (error) {
            console.error('Ошибка отображения видео:', error);
            showMessage('Не удалось загрузить данные видео', 'error');
        }
    }

    // Генерация отчета
    async function generateReport(format) {
        try {
            // Сначала получаем список видео пользователя
            const videosResponse = await fetch('http://localhost:8081/analytics/videos', {
                method: 'GET',
                credentials: 'include'
            });

            if (!videosResponse.ok) {
                throw new Error('Не удалось получить список видео');
            }

            const videosData = await videosResponse.json();
        
            if (!videosData.videos || videosData.videos.length === 0) {
                throw new Error('У вас нет загруженных видео для генерации отчета');
            }

            // Берем последнее видео
            const latestVideo = videosData.videos[videosData.videos.length - 1];
            const videoId = latestVideo.id;

            // Формируем URL для запроса отчета
            const reportUrl = `http://localhost:8081/analytics/videos/${videoId}/analytics/report?format=${format}`;
            console.log('Запрашиваем отчет по URL:', reportUrl);

            const responseUrl = await fetch(reportUrl, {
                method: 'GET',
                credentials: 'include'
            });

            if (!responseUrl.ok) {
                const errorData = await responseUrl.json().catch(() => null);
                throw new Error(errorData?.detail || 'Ошибка генерации отчета');
            }

            const reportData = await responseUrl.json();
            const reportId = reportData.id;

            if (!reportId) {
                throw new Error('Не удалось получить ID отчета');
            }

    
            const downloadUrl = `http://localhost:8081/reports/video-analytics/${reportId}`;
        
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
            showMessage(error.message, 'error');
        }
    }

    // Обработка выхода
    document.querySelector('.auth-link a')?.addEventListener('click', async (e) => {
    if (e.target.textContent.trim() === 'Выйти') {
        e.preventDefault();
        try {
            const response = await fetch('http://localhost:8081/auth/logout', {
                method: 'POST',
                credentials: 'include'
            });
            
            if (response.ok) {
                window.location.href = '/pages/login';
            }
        } catch (error) {
            console.error('Ошибка выхода:', error);
        }
    }
    });

});