document.getElementById('checkBtn').addEventListener('click', async () => {
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<div class="loading">Запуск анализа...</div>';

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.tabs.sendMessage(tab.id, { action: 'getVideoUrl' }, async (response) => {
      if (chrome.runtime.lastError) {
        resultDiv.innerHTML = '<div class="error">Не удалось получить ссылку на видео. Убедитесь, что вы на странице видео ВК.</div>';
        return;
      }

      const videoUrl = response.url;
      if (!videoUrl) {
        resultDiv.innerHTML = '<div class="error">Видео не найдено на этой странице.</div>';
        return;
      }

      // Отправляем в background для анализа (может быть уже анализируется)
      chrome.runtime.sendMessage({ action: 'checkVideo', url: videoUrl });

      // Показываем, что анализ запущен, и закроем popup (опционально)
      resultDiv.innerHTML = '<div class="loading">Анализ запущен. Результат придет в уведомлении.</div>';
      // Закрываем popup через 2 секунды, чтобы пользователь видел сообщение
      setTimeout(() => window.close(), 2000);
    });
  } catch (error) {
    resultDiv.innerHTML = `<div class="error">Ошибка: ${error.message}</div>`;
  }
});

// При открытии popup загружаем сохранённый результат для текущей страницы
async function loadCachedResult() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.tabs.sendMessage(tab.id, { action: 'getVideoUrl' }, (response) => {
    if (chrome.runtime.lastError || !response || !response.url) {
      document.getElementById('result').innerHTML = '<div class="error">Не удалось определить видео на странице.</div>';
      return;
    }
    const videoUrl = response.url;
    chrome.runtime.sendMessage({ action: 'getResult', url: videoUrl }, (cached) => {
      const resultDiv = document.getElementById('result');
      if (cached) {
        showResult(cached);
      } else {
        resultDiv.innerHTML = '<div class="loading">Видео ещё не проверялось. Нажмите кнопку, чтобы начать.</div>';
      }
    });
  });
}

function showResult(data) {
  const resultDiv = document.getElementById('result');
  let verdictClass = '';
  if (data.verdict === 'authentic') verdictClass = 'authentic';
  else if (data.verdict === 'suspicious') verdictClass = 'suspicious';
  else if (data.verdict === 'deepfake') verdictClass = 'deepfake';
  else verdictClass = 'insufficient_data';

  let detailsHtml = '';
  if (data.details && data.details.face_analysis) {
    detailsHtml += `<p>Лица: схожесть ${(data.details.face_analysis.avg_similarity * 100).toFixed(1)}%</p>`;
  }
  if (data.details && data.details.audio_analysis) {
    detailsHtml += `<p>Аудио: вероятность дипфейка ${(data.details.audio_analysis.probability * 100).toFixed(1)}%</p>`;
  }

  resultDiv.innerHTML = `
    <div class="${verdictClass}">
      <strong>${data.message}</strong><br>
      Вероятность дипфейка: ${(data.deepfake_probability * 100).toFixed(1)}%
      ${detailsHtml}
    </div>
  `;
}

loadCachedResult();