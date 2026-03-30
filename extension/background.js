const API_BASE = 'http://localhost:8000';

// Храним текущие анализы, чтобы не запускать повторно
const analyzing = new Map();

async function analyzeVideo(url, tabId) {
  // Проверяем, не анализируется ли уже
  if (analyzing.has(url)) {
    console.log('Already analyzing', url);
    return;
  }
  analyzing.set(url, true);

  try {
    const response = await fetch(`${API_BASE}/analyze-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vk_url: url })
    });
    const result = await response.json();

    // Сохраняем результат в storage
    const storageKey = `result_${url}`;
    await chrome.storage.local.set({ [storageKey]: result });

    // Показываем уведомление
    let message = '';
    if (result.verdict === 'authentic') message = 'Видео выглядит подлинным';
    else if (result.verdict === 'suspicious') message = 'Обнаружены признаки обработки';
    else if (result.verdict === 'deepfake') message = 'Высокая вероятность дипфейка';
    else message = 'Недостаточно данных';

    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'AntiDeepfake',
      message: `Результат проверки: ${message} (вероятность ${(result.deepfake_probability * 100).toFixed(1)}%)`
    });
  } catch (error) {
    console.error('Analysis error:', error);
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'AntiDeepfake',
      message: `Ошибка анализа: ${error.message}`
    });
  } finally {
    analyzing.delete(url);
  }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkVideo' && request.url) {
    const tabId = sender.tab?.id;
    analyzeVideo(request.url, tabId);
  }
  // Для popup, чтобы получить результат для текущей страницы
  if (request.action === 'getResult' && request.url) {
    const storageKey = `result_${request.url}`;
    chrome.storage.local.get(storageKey, (data) => {
      sendResponse(data[storageKey] || null);
    });
    return true; // асинхронный ответ
  }
});

// При установке расширения
chrome.runtime.onInstalled.addListener(() => {
  console.log('AntiDeepfake extension installed');
});