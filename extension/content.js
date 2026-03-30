// При загрузке страницы пытаемся отправить URL видео в background
function sendVideoUrl() {
  const currentUrl = window.location.href;
  const videoMatch = currentUrl.match(/\/video-?\d+_\d+/);
  let videoUrl = null;
  if (videoMatch) {
    videoUrl = currentUrl;
  } else {
    const videoElement = document.querySelector('video');
    if (videoElement && videoElement.src) {
      videoUrl = videoElement.src;
    }
  }
  if (videoUrl) {
    chrome.runtime.sendMessage({ action: 'checkVideo', url: videoUrl });
  }
}

// Запускаем при загрузке страницы
sendVideoUrl();

// Также слушаем изменения URL для SPA (VK использует history.pushState)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    sendVideoUrl();
  }
}).observe(document, { subtree: true, childList: true });

// Слушаем запросы от popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getVideoUrl') {
    sendVideoUrl(); // отправим в background, но нам нужно вернуть URL в popup
    const currentUrl = window.location.href;
    const videoMatch = currentUrl.match(/\/video-?\d+_\d+/);
    if (videoMatch) {
      sendResponse({ url: currentUrl });
    } else {
      const videoElement = document.querySelector('video');
      if (videoElement && videoElement.src) {
        sendResponse({ url: videoElement.src });
      } else {
        sendResponse({ url: null });
      }
    }
  }
});