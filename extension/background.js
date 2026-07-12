// Vidya extension service worker.
// Opens the side panel when the toolbar icon is clicked, and relays
// two messages from the side panel: get the current tab's URL, and
// get a media stream ID for capturing the current tab's audio.

chrome.runtime.onInstalled.addListener(() => {
  try {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  } catch (e) {}
});

chrome.action.onClicked.addListener(async (tab) => {
  try {
    await chrome.sidePanel.open({ tabId: tab.id });
  } catch (e) {}
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || !msg.type) return false;

  if (msg.type === "vidya-get-active-tab") {
    chrome.tabs.query({ active: true, lastFocusedWindow: true }, (tabs) => {
      const tab = tabs && tabs[0];
      sendResponse({
        id: tab ? tab.id : null,
        url: tab ? tab.url : null,
        title: tab ? tab.title : null,
      });
    });
    return true;
  }

  if (msg.type === "vidya-get-tab-stream-id") {
    chrome.tabCapture.getMediaStreamId(
      { targetTabId: msg.tabId },
      (streamId) => {
        sendResponse({
          streamId: streamId || null,
          error: chrome.runtime.lastError ? chrome.runtime.lastError.message : null,
        });
      }
    );
    return true;
  }

  return false;
});
