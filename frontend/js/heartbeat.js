// Keeps the backend informed that this tab is open, so it can shut itself
// down once every tab has been closed.
import { sendHeartbeat, sendLeaveBeacon } from "./api.js";

const TAB_ID_KEY = "pantrysense_tab_id";
const PULSE_MS = 2000;
const MIN_GAP_MS = 1000;

function getTabId() {
  let tabId = sessionStorage.getItem(TAB_ID_KEY);
  if (!tabId) {
    tabId = "tab_" + Math.random().toString(36).substring(2, 15) + "_" + Date.now();
  }
  sessionStorage.setItem(TAB_ID_KEY, tabId);
  return tabId;
}

export function startHeartbeat() {
  const tabId = getTabId();
  let lastSentAt = 0;

  function pulse(force = false) {
    const now = Date.now();
    if (!force && now - lastSentAt < MIN_GAP_MS) return;
    lastSentAt = now;
    sendHeartbeat(tabId).catch(() => {});
  }

  pulse(true);
  const intervalId = setInterval(() => pulse(), PULSE_MS);

  // Browsers throttle timers in background tabs; a worker timer keeps
  // ticking, so the backend does not mistake a hidden tab for a closed one.
  try {
    const workerSource = new Blob([`setInterval(() => postMessage("pulse"), ${PULSE_MS});`], {
      type: "application/javascript",
    });
    const worker = new Worker(URL.createObjectURL(workerSource));
    worker.onmessage = () => pulse();
  } catch (err) {
    console.debug("Heartbeat worker unavailable:", err);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") pulse(true);
  });
  window.addEventListener("focus", () => pulse(true));
  window.addEventListener("pageshow", () => pulse(true));
  window.addEventListener("click", () => pulse());
  window.addEventListener("keydown", () => pulse());

  function leave() {
    clearInterval(intervalId);
    sendLeaveBeacon(tabId);
  }
  window.addEventListener("pagehide", leave);
  window.addEventListener("beforeunload", leave);
}
