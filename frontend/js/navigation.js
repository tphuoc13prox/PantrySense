import { byId } from "./dom.js";

const navTabs = document.querySelectorAll(".nav-tab");
const tabViews = document.querySelectorAll(".tab-view");
const detailView = byId("detail-view");

const tabShownHandlers = new Map();

// Lets a feature reload its data whenever its tab is opened.
export function onTabShown(viewId, handler) {
  tabShownHandlers.set(viewId, handler);
}

export function hideTabViews() {
  tabViews.forEach((view) => view.classList.add("hidden"));
}

export function switchTab(viewId) {
  navTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.target === viewId));
  tabViews.forEach((view) => view.classList.toggle("hidden", view.id !== viewId));
  detailView.classList.add("hidden");

  const handler = tabShownHandlers.get(viewId);
  if (handler) handler();
}

export function initNavigation() {
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.target));
  });
}
