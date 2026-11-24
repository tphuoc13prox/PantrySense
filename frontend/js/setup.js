// First-run onboarding: starts the dataset/index build when the backend is
// not ready yet and shows its progress.
import * as api from "./api.js";
import { byId } from "./dom.js";
import { hideTabViews } from "./navigation.js";
import { searchRecipes } from "./search.js";
import { state } from "./state.js";

const DATASET_LIMIT = 380000;
const POLL_MS = 300;
const RETRY_MS = 3000;
const STEP_COUNT = 8;

const engineSelect = byId("engine-select");
const setupView = byId("setup-view");
const searchView = byId("search-view");
const detailView = byId("detail-view");
const progressBar = byId("setup-progress-bar");
const statusText = byId("setup-status-text");
const progressPercent = byId("setup-progress-percent");

let pollId = null;
let onReady = () => {};

function showApp() {
  setupView.classList.add("hidden");
  searchView.classList.remove("hidden");
  onReady();
}

function updateStepper(currentStep) {
  for (let i = 1; i <= STEP_COUNT; i++) {
    const item = byId(`step-item-${i}`);
    if (!item) continue;
    if (i < currentStep) item.className = "stepper-item completed";
    else if (i === currentStep) item.className = "stepper-item active";
    else item.className = "stepper-item";
  }
}

function pollProgress() {
  if (pollId) clearInterval(pollId);

  pollId = setInterval(async () => {
    try {
      const data = await api.getSetupStatus();
      if (!data) return;

      const progress = Math.round(data.progress || 0);
      progressBar.style.width = `${progress}%`;
      progressPercent.textContent = `${progress}%`;
      statusText.textContent = data.message || "Setting up...";
      updateStepper(data.current_step);

      if (data.is_ready || data.progress >= 100 || data.status === "ready") {
        clearInterval(pollId);
        pollId = null;
        showApp();
      }
    } catch (err) {
      console.error("Error polling setup:", err);
    }
  }, POLL_MS);
}

function startSetup(engine) {
  api
    .startSetup(engine || "onnx", DATASET_LIMIT)
    .then(() => pollProgress())
    .catch(() => setTimeout(() => startSetup(engine), RETRY_MS));
}

async function checkSetup() {
  try {
    const data = await api.getSetupStatus();
    if (!data) return;

    if (data.selected_engine) engineSelect.value = data.selected_engine;
    else engineSelect.value = data.cuda_available ? "cuda" : "onnx";

    if (data.is_ready) {
      showApp();
      return;
    }

    hideTabViews();
    detailView.classList.add("hidden");
    setupView.classList.remove("hidden");
    if (data.is_running) pollProgress();
    else startSetup(engineSelect.value);
  } catch (err) {
    console.warn("Could not check setup status:", err);
  }
}

export function initSetup(options) {
  onReady = options.onReady;

  engineSelect.addEventListener("change", async (e) => {
    await api.setEngine(e.target.value);
    if (state.searchIngredients.length > 0) searchRecipes();
  });

  checkSetup();
}
