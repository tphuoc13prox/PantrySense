// Step-by-step cooking mode with a per-step countdown timer.
import { parseCookingSteps } from "./api.js";
import { byId, el } from "./dom.js";
import { state } from "./state.js";

const TEMPERATURE_RE = /\b\d{2,3}\s*(?:°\s*[FC]|degrees\s*(?:F|C|fahrenheit|celsius)?|F\b|C\b)/gi;
const DURATION_RE = /\b\d+(?:\s*(?:-|–|to)\s*\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b/gi;

const modal = byId("cooking-modal");
const recipeNameEl = byId("cooking-recipe-name");
const progressFill = byId("cooking-step-progress");
const stepPills = byId("cooking-step-pills");
const stepBadge = byId("cooking-step-badge");
const instructionEl = byId("cooking-step-instruction");
const footerCounter = byId("cooking-footer-counter");
const prevBtn = byId("cooking-prev-btn");
const nextBtn = byId("cooking-next-btn");

const timerBox = byId("cooking-timer-box");
const timerCountdown = byId("timer-countdown");
const timerLabel = byId("timer-label");
const timerStartBtn = byId("timer-start-btn");

let steps = [];
let stepIndex = 0;
let timerId = null;
let remainingSeconds = 0;
let totalSeconds = 0;

function playTimerBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.2);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 1.2);
  } catch (err) {
    console.debug("Audio play error:", err);
  }
}

// Finds temperatures and durations in the raw text. Temperatures win when
// two matches overlap.
function findHighlights(text) {
  const found = [];
  const collect = (re, className) => {
    for (const m of text.matchAll(re)) {
      const start = m.index;
      const end = start + m[0].length;
      const overlaps = found.some((h) => start < h.end && end > h.start);
      if (!overlaps) found.push({ start, end, className });
    }
  };
  collect(TEMPERATURE_RE, "highlight-temp");
  collect(DURATION_RE, "highlight-time");
  return found.sort((a, b) => a.start - b.start);
}

function renderInstruction(container, rawText) {
  container.replaceChildren();
  if (!rawText) return;
  const text = String(rawText);
  let cursor = 0;
  for (const h of findHighlights(text)) {
    if (h.start > cursor) container.append(text.slice(cursor, h.start));
    container.appendChild(el("strong", h.className, text.slice(h.start, h.end)));
    cursor = h.end;
  }
  if (cursor < text.length) container.append(text.slice(cursor));
}

function stopTimer() {
  if (timerId) {
    clearInterval(timerId);
    timerId = null;
  }
}

function updateTimerDisplay() {
  const mins = Math.floor(remainingSeconds / 60);
  const secs = remainingSeconds % 60;
  timerCountdown.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function toggleTimer() {
  if (timerId) {
    stopTimer();
    timerStartBtn.textContent = "Resume Timer";
    return;
  }
  timerStartBtn.textContent = "Pause Timer";
  timerId = setInterval(() => {
    if (remainingSeconds > 0) {
      remainingSeconds--;
      updateTimerDisplay();
      return;
    }
    stopTimer();
    timerStartBtn.textContent = "Completed!";
    playTimerBeep();
    alert("⏰ Timer completed for current step!");
  }, 1000);
}

function resetTimer() {
  stopTimer();
  remainingSeconds = totalSeconds;
  updateTimerDisplay();
  timerStartBtn.textContent = "Start Timer";
}

function addMinute() {
  remainingSeconds += 60;
  totalSeconds += 60;
  updateTimerDisplay();
}

function renderStepPills() {
  const pills = steps.map((_, i) => {
    let className = "step-pill";
    if (i === stepIndex) className += " active";
    else if (i < stepIndex) className += " completed";
    const pill = el("button", className, i + 1);
    pill.type = "button";
    pill.title = `Jump to Step ${i + 1}`;
    pill.addEventListener("click", () => goToStep(i));
    return pill;
  });
  stepPills.replaceChildren(...pills);
}

function renderStep() {
  if (steps.length === 0) return;
  const step = steps[stepIndex];
  const total = steps.length;

  progressFill.style.width = `${Math.round(((stepIndex + 1) / total) * 100)}%`;
  stepBadge.textContent = `Step ${stepIndex + 1} of ${total}`;
  footerCounter.textContent = `${stepIndex + 1} / ${total}`;

  renderInstruction(instructionEl, step.instruction);
  renderStepPills();

  prevBtn.disabled = stepIndex === 0;
  nextBtn.textContent = stepIndex === total - 1 ? "Finish Cooking 🎉" : "Next Step →";

  stopTimer();
  if (step.timer_minutes && step.timer_minutes > 0) {
    timerBox.classList.remove("hidden");
    totalSeconds = step.timer_minutes * 60;
    remainingSeconds = totalSeconds;
    timerLabel.textContent = step.timer_description || `${step.timer_minutes} min timer`;
    updateTimerDisplay();
    timerStartBtn.textContent = "Start Timer";
  } else {
    timerBox.classList.add("hidden");
  }
}

function goToStep(index) {
  stepIndex = index;
  renderStep();
}

function prevStep() {
  if (stepIndex > 0) goToStep(stepIndex - 1);
}

function nextStep() {
  if (stepIndex < steps.length - 1) goToStep(stepIndex + 1);
}

function closeCookingMode() {
  stopTimer();
  modal.classList.add("hidden");
}

// Instructions may arrive as an array, a JSON-encoded array or plain text.
function normalizeInstructions(raw) {
  if (typeof raw !== "string") return raw || [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [raw];
  } catch {
    return raw.split("\n").filter((line) => line.trim().length > 0);
  }
}

async function startCookingMode() {
  const recipe = state.activeRecipe;
  if (!recipe) {
    alert("Please select a recipe first.");
    return;
  }

  const instructions = normalizeInstructions(recipe.instructions);
  if (instructions.length === 0) {
    alert("No cooking instructions available for this recipe.");
    return;
  }

  try {
    steps = await parseCookingSteps(instructions);
  } catch (err) {
    console.warn("Falling back to local step parsing:", err);
    steps = instructions.map((inst, idx) => ({
      step_number: idx + 1,
      instruction: typeof inst === "string" ? inst.trim() : JSON.stringify(inst),
      timer_minutes: null,
      timer_description: null,
    }));
  }

  if (!steps || steps.length === 0) {
    steps = [];
    alert("No valid cooking steps could be parsed.");
    return;
  }

  stepIndex = 0;
  recipeNameEl.textContent = recipe.title || "Recipe";
  renderStep();
  modal.classList.remove("hidden");
}

function handleKeydown(e) {
  if (modal.classList.contains("hidden")) return;
  if (e.key === "Escape") {
    e.preventDefault();
    closeCookingMode();
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    nextStep();
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    prevStep();
  }
}

export function initCooking() {
  byId("start-cooking-btn").addEventListener("click", startCookingMode);
  byId("close-cooking-modal").addEventListener("click", closeCookingMode);
  byId("timer-add-min-btn").addEventListener("click", addMinute);
  byId("timer-reset-btn").addEventListener("click", resetTimer);
  timerStartBtn.addEventListener("click", toggleTimer);
  prevBtn.addEventListener("click", prevStep);

  nextBtn.addEventListener("click", () => {
    if (stepIndex < steps.length - 1) {
      nextStep();
    } else {
      closeCookingMode();
      alert("🎉 Cooking complete! Enjoy your delicious meal!");
    }
  });

  // Clicking the backdrop (outside the card) closes the modal.
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeCookingMode();
  });
  document.addEventListener("keydown", handleKeydown);
}
