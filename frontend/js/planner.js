// Weekly meal planner, the "add to plan" modal and the shopping list modal.
import * as api from "./api.js";
import { byId, el, emptyState } from "./dom.js";
import { onTabShown } from "./navigation.js";
import { loadRecipeDetail } from "./recipe-detail.js";
import { state } from "./state.js";

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const SLOTS = ["breakfast", "lunch", "dinner", "snack"];

const grid = byId("meal-plan-grid");

const planModal = byId("plan-modal");
const planRecipeTitle = byId("plan-modal-recipe-title");
const planDaySelect = byId("plan-day-select");
const planSlotSelect = byId("plan-slot-select");
const planServingsInput = byId("plan-servings-input");

const shoppingModal = byId("shopping-modal");
const subtractPantryInput = byId("shopping-subtract-pantry");
const shoppingItems = byId("shopping-list-items");

async function fetchMealPlan() {
  try {
    const plan = await api.getMealPlan();
    if (!plan) return;
    state.mealPlan = plan;
    renderGrid();
  } catch (err) {
    console.debug("Fetch meal plan error:", err);
  }
}

function findEntry(day, slot) {
  return state.mealPlan.find(
    (p) => p.day_of_week.toLowerCase() === day.toLowerCase() && p.meal_slot.toLowerCase() === slot
  );
}

function renderSlot(day, slot) {
  const slotEl = el("div", "plan-slot-item");
  slotEl.appendChild(el("div", "plan-slot-label", slot));

  const content = el("div", "plan-slot-content");
  const entry = findEntry(day, slot);

  if (entry) {
    const clearBtn = el("button", "btn btn-sm btn-outline btn-clear-slot", "×");
    clearBtn.type = "button";
    clearBtn.title = "Remove";
    clearBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await api.clearMealPlanSlot(day, slot);
      fetchMealPlan();
    });
    content.append(el("span", null, entry.recipe_title), clearBtn);
    slotEl.style.cursor = "pointer";
    slotEl.addEventListener("click", () => loadRecipeDetail(entry.recipe_id));
  } else {
    content.appendChild(el("span", "plan-slot-empty", "+ Unscheduled"));
  }

  slotEl.appendChild(content);
  return slotEl;
}

function renderGrid() {
  const dayCards = DAYS.map((day) => {
    const card = el("div", "plan-day-card");
    card.appendChild(el("h3", "plan-day-title", day));
    const slots = el("div", "plan-slots-container");
    slots.append(...SLOTS.map((slot) => renderSlot(day, slot)));
    card.appendChild(slots);
    return card;
  });
  grid.replaceChildren(...dayCards);
}

function openPlanModal() {
  if (!state.activeRecipe) return;
  planRecipeTitle.textContent = state.activeRecipe.title;
  planModal.classList.remove("hidden");
}

async function confirmAddToPlan() {
  const recipe = state.activeRecipe;
  if (!recipe) return;
  const day = planDaySelect.value;
  const slot = planSlotSelect.value;

  try {
    await api.addMealPlanEntry({
      day_of_week: day,
      meal_slot: slot,
      recipe_id: recipe.id,
      recipe_title: recipe.title,
      servings: parseInt(planServingsInput.value, 10) || 2,
    });
    planModal.classList.add("hidden");
    alert(`Scheduled "${recipe.title}" for ${day} ${slot}!`);
  } catch (err) {
    console.error("Schedule error:", err);
  }
}

async function clearWeek() {
  if (!confirm("Are you sure you want to clear all scheduled meals for the week?")) return;
  await api.clearMealPlan();
  fetchMealPlan();
}

function renderShoppingItem(item) {
  const row = el("div", item.in_stock ? "shopping-item in-stock" : "shopping-item");
  const plural = item.recipe_count > 1 ? "s" : "";
  const label = el("span");
  label.append(
    el("strong", null, item.ingredient),
    ` (needed for ${item.recipe_count} meal${plural})`
  );
  row.append(label, el("span", null, item.in_stock ? "✅ In Pantry" : "🛒 To Buy"));
  return row;
}

async function showShoppingList() {
  try {
    const items = await api.getShoppingList(subtractPantryInput.checked);
    if (!items) return;
    if (items.length === 0) {
      shoppingItems.replaceChildren(
        emptyState("No scheduled meals or all ingredients are already in your pantry!")
      );
    } else {
      shoppingItems.replaceChildren(...items.map(renderShoppingItem));
    }
    shoppingModal.classList.remove("hidden");
  } catch (err) {
    console.error("Shopping list error:", err);
  }
}

// One line per item: "<ingredient> (needed for N meals) — <status>".
function copyShoppingList() {
  const lines = Array.from(shoppingItems.querySelectorAll(".shopping-item"), (row) =>
    Array.from(row.children, (cell) => cell.textContent.trim()).join(" — ")
  );
  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    alert("Shopping list copied to clipboard!");
  });
}

export function initPlanner() {
  byId("add-to-plan-btn").addEventListener("click", openPlanModal);
  byId("close-plan-modal").addEventListener("click", () => planModal.classList.add("hidden"));
  byId("confirm-add-plan-btn").addEventListener("click", confirmAddToPlan);
  byId("clear-all-plans-btn").addEventListener("click", clearWeek);

  byId("generate-shopping-list-btn").addEventListener("click", showShoppingList);
  subtractPantryInput.addEventListener("change", showShoppingList);
  byId("close-shopping-modal").addEventListener("click", () => shoppingModal.classList.add("hidden"));
  byId("copy-shopping-list-btn").addEventListener("click", copyShoppingList);

  onTabShown("planner-view", fetchMealPlan);
}
