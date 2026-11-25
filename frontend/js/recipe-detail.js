import { getRecipe } from "./api.js";
import { byId, el, setStatus, formatErrorMessage } from "./dom.js";
import { hideTabViews } from "./navigation.js";
import { state } from "./state.js";

const DIET_TAGS = [
  ["vegetarian", "🌱 Vegetarian"],
  ["vegan", "🌿 Vegan"],
  ["gluten_free", "🌾 Gluten-Free"],
  ["dairy_free", "🥛 Dairy-Free"],
  ["nut_free", "🥜 Nut-Free"],
  ["keto_low_carb", "🥩 Keto"],
];

const detailView = byId("detail-view");
const searchView = byId("search-view");
const statusEl = byId("detail-status-message");
const content = byId("recipe-detail-content");

const titleEl = byId("detail-title");
const timeEl = byId("detail-time");
const difficultyEl = byId("detail-difficulty");
const servingsEl = byId("detail-servings");
const categoryEl = byId("detail-category");
const dietTagsEl = byId("detail-dietary-tags");

const macroEls = {
  calories: byId("macro-calories"),
  protein: byId("macro-protein"),
  carbs: byId("macro-carbs"),
  fat: byId("macro-fat"),
  fiber: byId("macro-fiber"),
};

const substitutionsCard = byId("detail-substitutions-card");
const substitutionsList = byId("substitutions-list");
const ingredientsList = byId("detail-ingredients");
const instructionsList = byId("detail-instructions");

const shownListeners = [];

export function onRecipeShown(listener) {
  shownListeners.push(listener);
}

function asText(value) {
  return value === null || value === undefined ? "" : String(value);
}

export function renderDietTags(container, tags) {
  if (!tags) return;
  for (const [key, label] of DIET_TAGS) {
    if (tags[key]) container.appendChild(el("span", "diet-tag", label));
  }
}

function renderSubstitutionOption(option) {
  const row = el("div", "sub-opt");
  row.append(
    el("strong", null, asText(option.substitute)),
    ` (${asText(option.ratio)}) — `,
    el("em", null, asText(option.context))
  );
  if (option.in_pantry) {
    row.append(" ", el("span", "pantry-has-sub-badge", "In Your Pantry"));
  }
  return row;
}

function renderSubstitutions(substitutions) {
  substitutionsList.replaceChildren();
  if (!substitutions || substitutions.length === 0) {
    substitutionsCard.classList.add("hidden");
    return;
  }
  substitutionsCard.classList.remove("hidden");

  for (const sub of substitutions) {
    const row = el("div", "substitution-item-row");
    row.appendChild(el("div", "sub-missing-title", `Substitute for "${sub.missing_ingredient}":`));
    const options = el("div", "sub-options-list");
    sub.substitutes.forEach((option) => options.appendChild(renderSubstitutionOption(option)));
    row.appendChild(options);
    substitutionsList.appendChild(row);
  }
}

function formatIngredient(item) {
  let text = "";
  if (item.quantity !== null && item.quantity !== undefined) text += `${item.quantity} `;
  if (item.unit) text += `${item.unit} `;
  text += item.name;
  return text.trim();
}

function renderRecipe(recipe) {
  titleEl.textContent = recipe.title || "Untitled Recipe";
  timeEl.textContent = recipe.cooking_time ? `⏱ ${recipe.cooking_time} mins` : "⏱ N/A";
  difficultyEl.textContent = recipe.difficulty ? `⚡ ${recipe.difficulty}` : "⚡ N/A";
  servingsEl.textContent = recipe.servings ? `👥 ${recipe.servings} servings` : "👥 N/A";
  categoryEl.textContent = recipe.category ? `🏷 ${recipe.category}` : "🏷 General";

  dietTagsEl.replaceChildren();
  renderDietTags(dietTagsEl, recipe.dietary_tags);

  const perServing = recipe.nutrition && recipe.nutrition.per_serving;
  if (perServing) {
    macroEls.calories.textContent = `${Math.round(perServing.calories)}`;
    macroEls.protein.textContent = `${perServing.protein}g`;
    macroEls.carbs.textContent = `${perServing.carbs}g`;
    macroEls.fat.textContent = `${perServing.fat}g`;
    macroEls.fiber.textContent = `${perServing.fiber}g`;
  }

  renderSubstitutions(recipe.substitutions);

  ingredientsList.replaceChildren(
    ...(recipe.ingredients || []).map((item) => el("li", "ingredient-item", formatIngredient(item)))
  );
  instructionsList.replaceChildren(...(recipe.instructions || []).map((step) => el("li", null, step)));
}

export async function loadRecipeDetail(recipeId) {
  hideTabViews();
  detailView.classList.remove("hidden");
  content.classList.add("hidden");
  setStatus(statusEl, "Loading recipe details...", "info");

  try {
    const recipe = await getRecipe(recipeId);
    state.activeRecipe = recipe;
    renderRecipe(recipe);
    shownListeners.forEach((listener) => listener(recipe));
    setStatus(statusEl, "");
    content.classList.remove("hidden");
  } catch (err) {
    setStatus(statusEl, formatErrorMessage(err), "error");
  }
}

export function initRecipeDetail() {
  byId("back-btn").addEventListener("click", () => {
    detailView.classList.add("hidden");
    searchView.classList.remove("hidden");
  });
}
