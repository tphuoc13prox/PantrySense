import * as api from "./api.js";
import { initAutocomplete, closeAutocomplete, handleAutocompleteKey } from "./autocomplete.js";
import { byId, el, setStatus, formatErrorMessage } from "./dom.js";
import { loadRecipeDetail, renderDietTags } from "./recipe-detail.js";
import { state } from "./state.js";

const ingredientInput = byId("ingredient-input");
const ingredientChips = byId("ingredient-chips");
const searchBtn = byId("search-btn");
const statusMessage = byId("status-message");
const resultsSection = byId("results-section");
const recipeList = byId("recipe-list");

const filterInputs = {
  vegetarian: byId("filter-vegetarian"),
  vegan: byId("filter-vegan"),
  glutenFree: byId("filter-gluten-free"),
  dairyFree: byId("filter-dairy-free"),
  nutFree: byId("filter-nut-free"),
  keto: byId("filter-keto"),
  time: byId("filter-time"),
  allergen: byId("filter-allergen"),
};

function renderChips() {
  if (state.searchIngredients.length === 0) {
    ingredientChips.replaceChildren(
      el(
        "span",
        "placeholder-text",
        "No ingredients added yet. Type an ingredient above or load from your pantry."
      )
    );
    return;
  }

  const chips = state.searchIngredients.map((ingredient, index) => {
    const chip = el("div", "chip");
    const removeBtn = el("button", "chip-remove", "×");
    removeBtn.type = "button";
    removeBtn.addEventListener("click", () => {
      state.searchIngredients.splice(index, 1);
      renderChips();
    });
    chip.append(el("span", null, ingredient), removeBtn);
    return chip;
  });
  ingredientChips.replaceChildren(...chips);
}

export function setSearchIngredients(ingredients) {
  state.searchIngredients = ingredients;
  renderChips();
}

// Accepts a single name or a comma/semicolon/newline separated list.
function addSearchIngredient(value = null) {
  const raw = (value !== null ? value : ingredientInput.value).trim();
  if (!raw) return;

  const names = raw.split(/[,;\n]+/).map((s) => s.trim()).filter(Boolean);
  for (const name of names) {
    const exists = state.searchIngredients.some((e) => e.toLowerCase() === name.toLowerCase());
    if (!exists) state.searchIngredients.push(name);
  }
  renderChips();
  ingredientInput.value = "";
  closeAutocomplete();
  ingredientInput.focus();
}

function buildFilters() {
  return {
    vegetarian: filterInputs.vegetarian.checked,
    vegan: filterInputs.vegan.checked,
    gluten_free: filterInputs.glutenFree.checked,
    dairy_free: filterInputs.dairyFree.checked,
    nut_free: filterInputs.nutFree.checked,
    keto_low_carb: filterInputs.keto.checked,
    excluded_allergens: filterInputs.allergen.value ? [filterInputs.allergen.value] : [],
    max_cooking_time: filterInputs.time.value ? parseInt(filterInputs.time.value, 10) : null,
  };
}

export async function searchRecipes() {
  if (state.searchIngredients.length === 0) {
    setStatus(statusMessage, "Please enter at least one ingredient.", "info");
    resultsSection.classList.add("hidden");
    return;
  }

  setStatus(statusMessage, "Searching & ranking matching recipes...", "info");
  resultsSection.classList.add("hidden");
  searchBtn.disabled = true;

  try {
    const data = await api.searchRecipes(state.searchIngredients, buildFilters());
    renderRecipeList(data.recipes || []);
  } catch (err) {
    console.error("Search failed:", err);
    setStatus(statusMessage, formatErrorMessage(err), "error");
    resultsSection.classList.add("hidden");
  } finally {
    searchBtn.disabled = false;
  }
}

function renderRecipeCard(recipe) {
  const card = el("div", "recipe-card");
  card.tabIndex = 0;

  const header = el("div", "recipe-card-header");
  const badges = el("div", "badges-group");
  badges.appendChild(el("span", "coverage-badge", `${Math.round(recipe.coverage * 100)}% match`));
  if (recipe.ml_score !== null && recipe.ml_score !== undefined) {
    badges.appendChild(el("span", "ml-badge", `★ ${recipe.ml_score.toFixed(2)}`));
  }
  header.append(el("h3", "recipe-card-title", recipe.title), badges);
  card.appendChild(header);

  const dietRow = el("div", "recipe-dietary-row");
  renderDietTags(dietRow, recipe.dietary_tags);
  if (dietRow.children.length > 0) card.appendChild(dietRow);

  const perServing = recipe.nutrition && recipe.nutrition.per_serving;
  if (perServing) {
    const summary =
      `🔥 ${Math.round(perServing.calories)} kcal | P: ${perServing.protein}g | ` +
      `C: ${perServing.carbs}g | F: ${perServing.fat}g`;
    card.appendChild(el("div", "recipe-nutrition-pill", summary));
  }

  card.appendChild(
    el(
      "div",
      "recipe-card-availability",
      `${recipe.matched_count} / ${recipe.required_count} ingredients available`
    )
  );

  if (recipe.missing_ingredients && recipe.missing_ingredients.length > 0) {
    card.appendChild(
      el("div", "recipe-card-missing", `Missing: ${recipe.missing_ingredients.join(", ")}`)
    );
  }

  if (recipe.substitutions && recipe.substitutions.length > 0) {
    const pantryMatch = recipe.substitutions.some((s) => s.has_pantry_match);
    const hint = pantryMatch
      ? "💡 Substitutes available in your pantry!"
      : "💡 Substitutions available for missing items";
    card.appendChild(el("div", "recipe-card-sub-hint", hint));
  }

  const openDetails = () => loadRecipeDetail(recipe.id);
  card.addEventListener("click", openDetails);
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openDetails();
    }
  });
  return card;
}

function renderRecipeList(recipes) {
  recipeList.replaceChildren();

  if (!recipes || recipes.length === 0) {
    resultsSection.classList.add("hidden");
    setStatus(statusMessage, "No recipes matching your ingredients and dietary filters.", "info");
    return;
  }

  setStatus(statusMessage, "");
  resultsSection.classList.remove("hidden");
  recipeList.replaceChildren(...recipes.map(renderRecipeCard));
}

function clearSearch() {
  state.searchIngredients = [];
  renderChips();
  closeAutocomplete();
  setStatus(statusMessage, "");
  resultsSection.classList.add("hidden");
  recipeList.replaceChildren();
  ingredientInput.value = "";
  ingredientInput.focus();
}

function loadFromPantry() {
  if (state.pantryItems.length === 0) {
    alert("No items in pantry yet.");
    return;
  }
  const merged = new Set([...state.searchIngredients, ...state.pantryItems.map((i) => i.name)]);
  setSearchIngredients(Array.from(merged));
}

export function focusIngredientInput() {
  ingredientInput.focus();
}

export function initSearch() {
  initAutocomplete({
    inputEl: ingredientInput,
    dropdownEl: byId("ingredient-autocomplete-dropdown"),
    onSelect: (name) => addSearchIngredient(name),
  });

  ingredientInput.addEventListener("keydown", (e) => {
    if (handleAutocompleteKey(e)) return;
    if (e.key === "Enter") {
      e.preventDefault();
      addSearchIngredient();
    }
  });

  byId("ingredient-form").addEventListener("submit", (e) => {
    e.preventDefault();
    addSearchIngredient();
  });
  byId("add-btn").addEventListener("click", () => addSearchIngredient());
  byId("load-from-pantry-btn").addEventListener("click", loadFromPantry);
  searchBtn.addEventListener("click", searchRecipes);
  byId("clear-btn").addEventListener("click", clearSearch);

  renderChips();
}
