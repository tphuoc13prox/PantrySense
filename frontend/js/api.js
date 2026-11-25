// Thin wrappers around the backend endpoints. Path segments built from data
// are URI-encoded.

const JSON_HEADERS = { "Content-Type": "application/json" };
const seg = encodeURIComponent;

function postJSON(url, body) {
  return fetch(url, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) });
}

function del(url) {
  return fetch(url, { method: "DELETE" });
}

// Resolves to null on a non-2xx response; network errors still reject.
async function getJSON(url) {
  const res = await fetch(url);
  return res.ok ? res.json() : null;
}

// System
export const getSetupStatus = () => getJSON("/api/system/setup-status");

export const startSetup = (engine, datasetLimit) =>
  postJSON("/api/system/setup-init", { engine, dataset_limit: datasetLimit }).then((res) => res.json());

export const setEngine = (engine) => postJSON("/api/system/set-engine", { engine });

export const sendHeartbeat = (tabId) => postJSON("/api/system/heartbeat", { tab_id: tabId });

export function sendLeaveBeacon(tabId) {
  if (!navigator.sendBeacon) return;
  const body = new Blob([JSON.stringify({ tab_id: tabId })], { type: "application/json" });
  navigator.sendBeacon("/api/system/heartbeat/leave", body);
}

// Ingredients and recipes
export const suggestIngredients = (query, limit = 8) =>
  getJSON(`/api/ingredients/suggest?q=${seg(query)}&limit=${limit}`);

export async function searchRecipes(ingredients, filters) {
  const res = await postJSON("/api/recipes/search", {
    ingredients,
    filters,
    max_cooking_time: filters.max_cooking_time,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server returned status ${res.status}`);
  }
  return res.json();
}

export async function getRecipe(recipeId) {
  const res = await fetch(`/api/recipes/${seg(recipeId)}`);
  if (!res.ok) throw new Error("Recipe not found.");
  return res.json();
}

export async function parseCookingSteps(instructions) {
  const res = await postJSON("/api/assistant/parse-steps", { instructions });
  if (!res.ok) throw new Error("Server parse steps failed");
  return res.json();
}

// Favorites
export const listFavorites = () => getJSON("/api/favorites");
export const checkFavorite = (recipeId) => getJSON(`/api/favorites/${seg(recipeId)}/check`);
export const removeFavorite = (recipeId) => del(`/api/favorites/${seg(recipeId)}`);

export const addFavorite = (recipe) =>
  postJSON("/api/favorites", {
    recipe_id: recipe.id,
    recipe_title: recipe.title,
    recipe_data: recipe,
  });

// Pantry
export const listPantryItems = () => getJSON("/api/pantry/items");
export const addPantryItem = (item) => postJSON("/api/pantry/items", item);
export const deletePantryItem = (itemId) => del(`/api/pantry/items/${seg(itemId)}`);

// Meal plan
export const getMealPlan = () => getJSON("/api/meal-plan");
export const addMealPlanEntry = (entry) => postJSON("/api/meal-plan", entry);
export const clearMealPlanSlot = (day, slot) => del(`/api/meal-plan/${seg(day)}/${seg(slot)}`);
export const clearMealPlan = () => del("/api/meal-plan/clear");

export const getShoppingList = (subtractPantry) =>
  getJSON(`/api/meal-plan/shopping-list?subtract_pantry=${subtractPantry}`);
