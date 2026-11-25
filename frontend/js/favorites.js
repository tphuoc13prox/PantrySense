import * as api from "./api.js";
import { byId, el, emptyState } from "./dom.js";
import { onTabShown } from "./navigation.js";
import { loadRecipeDetail, onRecipeShown } from "./recipe-detail.js";
import { state } from "./state.js";

const favBadge = byId("fav-badge");
const favoritesList = byId("favorites-list");
const toggleBtn = byId("favorite-toggle-btn");

export async function fetchFavorites() {
  try {
    const favorites = await api.listFavorites();
    if (!favorites) return;
    state.favorites = favorites;
    favBadge.textContent = favorites.length;
    favBadge.classList.toggle("hidden", favorites.length === 0);
    renderFavoritesList();
  } catch (err) {
    console.debug("Fetch favorites error:", err);
  }
}

function setToggleButton(isFavorite) {
  if (isFavorite) {
    toggleBtn.textContent = "⭐ Saved to Favorites";
    toggleBtn.className = "btn btn-primary";
  } else {
    toggleBtn.textContent = "⭐ Bookmark";
    toggleBtn.className = "btn btn-outline";
  }
}

async function refreshToggleButton(recipeId) {
  try {
    const data = await api.checkFavorite(recipeId);
    if (data) setToggleButton(data.is_favorite);
  } catch (err) {
    console.debug("Check favorite error:", err);
  }
}

async function toggleActiveRecipe() {
  const recipe = state.activeRecipe;
  if (!recipe) return;
  const isFavorite = toggleBtn.classList.contains("btn-primary");

  try {
    if (isFavorite) {
      await api.removeFavorite(recipe.id);
      setToggleButton(false);
    } else {
      await api.addFavorite(recipe);
      setToggleButton(true);
    }
    fetchFavorites();
  } catch (err) {
    console.error("Toggle favorite error:", err);
  }
}

function renderFavoriteCard(favorite) {
  const card = el("div", "recipe-card");
  const header = el("div", "recipe-card-header");
  const removeBtn = el("button", "btn btn-sm btn-outline btn-remove-fav", "Remove");
  removeBtn.type = "button";
  header.append(el("h3", "recipe-card-title", favorite.recipe_title), removeBtn);
  card.appendChild(header);

  card.addEventListener("click", (e) => {
    if (e.target.closest(".btn-remove-fav")) return;
    loadRecipeDetail(favorite.recipe_id);
  });
  removeBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await api.removeFavorite(favorite.recipe_id);
    fetchFavorites();
  });
  return card;
}

function renderFavoritesList() {
  if (state.favorites.length === 0) {
    favoritesList.replaceChildren(
      emptyState("No favorite recipes bookmarked yet. Click the ⭐ icon on any recipe to save it here!")
    );
    return;
  }
  favoritesList.replaceChildren(...state.favorites.map(renderFavoriteCard));
}

export function initFavorites() {
  toggleBtn.addEventListener("click", toggleActiveRecipe);
  onRecipeShown((recipe) => refreshToggleButton(recipe.id));
  onTabShown("favorites-view", fetchFavorites);
}
