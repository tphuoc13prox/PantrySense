// PantrySense — Vanilla JavaScript Client

document.addEventListener("DOMContentLoaded", () => {
  // State
  let ingredients = [];
  let currentRecipes = [];

  // DOM Elements - Views
  const searchView = document.getElementById("search-view");
  const detailView = document.getElementById("detail-view");

  // DOM Elements - Search View
  const ingredientInput = document.getElementById("ingredient-input");
  const addBtn = document.getElementById("add-btn");
  const ingredientChips = document.getElementById("ingredient-chips");
  const searchBtn = document.getElementById("search-btn");
  const clearBtn = document.getElementById("clear-btn");
  const statusMessage = document.getElementById("status-message");
  const resultsSection = document.getElementById("results-section");
  const recipeList = document.getElementById("recipe-list");

  // DOM Elements - Detail View
  const backBtn = document.getElementById("back-btn");
  const detailStatusMessage = document.getElementById("detail-status-message");
  const recipeDetailContent = document.getElementById("recipe-detail-content");
  const detailTitle = document.getElementById("detail-title");
  const detailTime = document.getElementById("detail-time");
  const detailDifficulty = document.getElementById("detail-difficulty");
  const detailServings = document.getElementById("detail-servings");
  const detailCategory = document.getElementById("detail-category");
  const detailIngredients = document.getElementById("detail-ingredients");
  const detailInstructions = document.getElementById("detail-instructions");

  // --- Helper Functions ---

  function setStatus(element, text, type = "info") {
    if (!text) {
      element.textContent = "";
      element.className = "status-message hidden";
      return;
    }
    element.textContent = text;
    element.className = `status-message ${type}`;
  }

  function renderChips() {
    ingredientChips.innerHTML = "";
    if (ingredients.length === 0) {
      const placeholder = document.createElement("span");
      placeholder.className = "placeholder-text";
      placeholder.textContent = "No ingredients added yet. Type an ingredient above and click Add.";
      ingredientChips.appendChild(placeholder);
      return;
    }

    ingredients.forEach((ingredient, index) => {
      const chip = document.createElement("div");
      chip.className = "chip";

      const text = document.createElement("span");
      text.textContent = ingredient;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "chip-remove";
      removeBtn.innerHTML = "&times;";
      removeBtn.setAttribute("aria-label", `Remove ${ingredient}`);
      removeBtn.addEventListener("click", () => removeIngredient(index));

      chip.appendChild(text);
      chip.appendChild(removeBtn);
      ingredientChips.appendChild(chip);
    });
  }

  function addIngredient() {
    const value = ingredientInput.value.trim();
    if (!value) return;

    // Avoid duplicate ingredients (case-insensitive)
    const exists = ingredients.some((item) => item.toLowerCase() === value.toLowerCase());
    if (!exists) {
      ingredients.push(value);
      renderChips();
    }

    ingredientInput.value = "";
    ingredientInput.focus();
  }

  function removeIngredient(index) {
    ingredients.splice(index, 1);
    renderChips();
  }

  function clearAll() {
    ingredients = [];
    currentRecipes = [];
    renderChips();
    setStatus(statusMessage, "");
    resultsSection.classList.add("hidden");
    recipeList.innerHTML = "";
    ingredientInput.value = "";
    ingredientInput.focus();
  }

  function renderRecipeList(recipes) {
    recipeList.innerHTML = "";

    if (!recipes || recipes.length === 0) {
      resultsSection.classList.add("hidden");
      setStatus(statusMessage, "No matching recipes found.", "info");
      return;
    }

    setStatus(statusMessage, "");
    resultsSection.classList.remove("hidden");

    recipes.forEach((recipe) => {
      const card = document.createElement("div");
      card.className = "recipe-card";
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `View recipe for ${recipe.title}`);

      const header = document.createElement("div");
      header.className = "recipe-card-header";

      const title = document.createElement("h3");
      title.className = "recipe-card-title";
      title.textContent = recipe.title;

      const coveragePercent = Math.round(recipe.coverage * 100);
      const badge = document.createElement("span");
      badge.className = "coverage-badge";
      badge.textContent = `${coveragePercent}% match`;

      header.appendChild(title);
      header.appendChild(badge);

      const availability = document.createElement("div");
      availability.className = "recipe-card-availability";
      availability.textContent = `${recipe.matched_count} / ${recipe.required_count} ingredients available`;

      card.appendChild(header);
      card.appendChild(availability);

      if (recipe.missing_ingredients && recipe.missing_ingredients.length > 0) {
        const missing = document.createElement("div");
        missing.className = "recipe-card-missing";
        missing.textContent = `Missing: ${recipe.missing_ingredients.join(", ")}`;
        card.appendChild(missing);
      }

      const openDetails = () => loadRecipeDetail(recipe.id);
      card.addEventListener("click", openDetails);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openDetails();
        }
      });

      recipeList.appendChild(card);
    });
  }

  // --- API Calls ---

  async function searchRecipes() {
    if (ingredients.length === 0) {
      setStatus(statusMessage, "Please enter at least one ingredient.", "info");
      resultsSection.classList.add("hidden");
      return;
    }

    setStatus(statusMessage, "Searching for matching recipes...", "info");
    resultsSection.classList.add("hidden");
    searchBtn.disabled = true;

    try {
      const response = await fetch("/api/recipes/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ingredients }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      currentRecipes = data.recipes || [];
      renderRecipeList(currentRecipes);
    } catch (err) {
      console.error("Search failed:", err);
      setStatus(
        statusMessage,
        "Unable to connect to the backend server. Please make sure the server is running.",
        "error"
      );
      resultsSection.classList.add("hidden");
    } finally {
      searchBtn.disabled = false;
    }
  }

  async function loadRecipeDetail(recipeId) {
    // Switch to Detail View
    searchView.classList.add("hidden");
    detailView.classList.remove("hidden");
    recipeDetailContent.classList.add("hidden");
    setStatus(detailStatusMessage, "Loading recipe details...", "info");

    try {
      const response = await fetch(`/api/recipes/${recipeId}`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Recipe not found.");
        }
        throw new Error(`Server returned status ${response.status}`);
      }

      const recipe = await response.json();
      renderRecipeDetail(recipe);
      setStatus(detailStatusMessage, "");
      recipeDetailContent.classList.remove("hidden");
    } catch (err) {
      console.error("Failed to load recipe detail:", err);
      setStatus(
        detailStatusMessage,
        err.message || "Failed to load recipe details. Please try again.",
        "error"
      );
    }
  }

  function renderRecipeDetail(recipe) {
    detailTitle.textContent = recipe.title || "Untitled Recipe";

    detailTime.textContent = recipe.cooking_time
      ? `⏱ ${recipe.cooking_time} mins`
      : "⏱ N/A";
    detailDifficulty.textContent = recipe.difficulty
      ? `⚡ ${recipe.difficulty}`
      : "⚡ N/A";
    detailServings.textContent = recipe.servings
      ? `👥 ${recipe.servings} servings`
      : "👥 N/A";
    detailCategory.textContent = recipe.category
      ? `🏷 ${recipe.category}`
      : "🏷 Uncategorized";

    // Ingredients
    detailIngredients.innerHTML = "";
    if (recipe.ingredients && recipe.ingredients.length > 0) {
      recipe.ingredients.forEach((item) => {
        const li = document.createElement("li");
        li.className = "ingredient-item";

        let text = "";
        if (item.quantity !== null && item.quantity !== undefined) {
          text += `${item.quantity} `;
        }
        if (item.unit) {
          text += `${item.unit} `;
        }
        text += item.name;

        li.textContent = text.trim();
        detailIngredients.appendChild(li);
      });
    }

    // Instructions
    detailInstructions.innerHTML = "";
    if (recipe.instructions && recipe.instructions.length > 0) {
      recipe.instructions.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        detailInstructions.appendChild(li);
      });
    }
  }

  function backToResults() {
    detailView.classList.add("hidden");
    searchView.classList.remove("hidden");
  }

  // --- Event Listeners ---

  addBtn.addEventListener("click", addIngredient);

  ingredientInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addIngredient();
    }
  });

  searchBtn.addEventListener("click", searchRecipes);
  clearBtn.addEventListener("click", clearAll);
  backBtn.addEventListener("click", backToResults);

  // Initial render
  renderChips();
});
