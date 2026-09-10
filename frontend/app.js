// PantrySense — Vanilla JavaScript Client v1.0.1

document.addEventListener("DOMContentLoaded", () => {
  // Tab Heartbeat Tracking
  const tabId =
    sessionStorage.getItem("pantrysense_tab_id") ||
    "tab_" + Math.random().toString(36).substring(2, 15) + "_" + Date.now();
  sessionStorage.setItem("pantrysense_tab_id", tabId);

  let lastHeartbeatTime = 0;
  function sendHeartbeat(force = false) {
    const now = Date.now();
    if (!force && now - lastHeartbeatTime < 1000) return;
    lastHeartbeatTime = now;
    fetch("/api/system/heartbeat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tab_id: tabId }),
    }).catch(() => {});
  }

  sendHeartbeat(true);
  const heartbeatInterval = setInterval(() => sendHeartbeat(), 2000);

  try {
    const workerBlob = new Blob(
      ["setInterval(function() { postMessage('pulse'); }, 2000);"],
      { type: "application/javascript" }
    );
    const pulseWorker = new Worker(URL.createObjectURL(workerBlob));
    pulseWorker.onmessage = function () {
      sendHeartbeat();
    };
  } catch (e) {
    console.debug("Web worker heartbeat fallback:", e);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") sendHeartbeat(true);
  });
  window.addEventListener("focus", () => sendHeartbeat(true));
  window.addEventListener("pageshow", () => sendHeartbeat(true));
  window.addEventListener("click", () => sendHeartbeat(false));
  window.addEventListener("keydown", () => sendHeartbeat(false));

  function notifyLeave() {
    clearInterval(heartbeatInterval);
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        "/api/system/heartbeat/leave",
        new Blob([JSON.stringify({ tab_id: tabId })], { type: "application/json" })
      );
    }
  }

  window.addEventListener("pagehide", notifyLeave);
  window.addEventListener("beforeunload", notifyLeave);

  // ---------------------------------------------------------------------------
  // Global State
  // ---------------------------------------------------------------------------
  let searchIngredients = [];
  let currentSearchResults = [];
  let pantryItems = [];
  let favoriteRecipes = [];
  let weeklyMealPlan = [];
  let activeRecipeDetail = null;

  // Step-by-Step Cooking Assistant State
  let cookingSteps = [];
  let currentStepIndex = 0;
  let timerInterval = null;
  let timerRemainingSeconds = 0;
  let timerTotalSeconds = 0;

  // ---------------------------------------------------------------------------
  // DOM Elements
  // ---------------------------------------------------------------------------
  const engineSelect = document.getElementById("engine-select");
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabViews = document.querySelectorAll(".tab-view");
  const pantryBadge = document.getElementById("pantry-badge");
  const favBadge = document.getElementById("fav-badge");

  // Setup View
  const setupView = document.getElementById("setup-view");
  const setupProgressBar = document.getElementById("setup-progress-bar");
  const setupStatusText = document.getElementById("setup-status-text");
  const setupProgressPercent = document.getElementById("setup-progress-percent");

  // Search View
  const searchView = document.getElementById("search-view");
  const ingredientInput = document.getElementById("ingredient-input");
  const ingredientAutocompleteDropdown = document.getElementById("ingredient-autocomplete-dropdown");
  const addBtn = document.getElementById("add-btn");
  const ingredientChips = document.getElementById("ingredient-chips");
  const loadFromPantryBtn = document.getElementById("load-from-pantry-btn");
  const searchBtn = document.getElementById("search-btn");
  const clearBtn = document.getElementById("clear-btn");
  const statusMessage = document.getElementById("status-message");
  const resultsSection = document.getElementById("results-section");
  const recipeList = document.getElementById("recipe-list");

  // Dietary & Filter inputs
  const filterVegetarian = document.getElementById("filter-vegetarian");
  const filterVegan = document.getElementById("filter-vegan");
  const filterGlutenFree = document.getElementById("filter-gluten-free");
  const filterDairyFree = document.getElementById("filter-dairy-free");
  const filterNutFree = document.getElementById("filter-nut-free");
  const filterKeto = document.getElementById("filter-keto");
  const filterTime = document.getElementById("filter-time");
  const filterAllergen = document.getElementById("filter-allergen");

  // Pantry View
  const pantryView = document.getElementById("pantry-view");
  const pantryAlertBanner = document.getElementById("pantry-alert-banner");
  const pantryItemName = document.getElementById("pantry-item-name");
  const pantryItemQty = document.getElementById("pantry-item-qty");
  const pantryItemUnit = document.getElementById("pantry-item-unit");
  const pantryItemCat = document.getElementById("pantry-item-cat");
  const pantryItemExpiry = document.getElementById("pantry-item-expiry");
  const pantryAddBtn = document.getElementById("pantry-add-btn");
  const pantryItemsList = document.getElementById("pantry-items-list");
  const pantrySearchRecipesBtn = document.getElementById("pantry-search-recipes-btn");

  // Planner View
  const plannerView = document.getElementById("planner-view");
  const mealPlanGrid = document.getElementById("meal-plan-grid");
  const generateShoppingListBtn = document.getElementById("generate-shopping-list-btn");
  const clearAllPlansBtn = document.getElementById("clear-all-plans-btn");

  // Favorites View
  const favoritesView = document.getElementById("favorites-view");
  const favoritesList = document.getElementById("favorites-list");

  // Detail View
  const detailView = document.getElementById("detail-view");
  const backBtn = document.getElementById("back-btn");
  const favoriteToggleBtn = document.getElementById("favorite-toggle-btn");
  const startCookingBtn = document.getElementById("start-cooking-btn");
  const addToPlanBtn = document.getElementById("add-to-plan-btn");
  const detailStatusMessage = document.getElementById("detail-status-message");
  const recipeDetailContent = document.getElementById("recipe-detail-content");
  const detailTitle = document.getElementById("detail-title");
  const detailTime = document.getElementById("detail-time");
  const detailDifficulty = document.getElementById("detail-difficulty");
  const detailServings = document.getElementById("detail-servings");
  const detailCategory = document.getElementById("detail-category");
  const detailDietaryTags = document.getElementById("detail-dietary-tags");
  const macroCalories = document.getElementById("macro-calories");
  const macroProtein = document.getElementById("macro-protein");
  const macroCarbs = document.getElementById("macro-carbs");
  const macroFat = document.getElementById("macro-fat");
  const macroFiber = document.getElementById("macro-fiber");
  const detailSubstitutionsCard = document.getElementById("detail-substitutions-card");
  const substitutionsList = document.getElementById("substitutions-list");
  const detailIngredients = document.getElementById("detail-ingredients");
  const detailInstructions = document.getElementById("detail-instructions");

  // Modals
  const cookingModal = document.getElementById("cooking-modal");
  const closeCookingModal = document.getElementById("close-cooking-modal");
  const cookingStepProgress = document.getElementById("cooking-step-progress");
  const cookingStepPills = document.getElementById("cooking-step-pills");
  const cookingStepBadge = document.getElementById("cooking-step-badge");
  const cookingRecipeName = document.getElementById("cooking-recipe-name");
  const cookingStepInstruction = document.getElementById("cooking-step-instruction");
  const cookingFooterCounter = document.getElementById("cooking-footer-counter");
  const cookingTimerBox = document.getElementById("cooking-timer-box");
  const timerCountdown = document.getElementById("timer-countdown");
  const timerLabel = document.getElementById("timer-label");
  const timerStartBtn = document.getElementById("timer-start-btn");
  const timerAddMinBtn = document.getElementById("timer-add-min-btn");
  const timerResetBtn = document.getElementById("timer-reset-btn");
  const cookingPrevBtn = document.getElementById("cooking-prev-btn");
  const cookingNextBtn = document.getElementById("cooking-next-btn");

  const planModal = document.getElementById("plan-modal");
  const closePlanModal = document.getElementById("close-plan-modal");
  const planModalRecipeTitle = document.getElementById("plan-modal-recipe-title");
  const planDaySelect = document.getElementById("plan-day-select");
  const planSlotSelect = document.getElementById("plan-slot-select");
  const planServingsInput = document.getElementById("plan-servings-input");
  const confirmAddPlanBtn = document.getElementById("confirm-add-plan-btn");

  const shoppingModal = document.getElementById("shopping-modal");
  const closeShoppingModal = document.getElementById("close-shopping-modal");
  const shoppingSubtractPantry = document.getElementById("shopping-subtract-pantry");
  const copyShoppingListBtn = document.getElementById("copy-shopping-list-btn");
  const shoppingListItems = document.getElementById("shopping-list-items");

  let setupPollInterval = null;

  // ---------------------------------------------------------------------------
  // Tab Navigation
  // ---------------------------------------------------------------------------
  function switchTab(targetId) {
    navTabs.forEach((tab) => {
      tab.classList.toggle("active", tab.getAttribute("data-target") === targetId);
    });

    tabViews.forEach((view) => {
      view.classList.toggle("hidden", view.id !== targetId);
    });

    detailView.classList.add("hidden");

    if (targetId === "pantry-view") {
      fetchPantryItems();
    } else if (targetId === "planner-view") {
      fetchMealPlans();
    } else if (targetId === "favorites-view") {
      fetchFavorites();
    }
  }

  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-target");
      switchTab(target);
    });
  });

  // ---------------------------------------------------------------------------
  // Audio Chime (Web Audio API)
  // ---------------------------------------------------------------------------
  function playTimerBeep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.2);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 1.2);
    } catch (e) {
      console.debug("Audio play error:", e);
    }
  }

  // ---------------------------------------------------------------------------
  // Initial Setup & State Synchronization
  // ---------------------------------------------------------------------------
  async function checkInitialSetup() {
    try {
      const res = await fetch("/api/system/setup-status");
      if (!res.ok) return;
      const data = await res.json();

      if (engineSelect) {
        if (data.selected_engine) {
          engineSelect.value = data.selected_engine;
        } else if (data.cuda_available) {
          engineSelect.value = "cuda";
        } else {
          engineSelect.value = "onnx";
        }
      }

      if (!data.is_ready) {
        tabViews.forEach((v) => v.classList.add("hidden"));
        detailView.classList.add("hidden");
        setupView.classList.remove("hidden");

        if (!data.is_running) {
          const chosen = engineSelect ? engineSelect.value : (data.cuda_available ? "cuda" : "onnx");
          autoStartSetup(chosen);
        } else {
          pollSetupProgress();
        }
      } else {
        setupView.classList.add("hidden");
        searchView.classList.remove("hidden");
        ingredientInput.focus();
        fetchPantryItems();
        fetchFavorites();
      }
    } catch (err) {
      console.warn("Could not check setup status:", err);
    }
  }

  function autoStartSetup(engine) {
    fetch("/api/system/setup-init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engine: engine || "onnx", dataset_limit: 380000 }),
    })
      .then((res) => res.json())
      .then(() => pollSetupProgress())
      .catch(() => setTimeout(() => autoStartSetup(engine), 3000));
  }

  function pollSetupProgress() {
    if (setupPollInterval) clearInterval(setupPollInterval);

    setupPollInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/system/setup-status");
        if (!res.ok) return;
        const data = await res.json();

        const prog = Math.round(data.progress || 0);
        setupProgressBar.style.width = `${prog}%`;
        setupProgressPercent.textContent = `${prog}%`;
        setupStatusText.textContent = data.message || "Setting up...";

        for (let i = 1; i <= 8; i++) {
          const el = document.getElementById(`step-item-${i}`);
          if (!el) continue;
          if (i < data.current_step) el.className = "stepper-item completed";
          else if (i === data.current_step) el.className = "stepper-item active";
          else el.className = "stepper-item";
        }

        if (data.is_ready || data.progress >= 100 || data.status === "ready") {
          clearInterval(setupPollInterval);
          setupPollInterval = null;
          setupView.classList.add("hidden");
          searchView.classList.remove("hidden");
          ingredientInput.focus();
          fetchPantryItems();
          fetchFavorites();
        }
      } catch (e) {
        console.error("Error polling setup:", e);
      }
    }, 300);
  }

  // ---------------------------------------------------------------------------
  // Helper Functions
  // ---------------------------------------------------------------------------
  function formatErrorMessage(err) {
    if (!err) return "An unexpected error occurred.";
    if (err.name === "TypeError" && (err.message === "Failed to fetch" || (typeof err.message === "string" && err.message.includes("fetch")))) {
      return "Cannot connect to server. Ensure uvicorn backend is running.";
    }
    return err.message || "An unexpected error occurred.";
  }

  function setStatus(element, text, type = "info") {
    if (!text) {
      element.textContent = "";
      element.className = "status-message hidden";
      return;
    }
    element.textContent = text;
    element.className = `status-message ${type}`;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ---------------------------------------------------------------------------
  // Ingredient Chips & Search Input
  // ---------------------------------------------------------------------------
  function renderChips() {
    ingredientChips.innerHTML = "";
    if (searchIngredients.length === 0) {
      const placeholder = document.createElement("span");
      placeholder.className = "placeholder-text";
      placeholder.textContent = "No ingredients added yet. Type an ingredient above or load from your pantry.";
      ingredientChips.appendChild(placeholder);
      return;
    }

    searchIngredients.forEach((ingredient, index) => {
      const chip = document.createElement("div");
      chip.className = "chip";
      const text = document.createElement("span");
      text.textContent = ingredient;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "chip-remove";
      removeBtn.innerHTML = "&times;";
      removeBtn.addEventListener("click", () => {
        searchIngredients.splice(index, 1);
        renderChips();
      });

      chip.appendChild(text);
      chip.appendChild(removeBtn);
      ingredientChips.appendChild(chip);
    });
  }

  function addSearchIngredient(val = null) {
    const rawValue = (val !== null ? val : ingredientInput.value).trim();
    if (!rawValue) return;

    const items = rawValue.split(/[,;\n]+/).map((s) => s.trim()).filter((s) => s.length > 0);
    for (const item of items) {
      if (!searchIngredients.some((e) => e.toLowerCase() === item.toLowerCase())) {
        searchIngredients.push(item);
      }
    }
    renderChips();
    ingredientInput.value = "";
    closeAutocomplete();
    ingredientInput.focus();
  }

  // ---------------------------------------------------------------------------
  // Autocomplete
  // ---------------------------------------------------------------------------
  let suggestionsList = [];
  let selectedSuggestionIndex = -1;
  let autocompleteDebounceTimer = null;

  function closeAutocomplete() {
    if (ingredientAutocompleteDropdown) {
      ingredientAutocompleteDropdown.innerHTML = "";
      ingredientAutocompleteDropdown.classList.add("hidden");
    }
    suggestionsList = [];
    selectedSuggestionIndex = -1;
  }

  function renderSuggestions(query, items) {
    if (!ingredientAutocompleteDropdown) return;
    if (!items || items.length === 0) {
      closeAutocomplete();
      return;
    }

    suggestionsList = items;
    selectedSuggestionIndex = -1;
    ingredientAutocompleteDropdown.innerHTML = "";

    items.forEach((item, idx) => {
      const div = document.createElement("div");
      div.className = "suggestion-item";
      div.setAttribute("role", "option");

      const mainDiv = document.createElement("div");
      mainDiv.className = "suggestion-main";

      const nameSpan = document.createElement("span");
      nameSpan.className = "suggestion-text";
      nameSpan.textContent = item.name;
      mainDiv.appendChild(nameSpan);

      if (item.is_correction) {
        const typoBadge = document.createElement("span");
        typoBadge.className = "suggestion-typo-badge";
        typoBadge.textContent = "✨ Did you mean?";
        mainDiv.appendChild(typoBadge);
      }

      div.appendChild(mainDiv);

      const countVal = item.frequency || item.count || 0;
      if (countVal > 0) {
        const countSpan = document.createElement("span");
        countSpan.className = "suggestion-count";
        countSpan.textContent = `${countVal.toLocaleString()} recipes`;
        div.appendChild(countSpan);
      }

      div.addEventListener("mousedown", (e) => {
        e.preventDefault();
        addSearchIngredient(item.name);
      });

      ingredientAutocompleteDropdown.appendChild(div);
    });

    ingredientAutocompleteDropdown.classList.remove("hidden");
  }

  function getActiveToken() {
    if (!ingredientInput) return "";
    const parts = ingredientInput.value.split(/[,;\n]+/);
    return parts[parts.length - 1].trim();
  }

  async function fetchSuggestions(query) {
    const trimmed = (query || "").trim();
    if (!trimmed) {
      closeAutocomplete();
      return;
    }
    try {
      const res = await fetch(`/api/ingredients/suggest?q=${encodeURIComponent(trimmed)}&limit=8`);
      if (!res.ok) return;
      const data = await res.json();
      renderSuggestions(trimmed, data.suggestions || []);
    } catch (e) {
      console.debug("Suggestion error:", e);
    }
  }

  function handleIngredientInput() {
    if (autocompleteDebounceTimer) clearTimeout(autocompleteDebounceTimer);
    const token = getActiveToken();
    if (!token) {
      closeAutocomplete();
      return;
    }
    autocompleteDebounceTimer = setTimeout(() => fetchSuggestions(token), 100);
  }

  // ---------------------------------------------------------------------------
  // Recipe Search API
  // ---------------------------------------------------------------------------
  function buildDietaryFiltersPayload() {
    const filters = {
      vegetarian: filterVegetarian.checked,
      vegan: filterVegan.checked,
      gluten_free: filterGlutenFree.checked,
      dairy_free: filterDairyFree.checked,
      nut_free: filterNutFree.checked,
      keto_low_carb: filterKeto.checked,
      excluded_allergens: filterAllergen.value ? [filterAllergen.value] : [],
      max_cooking_time: filterTime.value ? parseInt(filterTime.value, 10) : null,
    };
    return filters;
  }

  async function searchRecipes() {
    if (searchIngredients.length === 0) {
      setStatus(statusMessage, "Please enter at least one ingredient.", "info");
      resultsSection.classList.add("hidden");
      return;
    }

    setStatus(statusMessage, "Searching & ranking matching recipes...", "info");
    resultsSection.classList.add("hidden");
    searchBtn.disabled = true;

    try {
      const filters = buildDietaryFiltersPayload();
      const response = await fetch("/api/recipes/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ingredients: searchIngredients,
          filters: filters,
          max_cooking_time: filters.max_cooking_time,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      currentSearchResults = data.recipes || [];
      renderRecipeList(currentSearchResults);
    } catch (err) {
      console.error("Search failed:", err);
      setStatus(statusMessage, formatErrorMessage(err), "error");
      resultsSection.classList.add("hidden");
    } finally {
      searchBtn.disabled = false;
    }
  }

  function renderRecipeList(recipes) {
    recipeList.innerHTML = "";

    if (!recipes || recipes.length === 0) {
      resultsSection.classList.add("hidden");
      setStatus(statusMessage, "No recipes matching your ingredients and dietary filters.", "info");
      return;
    }

    setStatus(statusMessage, "");
    resultsSection.classList.remove("hidden");

    recipes.forEach((recipe) => {
      const card = document.createElement("div");
      card.className = "recipe-card";
      card.tabIndex = 0;

      const header = document.createElement("div");
      header.className = "recipe-card-header";

      const title = document.createElement("h3");
      title.className = "recipe-card-title";
      title.textContent = recipe.title;

      const badgesGroup = document.createElement("div");
      badgesGroup.className = "badges-group";

      const coveragePercent = Math.round(recipe.coverage * 100);
      const coverageBadge = document.createElement("span");
      coverageBadge.className = "coverage-badge";
      coverageBadge.textContent = `${coveragePercent}% match`;
      badgesGroup.appendChild(coverageBadge);

      if (recipe.ml_score !== null && recipe.ml_score !== undefined) {
        const mlBadge = document.createElement("span");
        mlBadge.className = "ml-badge";
        mlBadge.textContent = `★ ${recipe.ml_score.toFixed(2)}`;
        badgesGroup.appendChild(mlBadge);
      }

      header.appendChild(title);
      header.appendChild(badgesGroup);
      card.appendChild(header);

      // Dietary tags
      if (recipe.dietary_tags) {
        const dietRow = document.createElement("div");
        dietRow.className = "recipe-dietary-row";
        if (recipe.dietary_tags.vegetarian) dietRow.innerHTML += '<span class="diet-tag">🌱 Vegetarian</span>';
        if (recipe.dietary_tags.vegan) dietRow.innerHTML += '<span class="diet-tag">🌿 Vegan</span>';
        if (recipe.dietary_tags.gluten_free) dietRow.innerHTML += '<span class="diet-tag">🌾 Gluten-Free</span>';
        if (recipe.dietary_tags.dairy_free) dietRow.innerHTML += '<span class="diet-tag">🥛 Dairy-Free</span>';
        if (recipe.dietary_tags.nut_free) dietRow.innerHTML += '<span class="diet-tag">🥜 Nut-Free</span>';
        if (recipe.dietary_tags.keto_low_carb) dietRow.innerHTML += '<span class="diet-tag">🥩 Keto</span>';
        if (dietRow.children.length > 0) card.appendChild(dietRow);
      }

      // Nutrition overview pill
      if (recipe.nutrition && recipe.nutrition.per_serving) {
        const nut = recipe.nutrition.per_serving;
        const nutPill = document.createElement("div");
        nutPill.className = "recipe-nutrition-pill";
        nutPill.textContent = `🔥 ${Math.round(nut.calories)} kcal | P: ${nut.protein}g | C: ${nut.carbs}g | F: ${nut.fat}g`;
        card.appendChild(nutPill);
      }

      const availability = document.createElement("div");
      availability.className = "recipe-card-availability";
      availability.textContent = `${recipe.matched_count} / ${recipe.required_count} ingredients available`;
      card.appendChild(availability);

      if (recipe.missing_ingredients && recipe.missing_ingredients.length > 0) {
        const missing = document.createElement("div");
        missing.className = "recipe-card-missing";
        missing.textContent = `Missing: ${recipe.missing_ingredients.join(", ")}`;
        card.appendChild(missing);
      }

      // Check if substitutions are available for missing items
      if (recipe.substitutions && recipe.substitutions.length > 0) {
        const subHint = document.createElement("div");
        subHint.className = "recipe-card-sub-hint";
        const pantryMatch = recipe.substitutions.some((s) => s.has_pantry_match);
        subHint.textContent = pantryMatch
          ? "💡 Substitutes available in your pantry!"
          : "💡 Substitutions available for missing items";
        card.appendChild(subHint);
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

  // ---------------------------------------------------------------------------
  // Recipe Details & Actions
  // ---------------------------------------------------------------------------
  async function loadRecipeDetail(recipeId) {
    tabViews.forEach((v) => v.classList.add("hidden"));
    detailView.classList.remove("hidden");
    recipeDetailContent.classList.add("hidden");
    setStatus(detailStatusMessage, "Loading recipe details...", "info");

    try {
      const response = await fetch(`/api/recipes/${recipeId}`);
      if (!response.ok) throw new Error("Recipe not found.");

      const recipe = await response.json();
      activeRecipeDetail = recipe;
      renderRecipeDetail(recipe);
      checkIfRecipeIsFavorite(recipe.id);
      setStatus(detailStatusMessage, "");
      recipeDetailContent.classList.remove("hidden");
    } catch (err) {
      setStatus(detailStatusMessage, formatErrorMessage(err), "error");
    }
  }

  function renderRecipeDetail(recipe) {
    detailTitle.textContent = recipe.title || "Untitled Recipe";
    detailTime.textContent = recipe.cooking_time ? `⏱ ${recipe.cooking_time} mins` : "⏱ N/A";
    detailDifficulty.textContent = recipe.difficulty ? `⚡ ${recipe.difficulty}` : "⚡ N/A";
    detailServings.textContent = recipe.servings ? `👥 ${recipe.servings} servings` : "👥 N/A";
    detailCategory.textContent = recipe.category ? `🏷 ${recipe.category}` : "🏷 General";

    // Dietary tags
    detailDietaryTags.innerHTML = "";
    if (recipe.dietary_tags) {
      if (recipe.dietary_tags.vegetarian) detailDietaryTags.innerHTML += '<span class="diet-tag">🌱 Vegetarian</span>';
      if (recipe.dietary_tags.vegan) detailDietaryTags.innerHTML += '<span class="diet-tag">🌿 Vegan</span>';
      if (recipe.dietary_tags.gluten_free) detailDietaryTags.innerHTML += '<span class="diet-tag">🌾 Gluten-Free</span>';
      if (recipe.dietary_tags.dairy_free) detailDietaryTags.innerHTML += '<span class="diet-tag">🥛 Dairy-Free</span>';
      if (recipe.dietary_tags.nut_free) detailDietaryTags.innerHTML += '<span class="diet-tag">🥜 Nut-Free</span>';
      if (recipe.dietary_tags.keto_low_carb) detailDietaryTags.innerHTML += '<span class="diet-tag">🥩 Keto</span>';
    }

    // Nutrition
    if (recipe.nutrition && recipe.nutrition.per_serving) {
      const nut = recipe.nutrition.per_serving;
      macroCalories.textContent = `${Math.round(nut.calories)}`;
      macroProtein.textContent = `${nut.protein}g`;
      macroCarbs.textContent = `${nut.carbs}g`;
      macroFat.textContent = `${nut.fat}g`;
      macroFiber.textContent = `${nut.fiber}g`;
    }

    // Substitutions
    substitutionsList.innerHTML = "";
    if (recipe.substitutions && recipe.substitutions.length > 0) {
      detailSubstitutionsCard.classList.remove("hidden");
      recipe.substitutions.forEach((sub) => {
        const row = document.createElement("div");
        row.className = "substitution-item-row";

        const title = document.createElement("div");
        title.className = "sub-missing-title";
        title.textContent = `Substitute for "${sub.missing_ingredient}":`;
        row.appendChild(title);

        const list = document.createElement("div");
        list.className = "sub-options-list";
        sub.substitutes.forEach((opt) => {
          const optDiv = document.createElement("div");
          optDiv.className = "sub-opt";
          optDiv.innerHTML = `<strong>${escapeHtml(opt.substitute)}</strong> (${escapeHtml(opt.ratio)}) — <em>${escapeHtml(opt.context)}</em>`;
          if (opt.in_pantry) {
            optDiv.innerHTML += ' <span class="pantry-has-sub-badge">In Your Pantry</span>';
          }
          list.appendChild(optDiv);
        });

        row.appendChild(list);
        substitutionsList.appendChild(row);
      });
    } else {
      detailSubstitutionsCard.classList.add("hidden");
    }

    // Ingredients list
    detailIngredients.innerHTML = "";
    if (recipe.ingredients && recipe.ingredients.length > 0) {
      recipe.ingredients.forEach((item) => {
        const li = document.createElement("li");
        li.className = "ingredient-item";
        let text = "";
        if (item.quantity !== null && item.quantity !== undefined) text += `${item.quantity} `;
        if (item.unit) text += `${item.unit} `;
        text += item.name;
        li.textContent = text.trim();
        detailIngredients.appendChild(li);
      });
    }

    // Instructions list
    detailInstructions.innerHTML = "";
    if (recipe.instructions && recipe.instructions.length > 0) {
      recipe.instructions.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        detailInstructions.appendChild(li);
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Favorites Management
  // ---------------------------------------------------------------------------
  async function fetchFavorites() {
    try {
      const res = await fetch("/api/favorites");
      if (!res.ok) return;
      favoriteRecipes = await res.json();
      if (favBadge) {
        favBadge.textContent = favoriteRecipes.length;
        favBadge.classList.toggle("hidden", favoriteRecipes.length === 0);
      }
      renderFavoritesList();
    } catch (e) {
      console.debug("Fetch favorites error:", e);
    }
  }

  async function checkIfRecipeIsFavorite(recipeId) {
    try {
      const res = await fetch(`/api/favorites/${recipeId}/check`);
      if (!res.ok) return;
      const data = await res.json();
      updateFavoriteButtonUI(data.is_favorite);
    } catch (e) {
      console.debug("Check favorite error:", e);
    }
  }

  function updateFavoriteButtonUI(isFav) {
    if (isFav) {
      favoriteToggleBtn.innerHTML = "⭐ Saved to Favorites";
      favoriteToggleBtn.className = "btn btn-primary";
    } else {
      favoriteToggleBtn.innerHTML = "⭐ Bookmark";
      favoriteToggleBtn.className = "btn btn-outline";
    }
  }

  async function toggleFavoriteRecipe() {
    if (!activeRecipeDetail) return;
    const recipeId = activeRecipeDetail.id;
    const isCurrentlyFav = favoriteToggleBtn.classList.contains("btn-primary");

    try {
      if (isCurrentlyFav) {
        await fetch(`/api/favorites/${recipeId}`, { method: "DELETE" });
        updateFavoriteButtonUI(false);
      } else {
        await fetch("/api/favorites", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            recipe_id: recipeId,
            recipe_title: activeRecipeDetail.title,
            recipe_data: activeRecipeDetail,
          }),
        });
        updateFavoriteButtonUI(true);
      }
      fetchFavorites();
    } catch (e) {
      console.error("Toggle favorite error:", e);
    }
  }

  function renderFavoritesList() {
    favoritesList.innerHTML = "";
    if (favoriteRecipes.length === 0) {
      favoritesList.innerHTML = '<div class="empty-state">No favorite recipes bookmarked yet. Click the ⭐ icon on any recipe to save it here!</div>';
      return;
    }

    favoriteRecipes.forEach((fav) => {
      const card = document.createElement("div");
      card.className = "recipe-card";
      card.innerHTML = `
        <div class="recipe-card-header">
          <h3 class="recipe-card-title">${escapeHtml(fav.recipe_title)}</h3>
          <button class="btn btn-sm btn-outline btn-remove-fav" data-id="${fav.recipe_id}">Remove</button>
        </div>
      `;

      card.addEventListener("click", (e) => {
        if (e.target.closest(".btn-remove-fav")) return;
        loadRecipeDetail(fav.recipe_id);
      });

      const removeBtn = card.querySelector(".btn-remove-fav");
      removeBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/favorites/${fav.recipe_id}`, { method: "DELETE" });
        fetchFavorites();
      });

      favoritesList.appendChild(card);
    });
  }

  // ---------------------------------------------------------------------------
  // Step-by-Step Interactive Cooking Assistant
  // ---------------------------------------------------------------------------
  function formatStepInstruction(rawText) {
    if (!rawText) return "";
    let formatted = escapeHtml(rawText);
    // Highlight temperatures (e.g. 350°F, 400 degrees, 180°C, 375 F)
    formatted = formatted.replace(
      /(\b\d{2,3}\s*(?:°\s*[FC]|degrees\s*(?:F|C|fahrenheit|celsius)?|F\b|C\b))/gi,
      '<strong class="highlight-temp">$1</strong>'
    );
    // Highlight times (e.g. 15 minutes, 1 hour, 30 mins)
    formatted = formatted.replace(
      /(\b\d+(?:\s*(?:-|–|to)\s*\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b)/gi,
      '<strong class="highlight-time">$1</strong>'
    );
    return formatted;
  }

  function renderCookingPills() {
    if (!cookingStepPills) return;
    cookingStepPills.innerHTML = "";
    const total = cookingSteps.length;
    for (let i = 0; i < total; i++) {
      const pill = document.createElement("button");
      pill.type = "button";
      let cls = "step-pill";
      if (i === currentStepIndex) cls += " active";
      else if (i < currentStepIndex) cls += " completed";
      pill.className = cls;
      pill.textContent = `${i + 1}`;
      pill.title = `Jump to Step ${i + 1}`;
      pill.addEventListener("click", () => {
        currentStepIndex = i;
        renderCookingStep();
      });
      cookingStepPills.appendChild(pill);
    }
  }

  async function startCookingMode() {
    if (!activeRecipeDetail) {
      alert("Please select a recipe first.");
      return;
    }

    let rawInstructions = activeRecipeDetail.instructions || [];
    if (typeof rawInstructions === "string") {
      try {
        const parsed = JSON.parse(rawInstructions);
        rawInstructions = Array.isArray(parsed) ? parsed : [rawInstructions];
      } catch (e) {
        rawInstructions = rawInstructions.split("\n").filter((l) => l.trim().length > 0);
      }
    }

    if (!rawInstructions || rawInstructions.length === 0) {
      alert("No cooking instructions available for this recipe.");
      return;
    }

    try {
      const res = await fetch("/api/assistant/parse-steps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instructions: rawInstructions }),
      });
      if (res.ok) {
        cookingSteps = await res.json();
      } else {
        throw new Error("Server parse steps failed");
      }
    } catch (e) {
      console.warn("Falling back to local step parsing:", e);
      cookingSteps = rawInstructions.map((inst, idx) => ({
        step_number: idx + 1,
        instruction: typeof inst === "string" ? inst.trim() : JSON.stringify(inst),
        timer_minutes: null,
        timer_description: null,
      }));
    }

    if (!cookingSteps || cookingSteps.length === 0) {
      alert("No valid cooking steps could be parsed.");
      return;
    }

    currentStepIndex = 0;
    cookingRecipeName.textContent = activeRecipeDetail.title || "Recipe";
    renderCookingStep();
    cookingModal.classList.remove("hidden");
  }

  function renderCookingStep() {
    if (!cookingSteps || cookingSteps.length === 0) return;
    const step = cookingSteps[currentStepIndex];
    const total = cookingSteps.length;
    const percent = Math.round(((currentStepIndex + 1) / total) * 100);

    cookingStepProgress.style.width = `${percent}%`;
    cookingStepBadge.textContent = `Step ${currentStepIndex + 1} of ${total}`;
    if (cookingFooterCounter) {
      cookingFooterCounter.textContent = `${currentStepIndex + 1} / ${total}`;
    }

    cookingStepInstruction.innerHTML = formatStepInstruction(step.instruction);
    renderCookingPills();

    cookingPrevBtn.disabled = currentStepIndex === 0;
    cookingNextBtn.textContent = currentStepIndex === total - 1 ? "Finish Cooking 🎉" : "Next Step →";

    // Clean previous step timer
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }

    if (step.timer_minutes && step.timer_minutes > 0) {
      cookingTimerBox.classList.remove("hidden");
      timerTotalSeconds = step.timer_minutes * 60;
      timerRemainingSeconds = timerTotalSeconds;
      timerLabel.textContent = step.timer_description || `${step.timer_minutes} min timer`;
      updateTimerDisplay();
      timerStartBtn.textContent = "Start Timer";
    } else {
      cookingTimerBox.classList.add("hidden");
    }
  }

  function updateTimerDisplay() {
    const mins = Math.floor(timerRemainingSeconds / 60);
    const secs = timerRemainingSeconds % 60;
    timerCountdown.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function toggleCookingTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
      timerStartBtn.textContent = "Resume Timer";
    } else {
      timerStartBtn.textContent = "Pause Timer";
      timerInterval = setInterval(() => {
        if (timerRemainingSeconds > 0) {
          timerRemainingSeconds--;
          updateTimerDisplay();
        } else {
          clearInterval(timerInterval);
          timerInterval = null;
          timerStartBtn.textContent = "Completed!";
          playTimerBeep();
          alert("⏰ Timer completed for current step!");
        }
      }, 1000);
    }
  }

  function resetCookingTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
    timerRemainingSeconds = timerTotalSeconds;
    updateTimerDisplay();
    timerStartBtn.textContent = "Start Timer";
  }

  function addMinuteToTimer() {
    timerRemainingSeconds += 60;
    timerTotalSeconds += 60;
    updateTimerDisplay();
  }

  function closeCookingAssistantModal() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
    cookingModal.classList.add("hidden");
  }

  cookingPrevBtn.addEventListener("click", () => {
    if (currentStepIndex > 0) {
      currentStepIndex--;
      renderCookingStep();
    }
  });

  cookingNextBtn.addEventListener("click", () => {
    if (currentStepIndex < cookingSteps.length - 1) {
      currentStepIndex++;
      renderCookingStep();
    } else {
      closeCookingAssistantModal();
      alert("🎉 Cooking complete! Enjoy your delicious meal!");
    }
  });

  closeCookingModal.addEventListener("click", closeCookingAssistantModal);
  if (timerAddMinBtn) timerAddMinBtn.addEventListener("click", addMinuteToTimer);

  cookingModal.addEventListener("click", (e) => {
    if (e.target === cookingModal) {
      closeCookingAssistantModal();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (cookingModal && !cookingModal.classList.contains("hidden")) {
      if (e.key === "Escape") {
        e.preventDefault();
        closeCookingAssistantModal();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        if (currentStepIndex < cookingSteps.length - 1) {
          currentStepIndex++;
          renderCookingStep();
        }
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (currentStepIndex > 0) {
          currentStepIndex--;
          renderCookingStep();
        }
      }
    }
  });

  timerStartBtn.addEventListener("click", toggleCookingTimer);
  timerResetBtn.addEventListener("click", resetCookingTimer);
  startCookingBtn.addEventListener("click", startCookingMode);

  // ---------------------------------------------------------------------------
  // Virtual Pantry Management
  // ---------------------------------------------------------------------------
  async function fetchPantryItems() {
    try {
      const res = await fetch("/api/pantry/items");
      if (!res.ok) return;
      pantryItems = await res.json();
      if (pantryBadge) {
        pantryBadge.textContent = pantryItems.length;
        pantryBadge.classList.toggle("hidden", pantryItems.length === 0);
      }
      renderPantryList();
      checkPantryExpiryAlerts();
    } catch (e) {
      console.debug("Fetch pantry items error:", e);
    }
  }

  function checkPantryExpiryAlerts() {
    const expiring = pantryItems.filter((i) => i.status === "expiring_soon" || i.status === "expired");
    if (expiring.length > 0) {
      const expiredCount = expiring.filter((i) => i.status === "expired").length;
      const soonCount = expiring.filter((i) => i.status === "expiring_soon").length;
      let msg = "⚠️ Expiry Alert: ";
      if (expiredCount > 0) msg += `${expiredCount} item(s) expired! `;
      if (soonCount > 0) msg += `${soonCount} item(s) expiring within 3 days.`;
      pantryAlertBanner.textContent = msg;
      pantryAlertBanner.classList.remove("hidden");
    } else {
      pantryAlertBanner.classList.add("hidden");
    }
  }

  function renderPantryList() {
    pantryItemsList.innerHTML = "";
    if (pantryItems.length === 0) {
      pantryItemsList.innerHTML = '<div class="empty-state">No items in your virtual pantry yet. Add some ingredients above!</div>';
      return;
    }

    pantryItems.forEach((item) => {
      const row = document.createElement("div");
      row.className = "pantry-item-row";

      let statusBadge = '<span class="badge-fresh">Fresh</span>';
      if (item.status === "expired") {
        statusBadge = '<span class="badge-expired">Expired</span>';
      } else if (item.status === "expiring_soon") {
        statusBadge = `<span class="badge-expiring">${item.days_left}d left</span>`;
      }

      row.innerHTML = `
        <div class="pantry-item-info">
          <span class="pantry-item-name">${escapeHtml(item.name)}</span>
          <span class="pantry-item-qty">${item.quantity} ${escapeHtml(item.unit)}</span>
          <span class="pantry-item-cat">${escapeHtml(item.category)}</span>
          ${statusBadge}
        </div>
        <div class="pantry-item-actions">
          <button class="btn btn-sm btn-outline btn-delete-pantry" data-id="${item.id}">&times;</button>
        </div>
      `;

      const delBtn = row.querySelector(".btn-delete-pantry");
      delBtn.addEventListener("click", async () => {
        await fetch(`/api/pantry/items/${item.id}`, { method: "DELETE" });
        fetchPantryItems();
      });

      pantryItemsList.appendChild(row);
    });
  }

  async function handleAddPantryItem() {
    const name = pantryItemName.value.trim();
    if (!name) return;

    try {
      await fetch("/api/pantry/items", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name,
          quantity: parseFloat(pantryItemQty.value) || 1.0,
          unit: pantryItemUnit.value.trim() || "pcs",
          category: pantryItemCat.value,
          expiry_date: pantryItemExpiry.value || null,
        }),
      });

      pantryItemName.value = "";
      pantryItemExpiry.value = "";
      fetchPantryItems();
    } catch (e) {
      console.error("Add pantry item error:", e);
    }
  }

  pantryAddBtn.addEventListener("click", handleAddPantryItem);

  pantrySearchRecipesBtn.addEventListener("click", () => {
    if (pantryItems.length === 0) {
      alert("Your pantry is empty! Add some ingredients first.");
      return;
    }
    searchIngredients = pantryItems.map((i) => i.name);
    renderChips();
    switchTab("search-view");
    searchRecipes();
  });

  loadFromPantryBtn.addEventListener("click", () => {
    if (pantryItems.length === 0) {
      alert("No items in pantry yet.");
      return;
    }
    searchIngredients = Array.from(new Set([...searchIngredients, ...pantryItems.map((i) => i.name)]));
    renderChips();
  });

  // ---------------------------------------------------------------------------
  // Meal Planner & 7-Day Grid
  // ---------------------------------------------------------------------------
  const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const SLOTS = ["breakfast", "lunch", "dinner", "snack"];

  async function fetchMealPlans() {
    try {
      const res = await fetch("/api/meal-plan");
      if (!res.ok) return;
      weeklyMealPlan = await res.json();
      renderMealPlanGrid();
    } catch (e) {
      console.debug("Fetch meal plan error:", e);
    }
  }

  function renderMealPlanGrid() {
    mealPlanGrid.innerHTML = "";

    DAYS.forEach((day) => {
      const dayCard = document.createElement("div");
      dayCard.className = "plan-day-card";

      const title = document.createElement("h3");
      title.className = "plan-day-title";
      title.textContent = day;
      dayCard.appendChild(title);

      const slotsContainer = document.createElement("div");
      slotsContainer.className = "plan-slots-container";

      SLOTS.forEach((slot) => {
        const item = weeklyMealPlan.find(
          (p) => p.day_of_week.toLowerCase() === day.toLowerCase() && p.meal_slot.toLowerCase() === slot.toLowerCase()
        );

        const slotDiv = document.createElement("div");
        slotDiv.className = "plan-slot-item";

        const label = document.createElement("div");
        label.className = "plan-slot-label";
        label.textContent = slot;
        slotDiv.appendChild(label);

        const content = document.createElement("div");
        content.className = "plan-slot-content";

        if (item) {
          content.innerHTML = `
            <span>${escapeHtml(item.recipe_title)}</span>
            <button class="btn btn-sm btn-outline btn-clear-slot" title="Remove">&times;</button>
          `;
          const clearBtn = content.querySelector(".btn-clear-slot");
          clearBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            await fetch(`/api/meal-plan/${day}/${slot}`, { method: "DELETE" });
            fetchMealPlans();
          });
          slotDiv.style.cursor = "pointer";
          slotDiv.addEventListener("click", () => loadRecipeDetail(item.recipe_id));
        } else {
          content.innerHTML = '<span class="plan-slot-empty">+ Unscheduled</span>';
        }

        slotDiv.appendChild(content);
        slotsContainer.appendChild(slotDiv);
      });

      dayCard.appendChild(slotsContainer);
      mealPlanGrid.appendChild(dayCard);
    });
  }

  function openAddToPlanModal() {
    if (!activeRecipeDetail) return;
    planModalRecipeTitle.textContent = activeRecipeDetail.title;
    planModal.classList.remove("hidden");
  }

  async function handleConfirmAddToPlan() {
    if (!activeRecipeDetail) return;
    const day = planDaySelect.value;
    const slot = planSlotSelect.value;
    const servings = parseInt(planServingsInput.value, 10) || 2;

    try {
      await fetch("/api/meal-plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          day_of_week: day,
          meal_slot: slot,
          recipe_id: activeRecipeDetail.id,
          recipe_title: activeRecipeDetail.title,
          servings: servings,
        }),
      });
      planModal.classList.add("hidden");
      alert(`Scheduled "${activeRecipeDetail.title}" for ${day} ${slot}!`);
    } catch (e) {
      console.error("Schedule error:", e);
    }
  }

  addToPlanBtn.addEventListener("click", openAddToPlanModal);
  closePlanModal.addEventListener("click", () => planModal.classList.add("hidden"));
  confirmAddPlanBtn.addEventListener("click", handleConfirmAddToPlan);

  clearAllPlansBtn.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear all scheduled meals for the week?")) return;
    await fetch("/api/meal-plan/clear", { method: "DELETE" });
    fetchMealPlans();
  });

  // ---------------------------------------------------------------------------
  // Shopping List Generator
  // ---------------------------------------------------------------------------
  async function generateShoppingList() {
    const subtract = shoppingSubtractPantry.checked;
    try {
      const res = await fetch(`/api/meal-plan/shopping-list?subtract_pantry=${subtract}`);
      if (!res.ok) return;
      const items = await res.json();
      renderShoppingListModal(items);
    } catch (e) {
      console.error("Shopping list error:", e);
    }
  }

  function renderShoppingListModal(items) {
    shoppingListItems.innerHTML = "";
    if (!items || items.length === 0) {
      shoppingListItems.innerHTML = '<div class="empty-state">No scheduled meals or all ingredients are already in your pantry!</div>';
      shoppingModal.classList.remove("hidden");
      return;
    }

    items.forEach((item) => {
      const div = document.createElement("div");
      div.className = `shopping-item ${item.in_stock ? "in-stock" : ""}`;
      div.innerHTML = `
        <span><strong>${escapeHtml(item.ingredient)}</strong> (needed for ${item.recipe_count} meal${item.recipe_count > 1 ? "s" : ""})</span>
        <span>${item.in_stock ? "✅ In Pantry" : "🛒 To Buy"}</span>
      `;
      shoppingListItems.appendChild(div);
    });

    shoppingModal.classList.remove("hidden");
  }

  generateShoppingListBtn.addEventListener("click", generateShoppingList);
  shoppingSubtractPantry.addEventListener("change", generateShoppingList);
  closeShoppingModal.addEventListener("click", () => shoppingModal.classList.add("hidden"));

  copyShoppingListBtn.addEventListener("click", () => {
    const textItems = [];
    shoppingListItems.querySelectorAll(".shopping-item").forEach((el) => {
      textItems.push(el.textContent.trim());
    });
    navigator.clipboard.writeText(textItems.join("\n")).then(() => {
      alert("Shopping list copied to clipboard!");
    });
  });

  // ---------------------------------------------------------------------------
  // Event Listeners
  // ---------------------------------------------------------------------------
  if (engineSelect) {
    engineSelect.addEventListener("change", async (e) => {
      await fetch("/api/system/set-engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: e.target.value }),
      });
      if (searchIngredients.length > 0) searchRecipes();
    });
  }

  const ingredientForm = document.getElementById("ingredient-form");
  if (ingredientForm) {
    ingredientForm.addEventListener("submit", (e) => {
      e.preventDefault();
      addSearchIngredient();
    });
  }

  addBtn.addEventListener("click", () => addSearchIngredient());
  ingredientInput.addEventListener("input", handleIngredientInput);
  ingredientInput.addEventListener("focus", handleIngredientInput);

  ingredientInput.addEventListener("keydown", (e) => {
    const isDropdownOpen = ingredientAutocompleteDropdown && !ingredientAutocompleteDropdown.classList.contains("hidden");

    if (isDropdownOpen && suggestionsList.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selectedSuggestionIndex = (selectedSuggestionIndex + 1) % suggestionsList.length;
        return;
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        selectedSuggestionIndex = (selectedSuggestionIndex - 1 + suggestionsList.length) % suggestionsList.length;
        return;
      } else if (e.key === "Tab" || e.key === "Enter") {
        e.preventDefault();
        const targetIdx = selectedSuggestionIndex >= 0 ? selectedSuggestionIndex : 0;
        if (suggestionsList[targetIdx]) {
          addSearchIngredient(suggestionsList[targetIdx].name);
        } else {
          addSearchIngredient();
        }
        return;
      } else if (e.key === "Escape") {
        e.preventDefault();
        closeAutocomplete();
        return;
      }
    }

    if (e.key === "Enter") {
      e.preventDefault();
      addSearchIngredient();
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".autocomplete-wrapper")) closeAutocomplete();
  });

  searchBtn.addEventListener("click", searchRecipes);
  clearBtn.addEventListener("click", () => {
    searchIngredients = [];
    currentSearchResults = [];
    renderChips();
    closeAutocomplete();
    setStatus(statusMessage, "");
    resultsSection.classList.add("hidden");
    recipeList.innerHTML = "";
    ingredientInput.value = "";
    ingredientInput.focus();
  });

  backBtn.addEventListener("click", () => {
    detailView.classList.add("hidden");
    searchView.classList.remove("hidden");
  });

  favoriteToggleBtn.addEventListener("click", toggleFavoriteRecipe);

  // Initial setup check
  checkInitialSetup();
  renderChips();
});
