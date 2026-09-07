// PantrySense — Vanilla JavaScript Client

document.addEventListener("DOMContentLoaded", () => {
  // Tab Heartbeat Tracking (Robust against background tab throttling)
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

  // Send immediately and periodically every 2 seconds
  sendHeartbeat(true);
  const heartbeatInterval = setInterval(() => sendHeartbeat(), 2000);

  // Web Worker timer to prevent browsers from throttling background tabs
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
    console.debug("Web worker heartbeat fallback to standard interval:", e);
  }

  // Send heartbeat on user interactions and tab visibility changes
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      sendHeartbeat(true);
    }
  });
  window.addEventListener("focus", () => sendHeartbeat(true));
  window.addEventListener("pageshow", () => sendHeartbeat(true));
  window.addEventListener("click", () => sendHeartbeat(false));
  window.addEventListener("keydown", () => sendHeartbeat(false));

  // Notify server when tab actually closes or navigates away
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

  // State
  let ingredients = [];
  let currentRecipes = [];

  // DOM Elements - Header Actions
  const engineSelect = document.getElementById("engine-select");

  // DOM Elements - Views
  const setupView = document.getElementById("setup-view");
  const searchView = document.getElementById("search-view");
  const detailView = document.getElementById("detail-view");

  // DOM Elements - Setup View
  const setupProgressBar = document.getElementById("setup-progress-bar");
  const setupStatusText = document.getElementById("setup-status-text");
  const setupProgressPercent = document.getElementById("setup-progress-percent");

  // DOM Elements - Search View
  const ingredientInput = document.getElementById("ingredient-input");
  const ingredientAutocompleteDropdown = document.getElementById("ingredient-autocomplete-dropdown");
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

  let setupPollInterval = null;

  // --- Initial Setup & State Synchronization ---

  async function checkInitialSetup() {
    try {
      const res = await fetch("/api/system/setup-status");
      if (!res.ok) return;

      const data = await res.json();

      // Configure Hardware options based on detected GPU
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
        // Show Automatic Setup View
        searchView.classList.add("hidden");
        detailView.classList.add("hidden");
        setupView.classList.remove("hidden");

        if (!data.is_running) {
          const chosenEngine = engineSelect ? engineSelect.value : (data.cuda_available ? "cuda" : "onnx");
          autoStartSetup(chosenEngine);
        } else {
          pollSetupProgress();
        }
      } else {
        // App is already initialized with dataset - enter Search View immediately
        setupView.classList.add("hidden");
        detailView.classList.add("hidden");
        searchView.classList.remove("hidden");
        ingredientInput.focus();
      }
    } catch (err) {
      console.warn("Could not check setup status:", err);
    }
  }

  function updateStepperProgress(currentStep, progress, message) {
    const prog = Math.round(progress || 0);
    setupProgressBar.style.width = `${prog}%`;
    setupProgressPercent.textContent = `${prog}%`;
    setupStatusText.textContent = message || "Processing setup...";

    for (let i = 1; i <= 8; i++) {
      const el = document.getElementById(`step-item-${i}`);
      if (!el) continue;

      if (i < currentStep) {
        el.className = "stepper-item completed";
      } else if (i === currentStep) {
        el.className = "stepper-item active";
      } else {
        el.className = "stepper-item";
      }
    }
  }

  function autoStartSetup(engine) {
    setupProgressBar.style.width = "5%";
    setupProgressPercent.textContent = "5%";
    setupStatusText.textContent = "Step 1/8: Initializing recipe database schema...";

    const step1 = document.getElementById("step-item-1");
    if (step1) step1.className = "stepper-item active";

    fetch("/api/system/setup-init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: engine || "onnx",
        dataset_limit: 380000,
      }),
    })
      .then((res) => res.json())
      .then(() => pollSetupProgress())
      .catch((e) => {
        console.error("Setup auto-init error:", e);
        setupStatusText.textContent = "Failed to start automatic setup. Retrying in 3s...";
        setTimeout(() => autoStartSetup(engine), 3000);
      });
  }

  function pollSetupProgress() {
    if (setupPollInterval) {
      clearInterval(setupPollInterval);
    }

    setupPollInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/system/setup-status");
        if (!res.ok) return;

        const data = await res.json();
        updateStepperProgress(data.current_step || 1, data.progress || 0, data.message);

        // Error Handling
        if (data.status === "error") {
          clearInterval(setupPollInterval);
          setupPollInterval = null;
          setupStatusText.textContent = data.message || "Setup encountered an error.";
          return;
        }

        // Complete state
        const isComplete = data.is_ready || data.progress >= 100 || data.status === "ready";
        if (isComplete) {
          clearInterval(setupPollInterval);
          setupPollInterval = null;

          setupProgressBar.style.width = "100%";
          setupProgressPercent.textContent = "100%";

          for (let i = 1; i <= 8; i++) {
            const el = document.getElementById(`step-item-${i}`);
            if (el) el.className = "stepper-item completed";
          }

          setupStatusText.textContent = "Setup complete! Opening PantrySense...";

          setTimeout(() => {
            setupView.classList.add("hidden");
            searchView.classList.remove("hidden");
            ingredientInput.focus();
          }, 600);
        }
      } catch (e) {
        console.error("Error polling setup:", e);
      }
    }, 300);
  }

  async function handleEngineChange(newEngine) {
    try {
      await fetch("/api/system/set-engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: newEngine }),
      });
      // Re-trigger search if ingredients are present
      if (ingredients.length > 0) {
        searchRecipes();
      }
    } catch (e) {
      console.error("Failed to switch engine:", e);
    }
  }

  // --- Helper Functions ---

  function formatErrorMessage(err) {
    if (!err) return "An unexpected error occurred.";
    if (err.name === "TypeError" && (err.message === "Failed to fetch" || (typeof err.message === "string" && err.message.includes("fetch")))) {
      return "Cannot connect to server. Please ensure the backend server is running (python -m uvicorn app.backend.main:app).";
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

  // --- Controlled Ingredient Autocomplete & Spellchecker ---
  let suggestionsList = [];
  let selectedSuggestionIndex = -1;
  let autocompleteDebounceTimer = null;

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function closeAutocomplete() {
    if (ingredientAutocompleteDropdown) {
      ingredientAutocompleteDropdown.innerHTML = "";
      ingredientAutocompleteDropdown.classList.add("hidden");
    }
    suggestionsList = [];
    selectedSuggestionIndex = -1;
  }

  function updateActiveSuggestion() {
    if (!ingredientAutocompleteDropdown) return;
    const items = ingredientAutocompleteDropdown.querySelectorAll(".suggestion-item");
    items.forEach((el, idx) => {
      if (idx === selectedSuggestionIndex) {
        el.classList.add("active");
        el.scrollIntoView({ block: "nearest" });
      } else {
        el.classList.remove("active");
      }
    });
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
      div.setAttribute("data-index", idx);

      const mainDiv = document.createElement("div");
      mainDiv.className = "suggestion-main";

      const nameSpan = document.createElement("span");
      nameSpan.className = "suggestion-text";

      // Highlight matching substring if not a typo correction
      const name = item.name;
      const lowerName = name.toLowerCase();
      const lowerQ = query.toLowerCase().trim();
      const matchPos = lowerName.indexOf(lowerQ);

      if (matchPos >= 0 && lowerQ.length > 0 && !item.is_correction) {
        const before = name.substring(0, matchPos);
        const match = name.substring(matchPos, matchPos + lowerQ.length);
        const after = name.substring(matchPos + lowerQ.length);
        nameSpan.innerHTML = `${escapeHtml(before)}<span class="suggestion-match">${escapeHtml(match)}</span>${escapeHtml(after)}`;
      } else {
        nameSpan.textContent = name;
      }

      mainDiv.appendChild(nameSpan);

      if (item.is_correction) {
        const typoBadge = document.createElement("span");
        typoBadge.className = "suggestion-typo-badge";
        typoBadge.innerHTML = "✨ Did you mean?";
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
        e.preventDefault(); // Prevent input blur from dropping click
        addIngredient(item.name);
      });

      div.addEventListener("mouseenter", () => {
        selectedSuggestionIndex = idx;
        updateActiveSuggestion();
      });

      ingredientAutocompleteDropdown.appendChild(div);
    });

    ingredientAutocompleteDropdown.classList.remove("hidden");
  }

  function getActiveToken() {
    if (!ingredientInput) return "";
    const raw = ingredientInput.value || "";
    const parts = raw.split(/[,;\n]+/);
    return parts[parts.length - 1].trim();
  }

  async function fetchSuggestions(query) {
    const trimmed = (query || "").trim();
    if (!trimmed || trimmed.length < 1) {
      closeAutocomplete();
      return;
    }

    try {
      const res = await fetch(`/api/ingredients/suggest?q=${encodeURIComponent(trimmed)}&limit=8`);
      if (!res.ok) return;
      const data = await res.json();
      const currentToken = getActiveToken();
      // Only render if current active token still matches query
      if (currentToken.toLowerCase() === trimmed.toLowerCase() || ingredientInput.value.trim().toLowerCase() === trimmed.toLowerCase()) {
        renderSuggestions(trimmed, data.suggestions || []);
      }
    } catch (e) {
      console.debug("Ingredient suggestion fetch error:", e);
    }
  }

  function handleIngredientInput() {
    if (autocompleteDebounceTimer) {
      clearTimeout(autocompleteDebounceTimer);
    }
    const token = getActiveToken();
    if (!token) {
      closeAutocomplete();
      return;
    }
    autocompleteDebounceTimer = setTimeout(() => {
      fetchSuggestions(token);
    }, 100);
  }

  function addIngredient(customValue = null) {
    const rawValue = (customValue !== null ? customValue : ingredientInput.value).trim();
    if (!rawValue) return;

    // Support single ingredient or comma/semicolon/newline separated list
    const items = rawValue
      .split(/[,;\n]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    let addedAny = false;
    for (const item of items) {
      const exists = ingredients.some(
        (existing) => existing.toLowerCase() === item.toLowerCase()
      );
      if (!exists) {
        ingredients.push(item);
        addedAny = true;
      }
    }

    if (addedAny) {
      renderChips();
    }

    ingredientInput.value = "";
    closeAutocomplete();
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
    closeAutocomplete();
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

      const badgesGroup = document.createElement("div");
      badgesGroup.className = "badges-group";

      const coveragePercent = Math.round(recipe.coverage * 100);
      const badge = document.createElement("span");
      badge.className = "coverage-badge";
      badge.textContent = `${coveragePercent}% match`;
      badgesGroup.appendChild(badge);

      if (recipe.ml_score !== null && recipe.ml_score !== undefined) {
        const mlBadge = document.createElement("span");
        mlBadge.className = "ml-badge";
        mlBadge.title = "Learning-to-Rank Score";
        mlBadge.textContent = `★ ${recipe.ml_score.toFixed(2)}`;
        badgesGroup.appendChild(mlBadge);
      }

      header.appendChild(title);
      header.appendChild(badgesGroup);

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

    setStatus(statusMessage, "Searching & ranking matching recipes...", "info");
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
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      currentRecipes = data.recipes || [];
      renderRecipeList(currentRecipes);
    } catch (err) {
      console.error("Search failed:", err);
      setStatus(
        statusMessage,
        formatErrorMessage(err),
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
        formatErrorMessage(err),
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

  if (engineSelect) {
    engineSelect.addEventListener("change", (e) => handleEngineChange(e.target.value));
  }

  const ingredientForm = document.getElementById("ingredient-form");
  if (ingredientForm) {
    ingredientForm.addEventListener("submit", (e) => {
      e.preventDefault();
      addIngredient();
    });
  }

  addBtn.addEventListener("click", () => addIngredient());

  ingredientInput.addEventListener("input", handleIngredientInput);
  ingredientInput.addEventListener("focus", handleIngredientInput);

  ingredientInput.addEventListener("keydown", (e) => {
    const isDropdownOpen = ingredientAutocompleteDropdown && !ingredientAutocompleteDropdown.classList.contains("hidden");

    if (isDropdownOpen && suggestionsList.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selectedSuggestionIndex = (selectedSuggestionIndex + 1) % suggestionsList.length;
        updateActiveSuggestion();
        return;
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        selectedSuggestionIndex = (selectedSuggestionIndex - 1 + suggestionsList.length) % suggestionsList.length;
        updateActiveSuggestion();
        return;
      } else if (e.key === "Tab") {
        e.preventDefault();
        const targetIdx = selectedSuggestionIndex >= 0 ? selectedSuggestionIndex : 0;
        if (suggestionsList[targetIdx]) {
          addIngredient(suggestionsList[targetIdx].name);
        }
        return;
      } else if (e.key === "Escape") {
        e.preventDefault();
        closeAutocomplete();
        return;
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (selectedSuggestionIndex >= 0 && suggestionsList[selectedSuggestionIndex]) {
          addIngredient(suggestionsList[selectedSuggestionIndex].name);
        } else {
          addIngredient();
        }
        return;
      }
    }

    if (e.key === "Enter") {
      e.preventDefault();
      addIngredient();
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".autocomplete-wrapper")) {
      closeAutocomplete();
    }
  });

  searchBtn.addEventListener("click", searchRecipes);
  clearBtn.addEventListener("click", clearAll);
  backBtn.addEventListener("click", backToResults);

  // Initial setup check
  checkInitialSetup();
  renderChips();
});
