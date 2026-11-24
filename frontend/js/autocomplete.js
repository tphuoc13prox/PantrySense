import { suggestIngredients } from "./api.js";
import { el } from "./dom.js";

const DEBOUNCE_MS = 100;

let input = null;
let dropdown = null;
let onPick = () => {};

let suggestions = [];
let selectedIndex = -1;
let debounceTimer = null;

export function closeAutocomplete() {
  if (dropdown) {
    dropdown.replaceChildren();
    dropdown.classList.add("hidden");
  }
  suggestions = [];
  selectedIndex = -1;
}

function renderSuggestion(item) {
  const row = el("div", "suggestion-item");
  row.setAttribute("role", "option");

  const main = el("div", "suggestion-main");
  main.appendChild(el("span", "suggestion-text", item.name));
  if (item.is_correction) {
    main.appendChild(el("span", "suggestion-typo-badge", "✨ Did you mean?"));
  }
  row.appendChild(main);

  const recipeCount = item.frequency || item.count || 0;
  if (recipeCount > 0) {
    row.appendChild(el("span", "suggestion-count", `${recipeCount.toLocaleString()} recipes`));
  }

  // mousedown fires before the input's blur, so the click is not lost.
  row.addEventListener("mousedown", (e) => {
    e.preventDefault();
    onPick(item.name);
  });
  return row;
}

function renderSuggestions(items) {
  if (!dropdown) return;
  if (!items || items.length === 0) {
    closeAutocomplete();
    return;
  }
  suggestions = items;
  selectedIndex = -1;
  dropdown.replaceChildren(...items.map(renderSuggestion));
  dropdown.classList.remove("hidden");
}

// The input accepts comma separated lists; only the last entry is completed.
function getActiveToken() {
  const parts = input.value.split(/[,;\n]+/);
  return parts[parts.length - 1].trim();
}

async function fetchSuggestions(query) {
  try {
    const data = await suggestIngredients(query);
    if (data) renderSuggestions(data.suggestions || []);
  } catch (err) {
    console.debug("Suggestion error:", err);
  }
}

function handleInput() {
  if (debounceTimer) clearTimeout(debounceTimer);
  const token = getActiveToken();
  if (!token) {
    closeAutocomplete();
    return;
  }
  debounceTimer = setTimeout(() => fetchSuggestions(token), DEBOUNCE_MS);
}

// Returns true when the key was consumed by the open dropdown.
export function handleAutocompleteKey(e) {
  const isOpen = dropdown && !dropdown.classList.contains("hidden");
  if (!isOpen || suggestions.length === 0) return false;

  switch (e.key) {
    case "ArrowDown":
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % suggestions.length;
      return true;
    case "ArrowUp":
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + suggestions.length) % suggestions.length;
      return true;
    case "Tab":
    case "Enter": {
      e.preventDefault();
      const picked = suggestions[selectedIndex >= 0 ? selectedIndex : 0];
      onPick(picked ? picked.name : null);
      return true;
    }
    case "Escape":
      e.preventDefault();
      closeAutocomplete();
      return true;
    default:
      return false;
  }
}

export function initAutocomplete({ inputEl, dropdownEl, onSelect }) {
  input = inputEl;
  dropdown = dropdownEl;
  onPick = onSelect;

  input.addEventListener("input", handleInput);
  input.addEventListener("focus", handleInput);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".autocomplete-wrapper")) closeAutocomplete();
  });
}
