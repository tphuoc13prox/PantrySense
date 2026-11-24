import * as api from "./api.js";
import { byId, el, emptyState } from "./dom.js";
import { onTabShown, switchTab } from "./navigation.js";
import { searchRecipes, setSearchIngredients } from "./search.js";
import { state } from "./state.js";

const pantryBadge = byId("pantry-badge");
const alertBanner = byId("pantry-alert-banner");
const itemsList = byId("pantry-items-list");

const form = {
  name: byId("pantry-item-name"),
  qty: byId("pantry-item-qty"),
  unit: byId("pantry-item-unit"),
  category: byId("pantry-item-cat"),
  expiry: byId("pantry-item-expiry"),
};

export async function fetchPantryItems() {
  try {
    const items = await api.listPantryItems();
    if (!items) return;
    state.pantryItems = items;
    pantryBadge.textContent = items.length;
    pantryBadge.classList.toggle("hidden", items.length === 0);
    renderPantryList();
    updateExpiryBanner();
  } catch (err) {
    console.debug("Fetch pantry items error:", err);
  }
}

function updateExpiryBanner() {
  const expiredCount = state.pantryItems.filter((i) => i.status === "expired").length;
  const soonCount = state.pantryItems.filter((i) => i.status === "expiring_soon").length;
  if (expiredCount === 0 && soonCount === 0) {
    alertBanner.classList.add("hidden");
    return;
  }
  let msg = "⚠️ Expiry Alert: ";
  if (expiredCount > 0) msg += `${expiredCount} item(s) expired! `;
  if (soonCount > 0) msg += `${soonCount} item(s) expiring within 3 days.`;
  alertBanner.textContent = msg;
  alertBanner.classList.remove("hidden");
}

function statusBadge(item) {
  if (item.status === "expired") return el("span", "badge-expired", "Expired");
  if (item.status === "expiring_soon") return el("span", "badge-expiring", `${item.days_left}d left`);
  return el("span", "badge-fresh", "Fresh");
}

function renderPantryRow(item) {
  const row = el("div", "pantry-item-row");

  const info = el("div", "pantry-item-info");
  info.append(
    el("span", "pantry-item-name", item.name),
    el("span", "pantry-item-qty", `${item.quantity} ${item.unit ?? ""}`),
    el("span", "pantry-item-cat", item.category),
    statusBadge(item)
  );

  const actions = el("div", "pantry-item-actions");
  const deleteBtn = el("button", "btn btn-sm btn-outline btn-delete-pantry", "×");
  deleteBtn.type = "button";
  deleteBtn.addEventListener("click", async () => {
    await api.deletePantryItem(item.id);
    fetchPantryItems();
  });
  actions.appendChild(deleteBtn);

  row.append(info, actions);
  return row;
}

function renderPantryList() {
  if (state.pantryItems.length === 0) {
    itemsList.replaceChildren(
      emptyState("No items in your virtual pantry yet. Add some ingredients above!")
    );
    return;
  }
  itemsList.replaceChildren(...state.pantryItems.map(renderPantryRow));
}

async function addItem() {
  const name = form.name.value.trim();
  if (!name) return;

  try {
    await api.addPantryItem({
      name,
      quantity: parseFloat(form.qty.value) || 1.0,
      unit: form.unit.value.trim() || "pcs",
      category: form.category.value,
      expiry_date: form.expiry.value || null,
    });
    form.name.value = "";
    form.expiry.value = "";
    fetchPantryItems();
  } catch (err) {
    console.error("Add pantry item error:", err);
  }
}

function searchWithPantry() {
  if (state.pantryItems.length === 0) {
    alert("Your pantry is empty! Add some ingredients first.");
    return;
  }
  setSearchIngredients(state.pantryItems.map((i) => i.name));
  switchTab("search-view");
  searchRecipes();
}

export function initPantry() {
  byId("pantry-add-form").addEventListener("submit", (e) => e.preventDefault());
  byId("pantry-add-btn").addEventListener("click", addItem);
  byId("pantry-search-recipes-btn").addEventListener("click", searchWithPantry);
  onTabShown("pantry-view", fetchPantryItems);
}
