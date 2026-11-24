// Entry point. Module scripts run after the document is parsed, so every
// module can look up its elements at import time.
import { initCooking } from "./cooking.js";
import { fetchFavorites, initFavorites } from "./favorites.js";
import { startHeartbeat } from "./heartbeat.js";
import { initNavigation } from "./navigation.js";
import { fetchPantryItems, initPantry } from "./pantry.js";
import { initPlanner } from "./planner.js";
import { initRecipeDetail } from "./recipe-detail.js";
import { focusIngredientInput, initSearch } from "./search.js";
import { initSetup } from "./setup.js";

startHeartbeat();
initNavigation();
initSearch();
initRecipeDetail();
initFavorites();
initCooking();
initPantry();
initPlanner();

initSetup({
  onReady: () => {
    focusIngredientInput();
    fetchPantryItems();
    fetchFavorites();
  },
});
