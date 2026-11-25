// Small DOM helpers. Everything that renders data from the API or the user
// goes through textContent, never through HTML strings.

export function byId(id) {
  return document.getElementById(id);
}

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

export function emptyState(text) {
  return el("div", "empty-state", text);
}

export function setStatus(element, text, type = "info") {
  if (!text) {
    element.textContent = "";
    element.className = "status-message hidden";
    return;
  }
  element.textContent = text;
  element.className = `status-message ${type}`;
}

export function formatErrorMessage(err) {
  if (!err) return "An unexpected error occurred.";
  const isNetworkError =
    err.name === "TypeError" && typeof err.message === "string" && err.message.includes("fetch");
  if (isNetworkError) {
    return "Cannot connect to server. Ensure uvicorn backend is running.";
  }
  return err.message || "An unexpected error occurred.";
}
