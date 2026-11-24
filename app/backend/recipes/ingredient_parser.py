"""Rule-based parser for recipe ingredient lines.

Turns "1 (14.5 oz) can diced tomatoes, drained" into quantity=1, unit="can",
package size 14.5 oz, name="tomato", preparation="diced drained". It also accepts the
records stored in ``recipes.ingredient_details`` ({"name", "quantity", "unit"}), where the
unit is usually the placeholder "portion" because the Food.com source has no units.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Canonical unit -> spellings (matched case-insensitively, except the T/t pair below).
_UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "tsp": ("teaspoon", "teaspoons", "tsp", "tsps", "tsp.", "teas", "ts"),
    "tbsp": ("tablespoon", "tablespoons", "tbsp", "tbsps", "tbsp.", "tbs", "tbs.", "tbl", "tbl.", "tblsp", "tbls"),
    "cup": ("cup", "cups", "c", "c."),
    "fl_oz": ("fl oz", "fl. oz", "fl. oz.", "fl oz.", "fluid ounce", "fluid ounces", "floz"),
    "oz": ("ounce", "ounces", "oz", "oz.", "ozs"),
    "lb": ("pound", "pounds", "lb", "lb.", "lbs", "lbs.", "#"),
    "g": ("gram", "grams", "g", "g.", "gr", "grs", "gm", "gms", "gramme", "grammes"),
    "kg": ("kilogram", "kilograms", "kg", "kg.", "kgs", "kilo", "kilos"),
    "mg": ("milligram", "milligrams", "mg"),
    "ml": ("milliliter", "milliliters", "millilitre", "millilitres", "ml", "ml.", "mls", "cc"),
    "cl": ("centiliter", "centilitre", "cl"),
    "dl": ("deciliter", "decilitre", "dl"),
    "l": ("liter", "liters", "litre", "litres", "l", "l.", "lt", "ltr"),
    "pint": ("pint", "pints", "pt", "pt.", "pts"),
    "quart": ("quart", "quarts", "qt", "qt.", "qts"),
    "gallon": ("gallon", "gallons", "gal", "gal.", "gals"),
    "pinch": ("pinch", "pinches"),
    "dash": ("dash", "dashes"),
    "drop": ("drop", "drops"),
    "smidgen": ("smidgen", "smidgens"),
    "clove": ("clove", "cloves"),
    "slice": ("slice", "slices"),
    "stick": ("stick", "sticks"),
    "sprig": ("sprig", "sprigs"),
    "bunch": ("bunch", "bunches"),
    "head": ("head", "heads"),
    "stalk": ("stalk", "stalks"),
    "rib": ("rib", "ribs"),
    "piece": ("piece", "pieces", "pc", "pcs"),
    "can": ("can", "cans", "tin", "tins"),
    "jar": ("jar", "jars"),
    "bottle": ("bottle", "bottles"),
    "box": ("box", "boxes"),
    "bag": ("bag", "bags"),
    "package": ("package", "packages", "pkg", "pkg.", "pkgs", "packet", "packets", "pack", "packs",
                "envelope", "envelopes", "sachet", "sachets"),
    "container": ("container", "containers", "tub", "tubs"),
    "carton": ("carton", "cartons"),
    "handful": ("handful", "handfuls"),
    "inch": ("inch", "inches", "in."),
    "fillet": ("fillet", "fillets"),
}
_UNIT_LOOKUP: dict[str, str] = {s: unit for unit, spellings in _UNIT_SPELLINGS.items() for s in spellings}
# Units that can follow the name in Food.com style data ("garlic cloves", "celery ribs").
_TRAILING_UNITS = {"clove": "garlic", "rib": "celery", "stalk": "celery"}
# Short forms that are ambiguous without a number in front ("c" in "c. sugar" vs a word).
_NEEDS_QUANTITY = {"c", "c.", "l", "l.", "g", "g.", "t", "in.", "pt", "#", "ts"}

MASS_UNITS_G = {"g": 1.0, "kg": 1000.0, "mg": 0.001, "oz": 28.3495, "lb": 453.592}
VOLUME_UNITS_ML = {
    "tsp": 4.92892, "tbsp": 14.7868, "cup": 236.588, "fl_oz": 29.5735, "ml": 1.0, "cl": 10.0,
    "dl": 100.0, "l": 1000.0, "pint": 473.176, "quart": 946.353, "gallon": 3785.41,
    "pinch": 0.31, "dash": 0.62, "drop": 0.05, "smidgen": 0.15,
}
CONTAINER_UNITS = {"can", "jar", "bottle", "box", "bag", "package", "container", "carton"}

_UNICODE_FRACTIONS = {
    "½": 0.5, "⅓": 1 / 3, "⅔": 2 / 3, "¼": 0.25, "¾": 0.75, "⅕": 0.2, "⅖": 0.4, "⅗": 0.6,
    "⅘": 0.8, "⅙": 1 / 6, "⅚": 5 / 6, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}
_FRACTION_PAIRS = {
    "½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4), "⅕": (1, 5), "⅖": (2, 5),
    "⅗": (3, 5), "⅘": (4, 5), "⅙": (1, 6), "⅚": (5, 6), "⅛": (1, 8), "⅜": (3, 8), "⅝": (5, 8),
    "⅞": (7, 8),
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "half": 0.5,
    "a half": 0.5, "a couple": 2, "couple": 2, "few": 3, "a few": 3, "several": 3,
}

# Words describing how the ingredient is prepared. They move to `preparation`.
_PREP_WORDS = {
    "chopped", "diced", "minced", "sliced", "grated", "shredded", "crushed", "cubed", "julienned",
    "peeled", "seeded", "deseeded", "cored", "halved", "quartered", "trimmed", "melted", "softened",
    "beaten", "whisked", "sifted", "drained", "rinsed", "divided", "packed", "toasted", "thawed",
    "mashed", "pitted", "zested", "juiced", "torn", "crumbled", "deveined", "hulled", "stemmed",
    "frozen", "scrubbed", "snipped", "slivered", "cut", "squeezed", "chilled", "warmed", "rinsed",
    "pureed", "puréed", "strained", "separated", "skinned", "boned", "flaked", "broken",
}
_PREP_ADVERBS = {"finely", "coarsely", "thinly", "thickly", "roughly", "lightly", "firmly", "loosely",
                 "well", "freshly", "very", "newly", "and", "or", "then"}
# Dropped from the name and not reported as preparation.
_DROP_WORDS = {"fresh", "freshly", "large", "medium", "small", "extra-large", "jumbo"}
_SIZE_WORDS = {"large", "medium", "small", "extra-large", "jumbo", "big", "medium-size", "medium-sized",
               "large-size", "small-size"}
# Multi-word names whose participle is part of the product, not preparation.
_PROTECTED_PHRASES = (
    "crushed red pepper", "shredded wheat", "whipped cream", "whipped topping", "cut green beans",
    "sliced almonds", "slivered almonds", "flaked almonds", "flaked coconut", "packed in",
    "frozen yogurt", "chopped clams", "minced clams", "toasted sesame oil", "toasted sesame seeds",
)
_KEEP_PLURAL = {
    "molasses", "hummus", "couscous", "asparagus", "grits", "swiss", "brussels", "lemongrass", "bass",
    "watercress", "citrus", "octopus", "cactus", "series", "species", "oats", "is", "gas", "anise",
    "chives", "cress", "schnapps", "glass", "jus", "tortellini", "pancreas", "haricots", "this",
    "fries", "baklava", "lentils",
}
_IRREGULAR_PLURALS = {
    "leaves": "leaf", "halves": "half", "loaves": "loaf", "knives": "knife", "calves": "calf",
    "potatoes": "potato", "tomatoes": "tomato", "mangoes": "mango", "avocados": "avocado",
    "chilies": "chili", "chillies": "chilli", "anchovies": "anchovy", "cookies": "cookie",
    "oats": "oat", "lentils": "lentil", "chives": "chive", "brownies": "brownie",
    "pies": "pie", "ties": "tie", "chives,": "chive", "geese": "goose", "teeth": "tooth",
}


@dataclass(frozen=True)
class ParsedIngredient:
    raw: str
    name: str
    quantity: float | None = None
    quantity_max: float | None = None
    unit: str | None = None
    size_quantity: float | None = None
    size_unit: str | None = None
    size_descriptor: str | None = None
    preparation: str | None = None

    @property
    def quantity_mean(self) -> float | None:
        if self.quantity is None:
            return None
        if self.quantity_max is None:
            return self.quantity
        return (self.quantity + self.quantity_max) / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw, "name": self.name, "quantity": self.quantity,
            "quantity_max": self.quantity_max, "unit": self.unit, "size_quantity": self.size_quantity,
            "size_unit": self.size_unit, "size_descriptor": self.size_descriptor,
            "preparation": self.preparation,
        }


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------

_NUM = r"(?:\d+(?:[.,]\d+)?)"
_FRAC = r"(?:\d+\s*/\s*\d+)"
# 1, 1.5, 1/2, 1 1/2, 1-1/2, 1½
_AMOUNT = rf"(?:{_NUM}\s*[-\s]\s*{_FRAC}|{_FRAC}|{_NUM}\s*[{''.join(_UNICODE_FRACTIONS)}]|[{''.join(_UNICODE_FRACTIONS)}]|{_NUM})"
_RANGE_SEP = r"\s*(?:-|–|—|to|or)\s*"
_QTY_RE = re.compile(rf"^\s*(?P<a>{_AMOUNT})(?:{_RANGE_SEP}(?P<b>{_AMOUNT}))?(?![\d/.])")


def _amount_to_float(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text:
        return None
    for char, value in _UNICODE_FRACTIONS.items():
        if char in text:
            rest = text.replace(char, "").strip(" -")
            return (float(rest) if rest else 0.0) + value
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-\s]\s*(\d+)\s*/\s*(\d+)", text)
    if m:
        whole, num, den = float(m[1]), float(m[2]), float(m[3])
        if den and num < den:
            return whole + num / den
        return None
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if m:
        den = float(m[2])
        return float(m[1]) / den if den else None
    try:
        return float(text)
    except ValueError:
        return None


def parse_quantity(value: Any) -> float | None:
    """Parse a quantity such as 2, "1 1/2", "1-1/2", "½", "1 -2" (lower bound of a range)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _clean_text(str(value)).lower()
    if not text or text in {"na", "none", "null"}:
        return None
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    lo, _hi, _end = _match_quantity(text)
    return lo


def _match_quantity(text: str) -> tuple[float | None, float | None, int]:
    m = _QTY_RE.match(text)
    if not m:
        return None, None, 0
    a = m["a"]
    b = m["b"]
    # "1-1/2" is a mixed number, not a range; _AMOUNT already prefers that reading.
    lo = _amount_to_float(a)
    hi = _amount_to_float(b) if b else None
    if lo is None:
        return None, None, 0
    if hi is not None and hi < lo:
        hi = None
    return lo, hi, m.end()


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def singularize(word: str) -> str:
    w = word.lower()
    if w in _KEEP_PLURAL and w not in _IRREGULAR_PLURALS:
        return w
    if w in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[w]
    if len(w) <= 3 or not w.endswith("s") or w.endswith(("ss", "us", "is", "'s")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    if re.search(r"(sh|ch|x|z)es$", w):
        return w[:-2]
    return w[:-1]


def _expand_unicode_fractions(text: str) -> str:
    for char, value in _UNICODE_FRACTIONS.items():
        if char in text:
            num, den = _FRACTION_PAIRS[char]
            text = re.sub(rf"(\d)?{char}", lambda m: f"{m[1] + ' ' if m[1] else ''}{num}/{den}", text)
    return text


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", _expand_unicode_fractions(text)).replace("⁄", "/")
    text = text.replace("&quot;", '"').replace("quot", " ")
    text = re.sub(r"[‐‑‒–—]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop size/freshness words, singularise the last word.

    Used for both parsed names and lookup keys so that "Fresh Tomatoes" and "tomato" meet.
    """
    text = _clean_text(name).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s%'-]", " ", text)
    words = [w.strip("-'") for w in text.split()]
    words = [w for w in words if w and w not in _DROP_WORDS and not re.fullmatch(r"[\d/.%-]+", w)]
    while words and words[0] in {"of", "a", "an", "the"}:
        words.pop(0)
    while words and words[-1] in {"of", "and", "or"}:
        words.pop()
    if not words:
        return ""
    words[-1] = singularize(words[-1])
    return " ".join(words)


def _split_prep_words(words: list[str]) -> tuple[list[str], list[str]]:
    """Separate preparation participles (and their adverbs) from the name words."""
    joined = " ".join(words)
    protected: set[int] = set()
    for phrase in _PROTECTED_PHRASES:
        idx = joined.find(phrase)
        if idx >= 0:
            start = len(joined[:idx].split())
            protected.update(range(start, start + len(phrase.split())))
    name_words: list[str] = []
    prep_words: list[str] = []
    pending_adverbs: list[str] = []
    for i, w in enumerate(words):
        if i in protected:
            name_words.extend(pending_adverbs)
            pending_adverbs = []
            name_words.append(w)
        elif w in _PREP_WORDS:
            prep_words.extend(pending_adverbs)
            pending_adverbs = []
            prep_words.append(w)
        elif w == "freshly":
            continue
        elif w in _PREP_ADVERBS and w not in {"and", "or"}:
            pending_adverbs.append(w)
        elif w in {"and", "or"} and prep_words and not name_words and i + 1 < len(words) and words[i + 1] in _PREP_WORDS:
            prep_words.append(w)
        else:
            name_words.extend(pending_adverbs)
            pending_adverbs = []
            name_words.append(w)
    name_words.extend(a for a in pending_adverbs if a != "freshly")
    return name_words, prep_words


def _lookup_unit(token: str, next_token: str | None, has_quantity: bool) -> tuple[str | None, int]:
    """Return (canonical unit, tokens consumed) for the token(s) at the head of the text."""
    if token in {"T", "Tbs", "Tb", "TB"}:
        return "tbsp", 1
    if token in {"t"}:
        return ("tsp", 1) if has_quantity else (None, 0)
    low = token.lower()
    if next_token is not None:
        two = f"{low} {next_token.lower()}"
        if two in _UNIT_LOOKUP:
            return _UNIT_LOOKUP[two], 2
    if low in _UNIT_LOOKUP:
        if low in _NEEDS_QUANTITY and not has_quantity:
            return None, 0
        return _UNIT_LOOKUP[low], 1
    return None, 0


_SIZE_IN_PARENS = re.compile(
    rf"\(\s*(?:about\s+|approx\.?\s+|approximately\s+)?(?P<q>{_AMOUNT})\s*[- ]?\s*(?P<u>[a-zA-Z. ]+?)\s*(?:each\s*)?\)"
)
_SIZE_INLINE = re.compile(rf"^(?P<q>{_AMOUNT})\s*-?\s*(?P<u>ounce|oz|pound|lb|g|gram|ml|inch)\b\.?\s*", re.I)
_JUICE_OF = re.compile(r"^(?:the\s+)?(?P<part>juice|zest|rind|peel)\s+(?:of|from)\s+(?P<rest>.+)$", re.I)
_PART_OF_SUFFIX = re.compile(r"^(?P<name>.+?),?\s+(?P<part>juice|zest|rind|peel)\s+of$", re.I)


def _strip_number_parenthetical(text: str) -> tuple[str, float | None, str | None, list[str]]:
    """Remove parentheticals. Ones with a number are measurements; others are notes."""
    size_q: float | None = None
    size_u: str | None = None
    notes: list[str] = []

    def repl(m: re.Match[str]) -> str:
        nonlocal size_q, size_u
        inner = m.group(0)[1:-1].strip()
        sm = _SIZE_IN_PARENS.fullmatch(m.group(0))
        if sm and size_q is None:
            unit_text = sm["u"].strip().rstrip(".").lower()
            unit = _UNIT_LOOKUP.get(unit_text) or _UNIT_LOOKUP.get(unit_text.split()[0])
            if unit:
                size_q = _amount_to_float(sm["q"])
                size_u = unit
                return " "
        if not re.search(r"\d", inner):
            notes.append(inner)
        return " "

    text = re.sub(r"\([^)]*\)", repl, text)
    return re.sub(r"\s+", " ", text).strip(), size_q, size_u, notes


def parse_ingredient(text: str) -> ParsedIngredient:
    """Parse one free-text ingredient line."""
    raw = text
    text = _clean_text(text)
    text, size_q, size_u, notes = _strip_number_parenthetical(text)

    # "Juice of 1 lemon" -> "1 lemon juice"
    part_prefix = None
    m = _JUICE_OF.match(text)
    if m:
        part_prefix = m["part"].lower()
        text = m["rest"]

    # Split off the note after the first comma, except "boneless, skinless".
    text = re.sub(r"\bboneless\s*,\s*skinless\b", "boneless skinless", text, flags=re.I)
    m = _PART_OF_SUFFIX.match(text)
    if m:
        base = m["name"].split()
        base[-1] = singularize(base[-1])
        text = f"{' '.join(base)} {m['part']}"
    head, _, tail = text.partition(",")
    tail = tail.strip()
    if tail.lower().endswith(("rind of", "zest of", "juice of", "peel of")):
        head_words = head.split()
        if head_words:
            head_words[-1] = singularize(head_words[-1])
        head = f"{' '.join(head_words)} {tail.rsplit(' ', 1)[0]}"
        tail = ""

    # "to taste", "for garnish", "optional" written without a comma
    trailing_note = None
    tm = re.search(r"\s+\b(to taste|for garnish|for serving|for dusting|optional|as needed|if desired)\b.*$", head, re.I)
    if tm:
        trailing_note = tm.group(0).strip()
        head = head[: tm.start()]

    s = head.strip()
    quantity: float | None = None
    quantity_max: float | None = None
    lower = s.lower()
    word_qty = re.match(r"^(a couple|a few|a half|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half|couple|few|several)\s+(?:of\s+)?", lower)
    lo, hi, end = _match_quantity(lower)
    if lo is not None:
        quantity, quantity_max = lo, hi
        s = s[end:].strip()
    elif word_qty:
        quantity = float(_NUMBER_WORDS[word_qty.group(1)])
        s = s[word_qty.end():].strip()

    # "2 x 400g cans", "1 x 400 g tin"
    s = re.sub(r"^[x×]\s*", "", s, flags=re.I)

    # Unit glued to the number ("250g") was left at the start of s.
    tokens = s.split()
    unit: str | None = None
    # Inline package size: "14-ounce can", "400g tin", "1-inch piece"
    sm = _SIZE_INLINE.match(s)
    if sm and quantity is not None:
        rest_tokens = s[sm.end():].split()
        follow_unit, used = _lookup_unit(rest_tokens[0], rest_tokens[1] if len(rest_tokens) > 1 else None, True) if rest_tokens else (None, 0)
        if follow_unit in CONTAINER_UNITS or follow_unit == "piece":
            size_q = _amount_to_float(sm["q"])
            size_u = _UNIT_LOOKUP.get(sm["u"].lower(), sm["u"].lower())
            unit = follow_unit
            tokens = rest_tokens[used:]
    if unit is None and tokens:
        unit, used = _lookup_unit(tokens[0], tokens[1] if len(tokens) > 1 else None, quantity is not None)
        tokens = tokens[used:]
        # "2 large eggs" style sizes between quantity and unit: "1 small bunch basil"
        if unit is None and len(tokens) > 1 and tokens[0].lower() in _SIZE_WORDS:
            u2, used2 = _lookup_unit(tokens[1], tokens[2] if len(tokens) > 2 else None, True)
            if u2 and u2 not in {"inch"}:
                unit = u2
                tokens = [tokens[0]] + tokens[1 + used2:]
    if tokens and tokens[0].lower() == "of":
        tokens = tokens[1:]
    # A container after the size: "1 (8 ounce) package cream cheese" already handled; "can (15 oz)" too.
    if unit is not None and unit in CONTAINER_UNITS and tokens:
        u2, used2 = _lookup_unit(tokens[0], tokens[1] if len(tokens) > 1 else None, True)
        if u2 in CONTAINER_UNITS:
            tokens = tokens[used2:]

    words = [w for w in re.sub(r"[^\w\s%'-]", " ", " ".join(tokens)).split()]
    words = [w for w in words if not re.fullmatch(r"[\d/.%-]+", w)]
    size_descriptor = next((w.lower() for w in words if w.lower() in _SIZE_WORDS), None)
    low_words = [w.lower() for w in words]
    name_words, prep_words = _split_prep_words(low_words)

    # Food.com style: count unit after the name ("garlic cloves", "celery ribs").
    if unit is None and len(name_words) >= 2:
        last = singularize(name_words[-1])
        if last in _TRAILING_UNITS and name_words[-2] == _TRAILING_UNITS[last]:
            unit = last
            name_words = name_words[:-1]

    name = normalize_name(" ".join(name_words))
    if part_prefix:
        name = f"{name} {part_prefix}".strip()

    prep_parts = []
    if prep_words:
        prep_parts.append(" ".join(prep_words))
    if tail:
        prep_parts.append(tail)
    if trailing_note:
        prep_parts.append(trailing_note)
    prep_parts.extend(notes)
    preparation = " ".join(p.strip() for p in prep_parts if p.strip()) or None
    if preparation:
        preparation = preparation.lower()

    return ParsedIngredient(
        raw=raw,
        name=name,
        quantity=quantity,
        quantity_max=quantity_max,
        unit=unit,
        size_quantity=size_q,
        size_unit=size_u,
        size_descriptor=size_descriptor,
        preparation=preparation,
    )


_PLACEHOLDER_UNITS = {"", "portion", "portions", "serving", "servings", "unit", "units", "each", "whole", "item"}


def parse_ingredient_detail(detail: dict[str, Any]) -> ParsedIngredient:
    """Parse an ``ingredient_details`` record. "portion" means the unit is unknown."""
    name_text = str(detail.get("name") or "")
    parsed = parse_ingredient(name_text)
    quantity = parse_quantity(detail.get("quantity"))
    unit_text = str(detail.get("unit") or "").strip()
    unit = parsed.unit
    if unit_text.lower() not in _PLACEHOLDER_UNITS:
        unit = canonical_unit(unit_text) or unit
    if quantity is None:
        quantity = parsed.quantity
    return ParsedIngredient(
        raw=name_text,
        name=parsed.name,
        quantity=quantity,
        quantity_max=parsed.quantity_max if parsed.quantity == quantity else None,
        unit=unit,
        size_quantity=parsed.size_quantity,
        size_unit=parsed.size_unit,
        size_descriptor=parsed.size_descriptor,
        preparation=parsed.preparation,
    )


def canonical_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    if unit in _UNIT_SPELLINGS:
        return unit
    if unit in {"T", "Tbs", "TB"}:
        return "tbsp"
    if unit == "t":
        return "tsp"
    return _UNIT_LOOKUP.get(unit.strip().lower())
