"""Dietary tags and allergen detection from ingredient names.

Each ingredient is matched against phrase rules, most specific first. A matched phrase is
blanked out so a generic rule cannot fire on part of it: "coconut milk" matches its own
rule and the plain "milk" rule no longer sees it. Qualifiers such as "gluten-free" or
"vegan" then remove the allergens they rule out for that ingredient.

Two levels are kept: "contains" (the usual retail form has it) and "may_contain" (common
formulations do, or cross-contact is common, e.g. oats and gluten). The public functions
report both together. Missing an allergen can hurt someone; a false warning only hides a
recipe, so the rules lean towards warning. This is a heuristic on ingredient names, not a
substitute for reading labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ALLERGENS = ("dairy", "eggs", "fish", "gluten", "peanuts", "sesame", "shellfish", "soy", "tree_nuts")


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    contains: frozenset[str] = frozenset()
    may_contain: frozenset[str] = frozenset()
    non_vegetarian: bool = False
    non_vegan: bool = False
    high_carb: bool = False


RULES: list[Rule] = []


def _r(phrases: str, contains: str = "", may: str = "", nv: bool = False, nvg: bool = False,
       carb: bool = False) -> None:
    """Register phrases (separated by "|"). Each phrase matches whole words, optional plural."""
    # Longest first so "coconut milk" is consumed before "coconut".
    for phrase in sorted({p.strip() for p in phrases.split("|") if p.strip()}, key=len, reverse=True):
        words = [re.escape(w) for w in phrase.split()]
        body = r"[\s-]+".join(words)
        pattern = re.compile(rf"(?<![\w-]){body}(?:e?s)?(?![\w-])")
        RULES.append(Rule(
            pattern=pattern,
            contains=frozenset(contains.split()),
            may_contain=frozenset(may.split()),
            non_vegetarian=nv,
            non_vegan=nvg or nv or bool(set(contains.split()) & {"dairy", "eggs", "fish", "shellfish"}),
            high_carb=carb,
        ))


# ---------------------------------------------------------------------------
# Specific phrases first: exceptions and hidden allergens.
# ---------------------------------------------------------------------------

# Not what the generic word suggests.
_r("cream of tartar|eggplant|egg replacer|eggless|nutmeg|butternut|butternut squash|butter bean|"
   "butter lettuce|butterhead lettuce|water chestnut|coconut|coconut water|coconut oil|coconut flour|"
   "nutritional yeast|cocoa butter|shea butter|apple butter|pumpkin butter|fruit butter|creme de cacao|"
   "creme de menthe|green creme de|white creme de|cream soda|kidney bean|crab apple|oyster mushroom|"
   "lobster mushroom|chicken of the woods|hen of the woods|sweetbread|ginger ale|root beer|"
   "milk thistle|milkweed|buckwheat|rice flour|corn flour|cornflour|tapioca flour|potato flour|"
   "chickpea flour|cassava flour|teff flour|sorghum flour|millet flour|coconut flour|masa harina|"
   "corn tortilla|rice noodle|rice vermicelli|glass noodle|cellophane noodle|bean thread|shirataki|"
   "rice paper|corn starch|cornstarch|potato starch|wheat-free|breadfruit|poultry seasoning|"
   "vegetable broth|vegetable stock|vegetable bouillon|mushroom broth|veggie broth|soda water|"
   "jackfruit|cream of coconut|coconut cream|coconut yogurt|coconut milk|light coconut milk|"
   "unsweetened coconut milk|lite coconut milk|rice milk|hemp milk|flax milk|pea milk|"
   "vegan sugar|sugar snap|seitan-free|peanut-free|nut-free|tree nut-free|coconut butter|"
   "coconut creamer|coconut whipped cream|artichoke heart|hearts of palm|heart of palm|palm heart|"
   "celery heart|romaine heart|duck sauce|chicken-fried|beefsteak tomato|sweet potato|jackfruit")
_r("celery rib|celery stalk|celery heart|rib of celery|rhubarb stalk")
_r("stuffing|stuffing mix|herb seasoned stuffing|crouton|garlic crouton|seasoned crouton", contains="gluten",
   may="dairy", nvg=True, carb=True)
_r("hamburger bun|hot dog bun|hamburger roll|hot dog roll|hamburger bread", contains="gluten", carb=True)
# Plant milks and butters: keep the plant allergen, drop dairy.
_r("almond milk|unsweetened almond milk|almond butter|almond cream|almond yogurt|almond cheese|"
   "cashew milk|cashew cream|cashew butter|cashew cheese|hazelnut milk|macadamia milk|walnut milk",
   contains="tree_nuts")
_r("soy milk|soymilk|soya milk|soy yogurt|soy cheese|soy creamer|soy butter|tofu cream cheese",
   contains="soy")
_r("oat milk|oat cream", may="gluten")
_r("peanut butter|peanut oil|peanut sauce|satay sauce|satay|groundnut|groundnut oil|arachis oil|"
   "peanut butter chip|peanut brittle", contains="peanuts")
_r("chocolate-covered peanut|chocolate covered peanut|reese's|peanut m&m", contains="peanuts dairy",
   may="soy")
_r("sunflower butter|sunflower seed butter|seed butter|soy nut butter")
_r("nut butter|mixed nut|nuts mixed", contains="tree_nuts", may="peanuts")
# Hidden allergens in compound ingredients.
_r("pesto|basil pesto|pesto sauce|genovese pesto", contains="tree_nuts dairy")
_r("worcestershire|worcestershire sauce", contains="fish", may="gluten", nv=True)
_r("fish sauce|thai fish sauce|nam pla|nuoc mam|anchovy paste|bonito|bonito flake|dashi|katsuobushi",
   contains="fish", nv=True)
_r("caesar|caesar dressing|caesar salad dressing|kittencals famous caesar", contains="fish eggs dairy", nv=True)
_r("caesar-flavor crouton|caesar crouton", contains="gluten", may="dairy", nvg=True)
_r("oyster sauce|shrimp paste|belacan|shrimp sauce|dried shrimp|crab paste", contains="shellfish", nv=True)
_r("green curry paste|red curry paste|yellow curry paste|thai curry paste|curry paste|massaman curry paste",
   may="shellfish fish", nv=True)
_r("imitation crab|imitation crabmeat|surimi|crab stick", contains="fish", may="shellfish eggs gluten", nv=True)
_r("ghee|clarified butter|butter oil|whey|whey protein|casein|caseinate|sodium caseinate|lactose|"
   "milk solid|milk powder|buttermilk|butterscotch|butterscotch chip", contains="dairy")
_r("nougat|torrone", contains="tree_nuts eggs", may="peanuts")
_r("marzipan|almond paste|amaretti|almond extract|praline", contains="tree_nuts")
_r("frangipane|gianduja|nutella|chocolate hazelnut spread|mock nutella spread|baklava|macaron",
   contains="tree_nuts", may="dairy eggs", nvg=True)
_r("amaretto|amaretto liqueur|frangelico|nocino|disaronno", may="tree_nuts")
_r("mayonnaise|mayo|aioli|tartar sauce|remoulade|miracle whip|coleslaw dressing", contains="eggs")
_r("hollandaise|bearnaise|thousand island dressing|ranch dressing", contains="eggs dairy")
_r("meringue|meringue powder|egg wash|albumen|eggnog|custard|custard powder|zabaglione|"
   "marshmallow creme|marshmallow fluff|lemon curd|egg noodle", contains="eggs", may="dairy")
_r("egg substitute|egg beater|liquid egg", contains="eggs")
_r("vegan mayo|vegan mayonnaise|eggless mayonnaise|just mayo", may="soy")
_r("tahini|sesame|sesame seed|sesame oil|sesame paste|halva|halvah|gomasio|benne seed|hummus|za'atar|"
   "zaatar|furikake|black sesame", contains="sesame")
_r("soy sauce|shoyu|soya sauce|light soy sauce|dark soy sauce|teriyaki|teriyaki sauce|hoisin|hoisin sauce|"
   "ponzu|kecap manis|black bean sauce|stir-fry sauce|stir fry sauce|low sodium soy|low-sodium soy",
   contains="soy gluten")
_r("tamari|miso|white miso|red miso|miso paste|shiro miso", contains="soy", may="gluten")
_r("tofu|bean curd|tempeh|edamame|soybean|soy bean|natto|textured vegetable protein|tvp|yuba|"
   "soy protein|soy flour|soy lecithin|soya|soy", contains="soy")
_r("lecithin|cooking spray|nonstick cooking spray|canola oil cooking|olive oil flavored|"
   "vegetable cooking spray|pam cooking spray", may="soy")
_r("milk chocolate|white chocolate|white chocolate chip|milk chocolate chip|chocolate milk|"
   "hot chocolate mix|hot cocoa mix", contains="dairy", may="soy")
_r("chocolate chip|semisweet chocolate|semi-sweet chocolate|dark chocolate|bittersweet chocolate|"
   "chocolate morsel|chocolate bar|chocolate-covered|chocolate covered|chocolate candy|chocolate|"
   "carob chip|candy bar|m&m", may="dairy soy", nvg=True)
_r("cocoa|cocoa powder|unsweetened cocoa|baking cocoa|cacao|cacao nib|unsweetened chocolate|"
   "baking chocolate")
_r("margarine|butter substitute|vegetable spread|buttery spread|oleo|i can't believe it's not butter",
   may="dairy soy", nvg=True)
_r("non-dairy creamer|nondairy creamer|non-dairy topping|coffee creamer|creamer|coffee whitener",
   may="dairy soy", nvg=True)
_r("cool whip|whipped topping|frozen whipped topping|dream whip", contains="dairy", may="soy")
_r("mashed potato|scalloped potato|au gratin|instant mashed potato|potato flake", contains="dairy")
_r("cream of chicken", contains="dairy gluten", may="soy", nv=True)
_r("cream of shrimp", contains="dairy gluten shellfish", nv=True)
_r("fat free sour|fat-free sour|nonfat sour|low-fat sour|light sour|reduced-fat sour", contains="dairy")
_r("cream of mushroom|cream of celery|cream of potato|"
   "condensed cream of|cheddar cheese soup|nacho cheese soup|gluten-free canned cream",
   contains="dairy gluten", may="soy")
_r("chicken gravy|turkey gravy|beef gravy|brown gravy|gravy",
   contains="gluten", may="dairy soy", nv=True)
_r("alfredo|alfredo sauce|bechamel|béchamel|white sauce|cheese sauce|queso|tzatziki|raita|"
   "cheez whiz|nacho cheese", contains="dairy", may="gluten")
_r("ice cream|gelato|frozen custard|frozen yogurt|sherbet|milkshake|malted milk", contains="dairy",
   may="eggs gluten")
_r("instant pudding|pudding mix|instant vanilla pudding|instant chocolate pudding|pudding",
   may="dairy")
_r("caramel|caramel sauce|dulce de leche|toffee|fudge|caramels", contains="dairy")
# Gluten-bearing products.
_r("flour|all-purpose flour|all purpose flour|plain flour|bread flour|cake flour|self-rising flour|"
   "self-raising flour|wheat flour|whole wheat flour|white flour|unbleached flour|pastry flour|"
   "semolina|durum|wheat|wheat germ|wheat bran|bulgur|bulgar|couscous|farro|spelt|kamut|freekeh|"
   "triticale|einkorn|emmer|barley|pearl barley|rye|pumpernickel|malt|malted|malt vinegar|"
   "malt extract|brewer's yeast|seitan|wheat gluten|vital wheat gluten|gluten|graham|graham cracker|"
   "cracker|saltine|oyster cracker|matzo|matzah|bread|breadcrumb|bread crumb|panko|crouton|stuffing|"
   "dressing mix|bun|roll|bagel|biscuit|croissant|muffin|english muffin|pita|naan|flatbread|ciabatta|"
   "baguette|focaccia|brioche|challah|tortilla|flour tortilla|wrap|taco shell|pasta|spaghetti|"
   "macaroni|noodle|fettuccine|fettucine|linguine|penne|rigatoni|rotini|ziti|orzo|lasagna|lasagne|"
   "tortellini|ravioli|gnocchi|vermicelli|angel hair|capellini|farfalle|fusilli|orecchiette|"
   "pappardelle|tagliatelle|cannelloni|manicotti|udon|ramen|soba|lo mein|chow mein|wonton|"
   "dumpling|egg roll wrapper|spring roll wrapper|phyllo|filo|puff pastry|pastry|pie crust|pie shell|"
   "pizza dough|pizza crust|dough|cake|cake mix|brownie|brownie mix|cookie|cookie dough|wafer|"
   "pretzel|cereal|bran|bran flake|granola|beer|ale|lager|stout|bisquick|baking mix|pancake mix|"
   "waffle|pancake|batter|roux|breading|cornbread|hush puppy|fish stick|chicken nugget|"
   "ladyfinger|lady finger|sponge cake|angel food|graham cracker crust|cracker crumb|cookie crumb|"
   "oreo|vanilla wafer|gingersnap|biscotti|tempura", contains="gluten", carb=True)
_r("oat|oats|rolled oat|oatmeal|old fashioned oat|old-fashioned oat|quick-cooking oat|quick oat|"
   "steel cut oat|oat bran|oat flour|instant oat|old-fashioned oatmeal|quick-cooking oatmeal",
   may="gluten", carb=True)
_r("seasoning mix|taco seasoning|soup mix|onion soup mix", may="gluten")
_r("bouillon|bouillon cube|stock cube|bouillon granule", may="gluten", nv=True)

# ---------------------------------------------------------------------------
# Meat, fish, shellfish.
# ---------------------------------------------------------------------------
_r("shrimp|prawn|crab|crabmeat|lobster|crawfish|crayfish|langoustine|scampi|krill|clam|mussel|oyster|"
   "scallop|squid|calamari|octopus|cuttlefish|abalone|conch|whelk|snail|escargot|sea urchin|"
   "seafood|shellfish", contains="shellfish", nv=True)
_r("seafood mix|mixed seafood|seafood", contains="shellfish fish", nv=True)
_r("fish|salmon|tuna|cod|tilapia|trout|halibut|snapper|anchovy|anchovies|sardine|mackerel|haddock|"
   "pollock|catfish|bass|sea bass|swordfish|mahi|mahi mahi|herring|flounder|sole|perch|pike|carp|eel|"
   "caviar|roe|lox|whitefish|orange roughy|grouper|monkfish|branzino|kipper|smelt|bluefish|"
   "fish stock|fish fillet|cod fish", contains="fish", nv=True)
_r("chicken|turkey|duck|goose|quail|pheasant|cornish hen|poultry|beef|steak|veal|pork|lamb|mutton|"
   "goat meat|venison|bison|buffalo meat|rabbit|elk|boar|ham|bacon|pancetta|prosciutto|salami|"
   "pepperoni|sausage|chorizo|kielbasa|bratwurst|andouille|hot dog|frankfurter|wiener|bologna|"
   "pastrami|corned beef|jerky|meatball|meatloaf|meat|ground meat|hamburger|burger|brisket|sirloin|"
   "tenderloin|chuck|ribeye|rib eye|short rib|spare rib|baby back rib|ribs|oxtail|liver|foie gras|"
   "kidneys|tongue|tripe|gizzard|giblet|heart|bone marrow|marrow bone|lard|suet|tallow|drippings|"
   "bacon grease|bacon fat|duck fat|schmaltz|gelatin|gelatine|jello|jell-o|aspic|marshmallow|"
   "bouillon|broth|stock|consomme|bone broth|demi-glace|pate|bacon bit|chicharron|cracklings|"
   "roast|pot roast|rump roast|london broil|cutlet|drumstick|wing|thigh|breast", nv=True)
_r("honey|honeycomb|bee pollen|royal jelly", nvg=True)
_r("carmine|cochineal|shellac|isinglass|bone char", nvg=True)

# ---------------------------------------------------------------------------
# Dairy, eggs, nuts (generic words last).
# ---------------------------------------------------------------------------
_r("milk|whole milk|skim milk|nonfat milk|low-fat milk|2% milk|evaporated milk|condensed milk|"
   "sweetened condensed milk|powdered milk|dry milk|cream|heavy cream|whipping cream|light cream|"
   "sour cream|half-and-half|half and half|creme fraiche|crème fraîche|clotted cream|butter|"
   "cheese|cheddar|parmesan|parmigiano|mozzarella|ricotta|feta|gouda|provolone|swiss|brie|camembert|"
   "gruyere|gruyère|asiago|romano|pecorino|monterey jack|colby|havarti|mascarpone|cottage cheese|"
   "cream cheese|neufchatel|queso fresco|cotija|paneer|panir|halloumi|gorgonzola|roquefort|"
   "blue cheese|bleu cheese|emmental|emmentaler|manchego|fontina|jarlsberg|velveeta|burrata|"
   "bocconcini|stilton|chevre|goat cheese|string cheese|american cheese|pepper jack|yogurt|yoghurt|"
   "greek yogurt|kefir|labneh|quark|skyr|curd|curds|dairy|irish cream|baileys", contains="dairy")
_r("egg|eggs|egg yolk|egg white|yolk|hard-boiled egg|hard boiled egg|boiled egg|quail egg|duck egg",
   contains="eggs")
_r("peanut|peanuts", contains="peanuts")
_r("almond|walnut|pecan|cashew|hazelnut|filbert|pistachio|macadamia|brazil nut|pine nut|pignoli|"
   "pignolia|chestnut|hickory nut|beechnut|shea nut|praline", contains="tree_nuts")
_r("nut|chopped nut|mixed nut", contains="tree_nuts", may="peanuts")

# High-carb markers for the keto/low-carb tag.
_r("sugar|brown sugar|powdered sugar|honey|syrup|maple syrup|corn syrup|molasses|agave|rice|potato|"
   "potatoes|sweet potato|corn|cornstarch|cornmeal|polenta|grits|banana|dates|raisin|quinoa|"
   "couscous|tortilla|chips|jam|jelly|juice|cake|candy|chocolate chip|marshmallow|flour",
   carb=True)


# Qualifiers that rule allergens out for the ingredient they modify. "lactose-free" is
# deliberately absent: lactose-free milk still contains milk protein.
_FREE_OF: list[tuple[re.Pattern[str], frozenset[str], bool, bool]] = []


def _free(pattern: str, removes: str, veg: bool = False, vegan: bool = False) -> None:
    _FREE_OF.append((re.compile(pattern), frozenset(removes.split()), veg, vegan))


_free(r"\b(gluten|wheat)[\s-]*free\b|\bgf\b", "gluten")
_free(r"\b(dairy|milk)[\s-]*free\b", "dairy")
_free(r"\begg[\s-]*free\b|\beggless\b", "eggs")
_free(r"\bsoy[\s-]*free\b", "soy")
_free(r"\b(nut|tree[\s-]*nut)[\s-]*free\b", "tree_nuts peanuts")
_free(r"\bpeanut[\s-]*free\b", "peanuts")
_free(r"\bsugar[\s-]*free\b|\bno sugar added\b|\bunsweetened\b", "")
_free(r"\bvegan\b|\bplant[\s-]*based\b", "dairy eggs fish shellfish", veg=True, vegan=True)
_free(r"\b(vegetarian|meatless|veggie|meat[\s-]*free|imitation bacon|bac[\s-]*os)\b", "fish shellfish", veg=True)
_SUGAR_FREE = re.compile(r"\bsugar[\s-]*free\b|\blow[\s-]*carb\b|\bketo\b|\bdiet\b|\bsplenda\b|\bstevia\b")
# "non-dairy" products often still contain casein, so it only downgrades dairy to "may contain".
_NON_DAIRY = re.compile(r"\bnon[\s-]*dairy\b|\bnondairy\b")


@dataclass
class IngredientFlags:
    ingredient: str
    contains: set[str] = field(default_factory=set)
    may_contain: set[str] = field(default_factory=set)
    non_vegetarian: bool = False
    non_vegan: bool = False
    high_carb: bool = False


def _normalize(text: str) -> str:
    text = text.lower().replace("&quot;", " ").replace("quot", " ")
    text = re.sub(r"[‐‑‒–—]", "-", text)
    text = re.sub(r"[^\w\s'&%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def analyze_ingredient(ingredient: str) -> IngredientFlags:
    """Flags for one ingredient name or line."""
    flags = IngredientFlags(ingredient=ingredient)
    text = _normalize(ingredient)
    for rule in RULES:
        if not rule.pattern.search(text):
            continue
        flags.contains |= rule.contains
        flags.may_contain |= rule.may_contain
        flags.non_vegetarian |= rule.non_vegetarian
        flags.non_vegan |= rule.non_vegan
        flags.high_carb |= rule.high_carb
        # Blank the match so generic rules do not fire on part of it.
        text = rule.pattern.sub(lambda m: " " * len(m.group(0)), text)

    original = _normalize(ingredient)
    for pattern, removes, veg, vegan in _FREE_OF:
        if pattern.search(original):
            flags.contains -= removes
            flags.may_contain -= removes
            if veg:
                flags.non_vegetarian = False
                flags.non_vegan = bool(flags.contains & {"dairy", "eggs"})
            if vegan:
                flags.non_vegan = False
    if _NON_DAIRY.search(original) and "dairy" in flags.contains:
        flags.contains.discard("dairy")
        flags.may_contain.add("dairy")
    if _SUGAR_FREE.search(original):
        flags.high_carb = False
    flags.may_contain -= flags.contains
    if flags.contains & {"dairy", "eggs", "fish", "shellfish"}:
        flags.non_vegan = True
    if flags.contains & {"fish", "shellfish"}:
        flags.non_vegetarian = True
    return flags


def _analyze_all(ingredients: list[str]) -> list[IngredientFlags]:
    return [analyze_ingredient(str(i)) for i in ingredients if i and str(i).strip()]


def classify_recipe_dietary(ingredients: list[str]) -> dict[str, bool]:
    """Dietary tags. A tag is False when any ingredient contains or may contain the item."""
    flags = _analyze_all(ingredients)
    flagged = set().union(*(f.contains | f.may_contain for f in flags)) if flags else set()
    vegetarian = not any(f.non_vegetarian for f in flags)
    vegan = vegetarian and not any(f.non_vegan for f in flags)
    return {
        "vegetarian": vegetarian,
        "vegan": vegan,
        "gluten_free": "gluten" not in flagged,
        "dairy_free": "dairy" not in flagged,
        "nut_free": not ({"tree_nuts", "peanuts"} & flagged),
        "keto_low_carb": not any(f.high_carb for f in flags),
    }


def detect_recipe_allergens(ingredients: list[str]) -> list[str]:
    """Allergens the recipe contains or may contain (sorted)."""
    flags = _analyze_all(ingredients)
    found: set[str] = set()
    for f in flags:
        found |= f.contains | f.may_contain
    return sorted(found)


def detect_recipe_allergens_detailed(ingredients: list[str]) -> dict[str, dict[str, Any]]:
    """{allergen: {"level": "contains" | "may_contain", "sources": [ingredient, ...]}}."""
    result: dict[str, dict[str, Any]] = {}
    for f in _analyze_all(ingredients):
        for level, items in (("contains", f.contains), ("may_contain", f.may_contain)):
            for allergen in items:
                entry = result.setdefault(allergen, {"level": level, "sources": []})
                if level == "contains":
                    entry["level"] = "contains"
                if f.ingredient not in entry["sources"]:
                    entry["sources"].append(f.ingredient)
    return dict(sorted(result.items()))
