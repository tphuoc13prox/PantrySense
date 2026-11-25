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

import html
import re
from collections.abc import Iterable
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
   "lobster mushroom|chicken of the woods|hen of the woods|ginger ale|root beer|"
   "milk thistle|milkweed|buckwheat|rice flour|corn flour|cornflour|tapioca flour|potato flour|"
   "chickpea flour|cassava flour|teff flour|sorghum flour|millet flour|coconut flour|masa harina|"
   "corn tortilla|rice noodle|rice vermicelli|glass noodle|cellophane noodle|bean thread|shirataki|"
   "rice paper|corn starch|cornstarch|potato starch|wheat-free|breadfruit|poultry seasoning|"
   "vegetable broth|vegetable stock|vegetable bouillon|mushroom broth|veggie broth|soda water|"
   "jackfruit|cream of coconut|coconut cream|coconut yogurt|coconut milk|light coconut milk|"
   "unsweetened coconut milk|lite coconut milk|rice milk|hemp milk|flax milk|pea milk|"
   "vegan sugar|sugar snap|seitan-free|peanut-free|nut-free|tree nut-free|coconut butter|"
   "coconut creamer|coconut whipped cream|artichoke heart|hearts of palm|heart of palm|palm heart|"
   "celery heart|romaine heart|duck sauce|chicken-fried|beefsteak tomato|sweet potato|jackfruit|"
   "coconut meat|nut meat|pate a choux|pate brisee|pate sucree|chocolate salami|angel wing|"
   "tonkatsu sauce|katsu sauce|sloppy joe sauce|sloppy joe seasoning|burger bun|burger roll")
# Fruit butters are cooked-down fruit.
_r("peach butter|pear butter|plum butter|apricot butter|cherry butter|berry butter|raspberry butter|"
   "strawberry butter|blueberry butter|blackberry butter|cranberry butter|mango butter|fig butter|"
   "quince butter|banana butter|date butter|persimmon butter|rhubarb butter")
_r("sweetbread", nv=True)
# Breaded products that the generic gluten list would otherwise swallow whole.
_r("fish stick|fish finger|fish fillet sandwich", contains="fish gluten", nv=True, carb=True)
_r("chicken nugget|chicken tender|chicken finger|corn dog", contains="gluten", nv=True, carb=True)
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
   "waffle|pancake|batter|roux|breading|cornbread|hush puppy|"
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
   "roast|pot roast|rump roast|london broil|cutlet|drumstick|wing|thigh|breast|rib|prime rib|rib roast|"
   "sparerib|spareribs|baby back|cheeseburger|loin|shank|flank|ground round|"
   "grouse|partridge|guinea fowl|squab|capon|poussin|ostrich|emu|kangaroo|alligator|frog leg|turtle soup|"
   "turtle meat|bear meat|curried goat|goat curry|game hen|wild game|haggis|scrapple|pemmican|"
   "bolognese|carbonara|coq au vin|bouillabaisse|cioppino|jambalaya|carne|carnitas|barbacoa|al pastor|"
   "birria|shepherd's pie|shepherds pie|cottage pie|stroganoff|bourguignon|bourguignonne|osso buco|"
   "ossobuco|saltimbocca|schnitzel|gyro|shawarma|souvlaki|reuben|blt|sloppy joe|cassoulet|moussaka|"
   "rendang|larb|bulgogi|galbi|kalbi|tonkatsu|katsu|yakitori|keema|kheema|nihari|rogan josh|"
   "picadillo|ropa vieja|feijoada|lechon|sisig|pho|pad thai|tom yum", nv=True)
_r("ceviche|sashimi|gravlax|kedgeree|bacalao|bacalhau|baccala|gefilte fish|lutefisk|fish taco|fish cake",
   contains="fish", nv=True)
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


def _match_rules(text: str) -> list[Rule]:
    """Rules that fire on normalized text, most specific first, each match blanked."""
    matched = []
    for rule in RULES:
        if not rule.pattern.search(text):
            continue
        matched.append(rule)
        # Blank the match so generic rules do not fire on part of it.
        text = rule.pattern.sub(lambda m: " " * len(m.group(0)), text)
    return matched


def analyze_ingredient(ingredient: str) -> IngredientFlags:
    """Flags for one ingredient name or line."""
    flags = IngredientFlags(ingredient=ingredient)
    for rule in _match_rules(_normalize(ingredient)):
        flags.contains |= rule.contains
        flags.may_contain |= rule.may_contain
        flags.non_vegetarian |= rule.non_vegetarian
        flags.non_vegan |= rule.non_vegan
        flags.high_carb |= rule.high_carb

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


# ---------------------------------------------------------------------------
# Recipe-level signals: the title, and Food.com keywords and category.
#
# Food.com drops ingredients it cannot link from its ingredient list, and the
# main protein is often among them ("Coleman Prime Rib Roast" lists olive oil,
# lemon juice, rosemary, garlic and pepper). The title and the source tags
# still name it. They can only add flags, never remove ones the ingredients
# raised.
# ---------------------------------------------------------------------------

ANIMAL_ALLERGENS = frozenset({"dairy", "eggs", "fish", "shellfish"})

# Plants that stand in for meat in a named dish: "black bean burger",
# "cauliflower steak", "tempeh bacon". Each form lists only the plants that make
# a known vegetarian dish with it; "mushroom burger" and "potato bacon soup"
# usually contain meat, so they are left out on purpose.
_PROTEIN_SUBS = (
    "tofu|tempeh|seitan|soy|soya|tvp|textured vegetable protein|quorn|tofurky|gardein|"
    "morningstar|boca|wheat meat"
)
_PORTOBELLO = r"(?:portobello|portabella|portabello|portobella)(?:[\s-]+mushroom)?"
_PLANT_FORMS: dict[str, str] = {
    r"burgers?|patty|patties": (
        "bean|black bean|white bean|kidney bean|pinto bean|lentil|chickpea|garbanzo|garbanzo bean|quinoa|"
        f"{_PORTOBELLO}|veggie|vegetable|falafel|nut|walnut|oat|millet|rice|"
        f"beet|beetroot|sweet potato|zucchini|eggplant|aubergine|cauliflower|spinach|carrot|{_PROTEIN_SUBS}"
    ),
    r"steaks?": (
        f"cauliflower|cabbage|{_PORTOBELLO}|eggplant|aubergine|celeriac|"
        f"squash|butternut|watermelon|beet|beetroot|{_PROTEIN_SUBS}"
    ),
    r"cutlets?": (
        f"vegetable|veg|veggie|mixed vegetable|potato|aloo|beet|beetroot|lentil|dal|chana|bread|{_PROTEIN_SUBS}"
    ),
    r"meatballs?|meat balls?": (
        f"lentil|eggplant|chickpea|black bean|bean|quinoa|veggie|vegetable|walnut|cauliflower|{_PROTEIN_SUBS}"
    ),
    r"sausages?|chorizo|hot dogs?|dogs?": f"carrot|veggie|{_PROTEIN_SUBS}",
    r"bacon": f"coconut|eggplant|carrot|rice paper|shiitake|mushroom|{_PROTEIN_SUBS}",
    r"jerky": f"mushroom|eggplant|coconut|banana|{_PROTEIN_SUBS}",
    r"wings?": f"cauliflower|buffalo cauliflower|{_PROTEIN_SUBS}",
    r"shepherd'?s pie|cottage pie|bolognese|sloppy joes?|stroganoff|bourguignon|carnitas|crumbles?|ground": (
        f"lentil|veggie|vegetable|jackfruit|{_PROTEIN_SUBS}"
    ),
}
_PLANT_FORM_PATTERNS = [
    re.compile(rf"(?<![\w-])(?:{plants})[\s-]+(?:{forms})(?![\w-])") for forms, plants in _PLANT_FORMS.items()
]

# Title words that look like a rule but are not food: "heart healthy", what a
# title says is absent ("no mayo", "gelatin-free"), and "roast" used as a verb
# in front of another word ("roast potatoes").
_TITLE_NOISE = [
    re.compile(r"(?<![\w-])hearts?(?![\w-])"),
    re.compile(r"(?<![\w-])(?:no|without|sans|minus)[\s-]+[\w']+"),
    re.compile(r"[\w']+[\s-]*free(?![\w-])"),
    re.compile(r"(?<![\w-])roast(?=\s+(?!with\b|for\b|in\b|and\b|dinner\b|recipe\b|ever\b|sandwich)[a-z])"),
]

# A qualifier covers the words after it up to the next separator: "vegan
# chicken noodle soup" is vegan, "salmon with vegan pesto" is not.
_TITLE_SEPARATORS = re.compile(
    r"\s(?:with|w/|and|n|or|plus|over|on|in|served|topped|stuffed|filled|wrapped|for)\s|"
    r"&|\+|,|;|:|/|\(|\)|\[|\]|\s-+\s"
)
_TITLE_QUALIFIERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<![\w-])(?:vegan|plant[\s-]*based)(?![\w-])"), "vegan"),
    # "veggie" is left out: "veggie beef soup" is beef soup with vegetables.
    (re.compile(r"(?<![\w-])(?:vegetarian|faux|fake)(?![\w-])"), "vegetarian"),
    (re.compile(r"(?<![\w-])(?:meatless|meat[\s-]*free|no[\s-]*meat)(?![\w-])"), "meatless"),
    (re.compile(r"(?<![\w-])(?:dairy[\s-]*free|non[\s-]*dairy|nondairy|milk[\s-]*free)(?![\w-])"), "dairy_free"),
    (re.compile(r"(?<![\w-])(?:egg[\s-]*free|eggless|no[\s-]*eggs?)(?![\w-])"), "egg_free"),
]


def _cleared_by(rule: Rule, qualifier: str) -> bool:
    """Whether a qualifier in front of the rule's word rules it out."""
    animal = rule.contains & ANIMAL_ALLERGENS
    if qualifier == "vegan":
        return rule.non_vegan
    if qualifier == "vegetarian":
        return rule.non_vegetarian or bool(animal & {"fish", "shellfish"})
    if qualifier == "meatless":
        return rule.non_vegetarian and not animal & {"fish", "shellfish"}
    if qualifier == "dairy_free":
        return animal == {"dairy"}
    if qualifier == "egg_free":
        return animal == {"eggs"}
    return False


def _title_segment_rules(segment: str) -> list[Rule]:
    text = _normalize(segment)
    hits = [(m.start(), kind) for pattern, kind in _TITLE_QUALIFIERS for m in pattern.finditer(text)]
    for pattern in _PLANT_FORM_PATTERNS + _TITLE_NOISE:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    # Rules do not match across a hyphen; in a title "chicken-vegetable" is two words.
    text = text.replace("-", " ")
    if not hits:
        return _match_rules(text)
    start = min(pos for pos, _ in hits)
    kinds = {kind for _, kind in hits}
    after = [rule for rule in _match_rules(" " + text[start:])
             if not any(_cleared_by(rule, kind) for kind in kinds)]
    return _match_rules(text[:start] + " ") + after


def _title_wide_removals(text: str) -> set[str]:
    """Animal products a whole title rules out, wherever the qualifier sits.

    "Vegan mac and cheese" and '"Cheese" sauce (dairy free)' name cheese that is
    not dairy. Only dairy and eggs (and honey, for vegan) are cleared this way,
    never meat or fish.
    """
    removes: set[str] = set()
    for pattern, kind in _TITLE_QUALIFIERS:
        if pattern.search(text):
            removes |= {"vegan": {"dairy", "eggs", "honey"}, "dairy_free": {"dairy"},
                        "egg_free": {"eggs"}}.get(kind, set())
    return removes


def analyze_title(title: str) -> IngredientFlags:
    """Animal-product flags named in a recipe title.

    Only meat, fish, shellfish, dairy, eggs and the other non-vegan words count:
    plant allergen words in titles ("cake", "nutty") are too noisy to be worth the
    false warnings. A qualifier ("vegan", "vegetarian", "meatless") clears the
    words after it within the same part of the title.
    """
    flags = IngredientFlags(ingredient=f"title: {title}")
    text = html.unescape(str(title or "")).lower()
    wide = _title_wide_removals(text)
    for segment in _TITLE_SEPARATORS.split(f" {text} "):
        for rule in _title_segment_rules(segment):
            animal = rule.contains & ANIMAL_ALLERGENS
            if rule.non_vegetarian or animal & {"fish", "shellfish"}:
                pass
            elif animal:
                animal -= wide
                if not animal:
                    continue
            elif not (rule.non_vegan and not rule.contains and not rule.may_contain) or "honey" in wide:
                # "May contain" words such as chocolate or margarine say too
                # little in a title; honey and the like count unless it says vegan.
                continue
            flags.contains |= animal
            flags.non_vegetarian |= rule.non_vegetarian
            flags.non_vegan = True
    if flags.contains & {"fish", "shellfish"}:
        flags.non_vegetarian = True
    return flags


# Food.com keywords and categories (the recipe's RecipeCategory counts as one
# more tag). Food.com also tags some egg dishes "Poultry" and "Meat"; that costs
# a few vegetarian recipes and is accepted.
_MEAT_TAGS = frozenset({
    "meat", "poultry", "chicken", "chicken breast", "chicken thigh & leg", "whole chicken", "chicken livers",
    "pork", "pork loin", "ham", "ham and bean soup", "roast beef", "steak", "veal", "lamb/sheep", "wild game",
    "deer", "elk", "moose", "bear", "rabbit", "duck", "duck breasts", "whole duck", "goose", "pheasant", "quail",
    "whole turkey", "turkey breasts", "turkey gravy", "beef organ meats", "beef liver", "beef kidney",
    "beef sauces", "meatballs", "meatloaf", "gelatin",
})
_FISH_TAGS = frozenset({
    "tuna", "fish tuna", "fish salmon", "halibut", "tilapia", "catfish", "trout", "bass", "whitefish",
    "orange roughy", "mahi mahi", "perch",
})
_SHELLFISH_TAGS = frozenset({"crab", "lobster", "mussels", "crawfish", "squid", "octopus", "oysters"})
_DAIRY_TAGS = frozenset({"cheese", "cheesecake"})
_EGG_TAGS = frozenset({"breakfast eggs", "eggs breakfast"})


def analyze_tags(tags: Iterable[str]) -> IngredientFlags:
    """Flags from source tags. Other tags (cuisines, times, "Easy") add nothing."""
    names = {str(t).strip().lower() for t in tags if t and str(t).strip()}
    flags = IngredientFlags(ingredient="tags: " + ", ".join(sorted(names)))
    if names & _FISH_TAGS:
        flags.contains.add("fish")
    if names & _SHELLFISH_TAGS:
        flags.contains.add("shellfish")
    if names & _DAIRY_TAGS:
        flags.contains.add("dairy")
    if names & _EGG_TAGS:
        flags.contains.add("eggs")
    flags.non_vegetarian = bool(names & _MEAT_TAGS or flags.contains & {"fish", "shellfish"})
    flags.non_vegan = flags.non_vegetarian or bool(flags.contains)
    if flags.contains or flags.non_vegetarian:
        flags.ingredient = "tags: " + ", ".join(
            sorted(names & (_MEAT_TAGS | _FISH_TAGS | _SHELLFISH_TAGS | _DAIRY_TAGS | _EGG_TAGS))
        )
    return flags


def _analyze_all(
    ingredients: list[str], title: str | None = None, tags: Iterable[str] | None = None
) -> list[IngredientFlags]:
    flags = [analyze_ingredient(str(i)) for i in ingredients if i and str(i).strip()]
    if title:
        flags.append(analyze_title(title))
    if tags:
        tag_flags = analyze_tags(tags)
        # Food.com tags a tofu cheesecake "Cheese"; the title knows better.
        removes = _title_wide_removals(html.unescape(title).lower()) if title else set()
        if tag_flags.contains & removes:
            tag_flags.contains -= removes
            tag_flags.non_vegan = tag_flags.non_vegetarian or bool(tag_flags.contains)
        flags.append(tag_flags)
    return flags


def classify_recipe_dietary(
    ingredients: list[str], title: str | None = None, tags: Iterable[str] | None = None
) -> dict[str, bool]:
    """Dietary tags. A tag is False when any ingredient contains or may contain the item.

    ``title`` and ``tags`` (source keywords and category) add the animal products
    an incomplete ingredient list misses.
    """
    flags = _analyze_all(ingredients, title, tags)
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


def detect_recipe_allergens(
    ingredients: list[str], title: str | None = None, tags: Iterable[str] | None = None
) -> list[str]:
    """Allergens the recipe contains or may contain (sorted)."""
    found: set[str] = set()
    for f in _analyze_all(ingredients, title, tags):
        found |= f.contains | f.may_contain
    return sorted(found)


def detect_recipe_allergens_detailed(
    ingredients: list[str], title: str | None = None, tags: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    """{allergen: {"level": "contains" | "may_contain", "sources": [ingredient, ...]}}."""
    result: dict[str, dict[str, Any]] = {}
    for f in _analyze_all(ingredients, title, tags):
        for level, items in (("contains", f.contains), ("may_contain", f.may_contain)):
            for allergen in items:
                entry = result.setdefault(allergen, {"level": level, "sources": []})
                if level == "contains":
                    entry["level"] = "contains"
                if f.ingredient not in entry["sources"]:
                    entry["sources"].append(f.ingredient)
    return dict(sorted(result.items()))
