"""Quantity-aware nutrition estimate for a recipe.

Each ingredient is parsed (quantity, unit, name), matched to a food in FOODS, converted to
grams and scaled from per-100 g values. Ingredients that match no food are reported as
unknown and contribute nothing; the result carries a coverage figure so the UI can say how
much of the recipe the estimate is based on.

Nutrient values are per 100 g, rounded, transcribed by hand from USDA FoodData Central
SR Legacy (https://fdc.nal.usda.gov/, public domain). Rows that group several foods use a
representative item (for example "cheddar" for hard cheeses, a spice average for dried
spices). Densities and piece weights come from the household measures listed in the same
SR Legacy entries, rounded.

Food.com quantities have no units (stored as "portion"), so each food has a `portion`
rule giving the unit that a bare number most likely means for it, e.g. "2 flour" -> cups,
"2 butter" -> tablespoons, "2 eggs" -> pieces. This is a heuristic and is the largest
source of error on that data.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.backend.recipes.ingredient_parser import (
    CONTAINER_UNITS,
    MASS_UNITS_G,
    VOLUME_UNITS_ML,
    ParsedIngredient,
    parse_ingredient,
    parse_ingredient_detail,
    singularize,
)

NUTRIENT_KEYS = ("calories", "protein", "carbs", "fat", "fiber")
DEFAULT_SERVINGS = 4
# Upper bound for one ingredient line; guards against shifted Food.com quantities
# ("8" meant for garlic landing on butter).
MAX_GRAMS_PER_INGREDIENT = 2500.0
FALLBACK_DENSITY = 0.8  # g/ml when a volume unit is used for a food without a density

# Grams for count-like units when the food does not define its own.
_GENERIC_UNIT_G = {
    "stick": 113.0, "slice": 25.0, "sprig": 1.0, "bunch": 50.0, "head": 500.0, "stalk": 40.0,
    "rib": 40.0, "handful": 30.0, "fillet": 170.0, "inch": 10.0, "clove": 3.0,
}


@dataclass(frozen=True)
class Food:
    key: str
    aliases: tuple[str, ...]
    per100: tuple[float, float, float, float, float]  # kcal, protein, carbs, fat, fiber
    density: float | None = None  # g per ml
    each_g: float | None = None  # one piece / medium item
    pkg_g: float | None = None  # one can / package as typically sold
    portion: tuple = ("each",)  # (unit,) or (unit_below, threshold, unit_at_or_above)
    unit_g: dict[str, float] = field(default_factory=dict)


FOODS: list[Food] = []


def _f(aliases: str, kcal: float, p: float, c: float, f: float, fib: float, *, d: float | None = None,
       each: float | None = None, pkg: float | None = None, portion: tuple = ("each",),
       units: dict[str, float] | None = None) -> None:
    names = tuple(a.strip() for a in aliases.split("|"))
    FOODS.append(Food(names[0], names, (kcal, p, c, f, fib), d, each, pkg, portion, units or {}))


CUP, TBSP, TSP = ("cup",), ("tbsp",), ("tsp",)

# --- grains, flours, starches ---------------------------------------------
_f("all-purpose flour|flour|plain flour|white flour|unbleached flour|bread flour|cake flour|"
   "self-rising flour|self-raising flour|pastry flour|strong white flour|unbleached white flour|"
   "unbleached all-purpose flour", 364, 10.3, 76.3, 1.0, 2.7, d=0.53, portion=CUP)
_f("whole wheat flour|wholemeal flour|whole wheat pastry flour", 340, 13.2, 72.0, 2.5, 10.7, d=0.51, portion=CUP)
_f("almond flour|almond meal|ground almond", 571, 21.0, 21.0, 50.0, 10.0, d=0.45, portion=CUP)
_f("rice flour|gluten-free flour|gluten-free all-purpose flour|gluten free flour|gluten-free flour blend",
   366, 6.0, 80.1, 1.4, 2.4, d=0.67, portion=CUP)
_f("cornstarch|corn starch|potato starch|tapioca starch|arrowroot|potato flour|cornflour|"
   "gluten-free cornflour", 381, 0.3, 91.3, 0.1, 0.9, d=0.54, portion=TBSP)
_f("cornmeal|yellow cornmeal|polenta|masa harina", 370, 8.1, 79.0, 3.6, 7.3, d=0.52, portion=CUP)
_f("rice|white rice|long-grain rice|long grain rice|basmati rice|jasmine rice|arborio rice|"
   "uncooked rice|minute rice|instant rice|wild rice", 365, 7.1, 80.0, 0.7, 1.3, d=0.78, portion=CUP)
_f("brown rice", 370, 7.9, 77.2, 2.9, 3.5, d=0.80, portion=CUP)
_f("cooked rice|cooked white rice|cooked brown rice|steamed rice|leftover rice", 130, 2.7, 28.2, 0.3, 0.4,
   d=0.66, portion=CUP)
_f("quinoa", 368, 14.1, 64.2, 6.1, 7.0, d=0.72, portion=CUP)
_f("cooked quinoa", 120, 4.4, 21.3, 1.9, 2.8, d=0.78, portion=CUP)
_f("couscous|bulgur|bulgur wheat", 376, 12.8, 77.4, 0.6, 5.0, d=0.73, portion=CUP)
_f("pasta|spaghetti|macaroni|elbow macaroni|penne|penne pasta|linguine|fettuccine|rotini|rotini pasta|"
   "lasagna noodle|egg noodle|noodle|pasta shell|small shell pasta|whole wheat pasta|ziti|fusilli|"
   "angel hair pasta|orzo|rice noodle|vermicelli|bow tie pasta|farfalle|rigatoni|udon|ramen noodle",
   371, 13.0, 74.7, 1.5, 3.2, d=0.42, portion=("lb", 4, "oz"))
_f("oat|rolled oat|old fashioned oat|old-fashioned oat|quick-cooking oat|quick oat|oatmeal|"
   "old-fashioned oatmeal|quick-cooking oatmeal|instant oat|steel cut oat", 379, 13.2, 67.7, 6.5, 10.1,
   d=0.34, portion=CUP)
_f("bread|white bread|whole wheat bread|french bread|italian bread|sourdough bread|baguette|"
   "bread slice|sandwich bread|rye bread", 266, 8.9, 49.4, 3.3, 2.7, each=28, portion=("each",),
   units={"slice": 28, "loaf": 450})
_f("breadcrumb|bread crumb|dry breadcrumb|panko|panko breadcrumb|panko bread crumb|seasoned breadcrumb|"
   "cracker crumb|graham cracker crumb", 395, 13.4, 71.9, 5.3, 4.5, d=0.46, portion=CUP)
_f("crouton|garlic crouton|caesar-flavor crouton", 407, 11.9, 73.5, 6.6, 5.1, d=0.13, portion=CUP)
_f("flour tortilla|tortilla|wheat tortilla", 306, 8.2, 49.9, 8.0, 3.5, each=45)
_f("corn tortilla", 218, 5.7, 44.6, 2.9, 6.3, each=26)
_f("tortilla chip|corn chip", 489, 7.0, 63.8, 23.4, 5.3, d=0.12, portion=CUP)
_f("cracker|saltine|graham cracker|chocolate graham cracker|ritz cracker", 430, 7.0, 74.0, 11.0, 2.6,
   each=7)
_f("english muffin", 227, 8.9, 44.2, 1.7, 3.5, each=57)
_f("hamburger bun|hot dog bun|bun|roll|dinner roll|bread roll|hoagie roll", 279, 9.7, 49.0, 4.3, 2.3,
   each=50)
_f("pie crust|pie shell|puff pastry|pastry|phyllo dough|pizza dough", 430, 5.6, 46.0, 25.0, 1.5, each=230,
   pkg=230)
_f("cereal|corn flake|rice krispie|crisp rice cereal", 380, 7.0, 85.0, 1.0, 3.0, d=0.12, portion=CUP)
_f("stuffing|stuffing mix|herb seasoned stuffing", 386, 11.0, 76.0, 3.8, 3.0, d=0.3, pkg=397,
   portion=("pkg", 3, "cup"))
_f("cake mix|yellow cake mix|white cake mix|chocolate cake mix|brownie mix", 426, 4.5, 77.0, 10.5, 1.0,
   pkg=432, portion=("pkg",))
_f("instant pudding|pudding mix|instant vanilla pudding|instant chocolate pudding|vanilla pudding mix|"
   "sugar-free instant chocolate", 360, 0.5, 89.0, 0.5, 1.0, pkg=96, portion=("pkg",))
_f("bisquick|biscuit mix|baking mix|pancake mix", 428, 8.0, 68.0, 14.0, 2.0, d=0.5, portion=CUP)

# --- sugars & sweets ---------------------------------------------------------
_f("sugar|white sugar|granulated sugar|caster sugar|superfine sugar|fine sugar|vegan sugar|raw sugar|"
   "turbinado sugar|cane sugar", 387, 0.0, 100.0, 0.0, 0.0, d=0.85, portion=CUP)
_f("brown sugar|light brown sugar|dark brown sugar|golden brown sugar|light-brown sugar", 380, 0.1, 98.1,
   0.0, 0.0, d=0.93, portion=CUP)
_f("powdered sugar|confectioners sugar|confectioner's sugar|confectioners' sugar|icing sugar",
   389, 0.0, 99.8, 0.0, 0.0, d=0.51, portion=CUP)
_f("sugar substitute|splenda|splenda granular|stevia|artificial sweetener|sweetener|splenda sugar blend",
   0, 0.0, 0.0, 0.0, 0.0, d=0.1, portion=CUP)
_f("honey", 304, 0.3, 82.4, 0.0, 0.2, d=1.42, portion=("cup", 1, "tbsp"))
_f("maple syrup|pancake syrup|syrup", 260, 0.0, 67.0, 0.1, 0.0, d=1.32, portion=("cup", 1, "tbsp"))
_f("corn syrup|light corn syrup|dark corn syrup|golden syrup", 286, 0.0, 77.6, 0.2, 0.0, d=1.38, portion=CUP)
_f("molasses|treacle", 290, 0.0, 74.7, 0.1, 0.0, d=1.42, portion=CUP)
_f("agave nectar|agave syrup|agave", 310, 0.1, 76.4, 0.5, 0.2, d=1.4, portion=TBSP)
_f("jam|jelly|preserve|fruit preserve|marmalade|apricot preserve", 278, 0.4, 68.9, 0.1, 1.1, d=1.35,
   portion=TBSP)
_f("chocolate chip|semisweet chocolate chip|semi-sweet chocolate chip|chocolate|semisweet chocolate|"
   "semi-sweet chocolate|dark chocolate|bittersweet chocolate|unsweetened chocolate|baking chocolate|"
   "chocolate morsel|mini chocolate chip", 480, 4.2, 63.0, 30.0, 5.9, d=0.71, portion=("cup", 2, "oz"))
_f("milk chocolate|milk chocolate chip", 535, 7.6, 59.4, 29.7, 3.4, d=0.71, portion=("cup", 2, "oz"))
_f("white chocolate|white chocolate chip", 539, 5.9, 59.2, 32.1, 0.2, d=0.71, portion=("cup", 2, "oz"))
_f("cocoa|cocoa powder|unsweetened cocoa|unsweetened cocoa powder|baking cocoa|dutch-process cocoa",
   228, 19.6, 57.9, 13.7, 37.0, d=0.36, portion=("cup", 2, "tbsp"))
_f("marshmallow|miniature marshmallow|mini marshmallow|marshmallow creme|marshmallow fluff",
   318, 1.8, 81.3, 0.2, 0.1, d=0.21, each=7, portion=CUP)
_f("nutella|chocolate hazelnut spread|mock nutella spread", 539, 6.3, 57.5, 30.9, 3.4, d=1.2, portion=TBSP)

# --- dairy & eggs ----------------------------------------------------------
_f("milk|whole milk|vitamin d milk", 61, 3.2, 4.8, 3.3, 0.0, d=1.03, portion=CUP)
_f("skim milk|nonfat milk|non-fat milk|fat-free milk|fat free milk", 34, 3.4, 5.0, 0.1, 0.0, d=1.03, portion=CUP)
_f("low-fat milk|lowfat milk|low fat milk|2% milk|1% milk|reduced-fat milk|reduced fat milk|"
   "2% low-fat milk|1% low-fat milk|1 low-fat milk|semi-skimmed milk", 50, 3.3, 4.8, 2.0, 0.0, d=1.03, portion=CUP)
_f("buttermilk|low-fat buttermilk|lowfat buttermilk", 40, 3.3, 4.8, 0.9, 0.0, d=1.03, portion=CUP)
_f("evaporated milk", 134, 6.8, 10.0, 7.6, 0.0, d=1.06, pkg=354, portion=("pkg", 2, "cup"))
_f("fat-free evaporated milk|evaporated skim milk|evaporated fat-free milk|low-fat evaporated milk",
   78, 7.6, 11.4, 0.2, 0.0, d=1.06, pkg=354, portion=("pkg", 2, "cup"))
_f("sweetened condensed milk|condensed milk", 321, 7.9, 54.4, 8.7, 0.0, d=1.28, pkg=397, portion=("pkg", 2, "cup"))
_f("powdered milk|dry milk|non-fat powdered milk|nonfat dry milk|milk powder|nonfat dry milk powder",
   362, 36.2, 52.0, 0.8, 0.0, d=0.29, portion=CUP)
_f("heavy cream|whipping cream|heavy whipping cream|double cream|cream|single cream",
   340, 2.8, 2.7, 36.1, 0.0, d=1.0, portion=CUP)
_f("half-and-half|half and half|half-and-half cream|light cream", 131, 3.1, 4.3, 11.5, 0.0, d=1.01, portion=CUP)
_f("sour cream", 198, 2.4, 4.6, 19.4, 0.0, d=0.97, portion=CUP)
_f("light sour cream|low-fat sour cream|reduced-fat sour cream|reduced fat sour cream", 136, 3.5, 7.1, 10.6,
   0.0, d=0.97, portion=CUP)
_f("fat free sour cream|fat-free sour cream|nonfat sour cream|fat free sour", 74, 3.1, 15.6, 0.0, 0.0, d=0.97,
   portion=CUP)
_f("yogurt|plain yogurt|natural yogurt|whole milk yogurt", 61, 3.5, 4.7, 3.3, 0.0, d=1.03, portion=CUP)
_f("greek yogurt|plain greek yogurt|nonfat greek yogurt", 73, 10.0, 3.9, 1.9, 0.0, d=1.03, portion=CUP)
_f("vanilla yogurt|low-fat vanilla yogurt|non-fat vanilla yogurt|fruit yogurt|low-fat yogurt|"
   "nonfat yogurt|non-fat yogurt", 85, 4.9, 13.8, 1.2, 0.0, d=1.03, portion=CUP)
_f("butter|unsalted butter|salted butter|real butter|sweet butter|sweet cream butter|butter flavored",
   717, 0.9, 0.1, 81.1, 0.0, d=0.96, portion=("cup", 1, "tbsp"), units={"stick": 113})
_f("margarine|unsalted margarine|vegan margarine|dairy-free margarine|vegan butter|butter substitute|"
   "spread|vegetable spread|earth balance", 717, 0.2, 0.7, 80.7, 0.0, d=0.96, portion=("cup", 1, "tbsp"),
   units={"stick": 113})
_f("light butter|light margarine|reduced-fat margarine|reduced-calorie margarine", 499, 3.3, 0.0, 55.1, 0.0,
   d=0.96, portion=("cup", 1, "tbsp"))
_f("ghee|clarified butter", 876, 0.3, 0.0, 99.5, 0.0, d=0.91, portion=TBSP)
_f("cream cheese|neufchatel cheese|neufchatel", 342, 5.9, 4.1, 34.2, 0.0, d=0.97, pkg=227,
   portion=("pkg", 3, "oz"))
_f("light cream cheese|reduced-fat cream cheese|fat-free cream cheese|low-fat cream cheese", 201, 7.9, 8.1,
   15.3, 0.0, d=0.97, pkg=227, portion=("pkg", 3, "oz"))
_f("cheese|cheddar|cheddar cheese|sharp cheddar|sharp cheddar cheese|extra-sharp cheddar cheese|"
   "mild cheddar cheese|american cheese|colby|colby cheese|monterey jack|monterey jack cheese|pepper jack|"
   "pepper jack cheese|swiss|swiss cheese|gruyere|gruyere cheese|gouda|provolone|provolone cheese|fontina|"
   "asiago|asiago cheese|havarti|jarlsberg|emmental|manchego|velveeta|velveeta cheese|processed cheese|"
   "mexican cheese|mexican blend cheese|colby jack cheese|cheese slice", 403, 24.9, 1.3, 33.1, 0.0, d=0.47,
   portion=CUP, units={"slice": 21})
_f("mozzarella|mozzarella cheese|part-skim mozzarella cheese|part-skim mozzarella|fresh mozzarella|"
   "bocconcini|string cheese", 280, 24.0, 3.1, 17.1, 0.0, d=0.47, portion=CUP, units={"slice": 21})
_f("parmesan|parmesan cheese|parmigiano-reggiano|parmigiano reggiano|romano|romano cheese|pecorino|"
   "pecorino romano|grated parmesan|parmesan-romano", 431, 38.5, 4.1, 28.6, 0.0, d=0.42, portion=CUP)
_f("feta|feta cheese|goat cheese|chevre|blue cheese|gorgonzola|roquefort|queso fresco|cotija|panir|paneer|"
   "halloumi|brie|camembert", 264, 14.2, 4.1, 21.3, 0.0, d=0.6, portion=CUP)
_f("ricotta|ricotta cheese|part-skim ricotta cheese|part-skim ricotta", 174, 11.3, 3.0, 13.0, 0.0, d=1.04,
   portion=CUP)
_f("fat-free ricotta cheese|low-fat ricotta cheese|light ricotta cheese|low-fat cottage cheese|"
   "cottage cheese|small-curd cottage cheese|low-fat small-curd cottage cheese", 98, 11.1, 3.4, 4.3, 0.0,
   d=0.96, portion=CUP)
_f("mascarpone|mascarpone cheese|creme fraiche", 429, 4.6, 4.6, 44.0, 0.0, d=0.96, portion=CUP)
_f("ice cream|vanilla ice cream|chocolate ice cream|frozen yogurt|gelato", 207, 3.5, 23.6, 11.0, 0.7,
   d=0.56, portion=CUP)
_f("cool whip|whipped topping|cool whip topping|cool whip free|frozen whipped topping|whipped cream",
   280, 1.0, 23.0, 19.0, 0.0, d=0.32, pkg=227, portion=("cup", 4, "oz"))
_f("egg|large egg|whole egg|hard-boiled egg|hard boiled egg|hard-cooked egg|boiled egg|beaten egg",
   143, 12.6, 0.7, 9.5, 0.0, each=50, d=1.03)
_f("egg white|large egg white", 52, 10.9, 0.7, 0.2, 0.0, each=33, d=1.03)
_f("egg yolk|large egg yolk", 322, 15.9, 3.6, 26.5, 0.0, each=17, d=1.03)
_f("egg substitute|egg beater|liquid egg substitute", 48, 10.0, 0.5, 0.0, 0.0, d=1.03, each=60, portion=CUP)
_f("mayonnaise|mayo|real mayonnaise|hellmann's mayonnaise", 680, 1.0, 0.6, 74.9, 0.0, d=0.92,
   portion=("cup", 2, "tbsp"))
_f("light mayonnaise|low-fat mayonnaise|reduced-fat mayonnaise|miracle whip|salad dressing|"
   "fat-free mayonnaise|light miracle whip", 300, 0.9, 12.0, 27.0, 0.0, d=0.95, portion=("cup", 2, "tbsp"))

# --- oils & fats -------------------------------------------------------------
_f("oil|olive oil|extra virgin olive oil|extra-virgin olive oil|virgin olive oil|light olive oil|"
   "vegetable oil|canola oil|corn oil|sunflower oil|safflower oil|peanut oil|sesame oil|toasted sesame oil|"
   "dark sesame oil|coconut oil|cooking oil|grapeseed oil|avocado oil|walnut oil|hazelnut oil|"
   "flax seed oil|flaxseed oil|salad oil|chili oil|extra virgin olive", 884, 0.0, 0.0, 100.0, 0.0,
   d=0.92, portion=("cup", 1, "tbsp"))
_f("shortening|vegetable shortening|crisco|lard|bacon grease|bacon fat|duck fat|suet", 900, 0.0, 0.0,
   100.0, 0.0, d=0.82, portion=("cup", 2, "tbsp"))
_f("cooking spray|nonstick cooking spray|non-stick cooking spray|canola oil cooking spray|"
   "olive oil flavored cooking spray|olive oil flavored|canola oil cooking|pam", 792, 0.0, 0.0, 88.0,
   0.0, each=0.3)

# --- meat ------------------------------------------------------------------
_f("chicken breast|boneless skinless chicken breast|boneless chicken breast|chicken breast half|"
   "boneless skinless chicken breast half|skinless chicken breast|chicken tender|chicken tenderloin|"
   "boneless skinless chicken", 120, 22.5, 0.0, 2.6, 0.0, each=174, d=0.6, portion=("each",))
_f("chicken|whole chicken|broiler-fryer chicken|fryer|chicken piece|roasting chicken|chicken leg|"
   "chicken quarter|un-boiled chicken|boneless chicken", 215, 18.6, 0.0, 15.1, 0.0, each=1100, d=0.6,
   portion=("lb",))
_f("chicken thigh|boneless skinless chicken thigh|boneless chicken thigh|skinless chicken thigh",
   177, 18.0, 0.0, 11.5, 0.0, each=110, d=0.6)
_f("chicken drumstick|drumstick|chicken wing|wing", 190, 18.0, 0.0, 12.5, 0.0, each=70)
_f("cooked chicken|cooked chicken breast|shredded chicken|rotisserie chicken|diced cooked chicken|"
   "chopped cooked chicken|cooked turkey", 190, 29.0, 0.0, 7.4, 0.0, d=0.56, portion=CUP)
_f("ground beef|hamburger|hamburger meat|ground chuck|minced beef|beef mince|ground round",
   254, 17.2, 0.0, 20.0, 0.0, d=0.95, portion=("lb",))
_f("lean ground beef|extra lean ground beef|ground sirloin|lean ground sirloin", 176, 20.0, 0.0, 10.0, 0.0,
   d=0.95, portion=("lb",))
_f("beef|beef chuck|chuck roast|beef chuck roast|boneless beef chuck|stew meat|beef stew meat|steak|"
   "sirloin|sirloin steak|beef tenderloin|beef tenderloin steak|flank steak|round steak|skirt steak|"
   "roast beef|brisket|beef brisket|corned beef|pot roast|rib eye|ribeye steak|top round|beef roast|"
   "short rib|beef short rib|veal|veal cutlet|eye of round roast|eye of round|rump roast|round roast|"
   "sirloin tip roast|london broil", 200, 20.0, 0.0, 13.0, 0.0, d=0.9, portion=("lb",))
_f("ground turkey|lean ground turkey|ground chicken", 150, 18.7, 0.0, 8.3, 0.0, d=0.95, portion=("lb",))
_f("turkey|turkey breast|whole turkey|turkey tenderloin", 135, 23.7, 0.0, 4.0, 0.0, d=0.6, portion=("lb",))
_f("pork|pork loin|pork tenderloin|pork shoulder|boneless pork shoulder|pork roast|pork butt|"
   "boston butt|pork rib|spare rib|baby back rib|rib", 190, 19.0, 0.0, 12.0, 0.0, d=0.9, portion=("lb",))
_f("pork chop|boneless pork chop|bone-in pork chop|pork cutlet", 172, 21.0, 0.0, 9.0, 0.0, each=170)
_f("ground pork|pork mince|sausage meat", 263, 16.9, 0.0, 21.2, 0.0, d=0.95, portion=("lb",))
_f("bacon|thick-cut bacon|bacon slice|bacon strip|turkey bacon|bacon bit|real bacon bit",
   417, 12.6, 1.4, 39.7, 0.0, each=28, d=0.5, portion=("lb", 2, "each"), units={"slice": 28})
_f("ham|cooked ham|diced ham|smoked ham|ham steak|deli ham|canadian bacon", 145, 21.0, 1.5, 5.5, 0.0,
   d=0.6, each=28, portion=CUP, units={"slice": 28})
_f("sausage|italian sausage|smoked sausage|kielbasa|pork sausage|chorizo|bratwurst|andouille sausage|"
   "breakfast sausage|polish sausage|turkey sausage|hot dog|frankfurter|wiener", 300, 13.0, 2.0, 27.0, 0.0,
   d=0.9, each=75, portion=("lb", 3, "each"))
_f("pepperoni|salami|prosciutto|pancetta|chorizo sausage|soppressata", 400, 22.0, 1.0, 34.0, 0.0, d=0.6,
   portion=("oz",), units={"slice": 5})
_f("lamb|leg of lamb|ground lamb|lamb chop|lamb shoulder|lamb shank", 282, 16.6, 0.0, 23.4, 0.0, d=0.9,
   each=110, portion=("lb",))

# --- fish & seafood ----------------------------------------------------------
_f("salmon|salmon fillet|boneless salmon fillet|salmon steak|fresh salmon", 208, 20.4, 0.0, 13.4, 0.0,
   each=170, d=0.9, portion=("lb", 2, "each"))
_f("smoked salmon|lox", 117, 18.3, 0.0, 4.3, 0.0, portion=("oz",))
_f("tuna|canned tuna|chunk tuna|tuna in water|solid white tuna|albacore tuna|tuna in vegetable|"
   "chunk light tuna", 116, 25.5, 0.0, 0.8, 0.0, pkg=140, d=0.6, portion=("pkg",))
_f("tuna steak|fresh tuna|ahi tuna", 109, 24.4, 0.0, 0.5, 0.0, each=170, portion=("lb", 2, "each"))
_f("fish|white fish|whitefish|fish fillet|cod|cod fillet|cod fish fillet|tilapia|tilapia fillet|haddock|"
   "halibut|sole|flounder|pollock|catfish|catfish fillet|snapper|red snapper|sea bass|bass|trout|orange roughy|"
   "mahi mahi|swordfish|perch", 92, 19.0, 0.0, 1.5, 0.0, each=170, d=0.9, portion=("lb", 2, "each"))
_f("shrimp|large shrimp|medium shrimp|small shrimp|jumbo shrimp|prawn|cooked shrimp|tiger prawn",
   85, 20.1, 0.0, 0.5, 0.0, d=0.6, each=15, portion=("lb",))
_f("crab|crabmeat|crab meat|lump crabmeat|lobster|lobster tail|scallop|sea scallop|bay scallop|clam|"
   "mussel|oyster|squid|calamari|octopus|crawfish|crayfish", 85, 17.0, 2.5, 1.0, 0.0, d=0.7, each=20,
   portion=("lb",))
_f("imitation crab|imitation crabmeat|surimi", 95, 7.6, 15.0, 0.5, 0.5, d=0.6, portion=("lb",))
_f("anchovy|anchovy fillet|anchovy paste", 210, 28.9, 0.0, 9.7, 0.0, each=4, d=1.0, portion=("each",))
_f("sardine|mackerel|herring|canned salmon", 208, 24.6, 0.0, 11.5, 0.0, pkg=106, portion=("pkg",))

# --- legumes, soy -------------------------------------------------------------
_f("tofu|firm tofu|extra firm tofu|extra-firm tofu|silken tofu|soft tofu|bean curd", 144, 17.3, 2.8, 8.7,
   2.3, d=1.0, pkg=397, portion=("pkg", 3, "oz"))
_f("tempeh", 192, 20.3, 7.6, 10.8, 0.0, pkg=227, portion=("oz",))
_f("bean|black bean|kidney bean|red kidney bean|pinto bean|cannellini bean|great northern bean|"
   "white bean|navy bean|chickpea|chick-pea|garbanzo bean|garbanzo|canned chick-pea|canned chickpea|"
   "canned black bean|butter bean|lima bean|refried bean|baked bean|black-eyed pea|cooked bean|"
   "canned kidney bean|canned bean", 130, 8.5, 23.0, 0.6, 7.0, d=0.73, pkg=250, portion=("pkg", 4, "cup"))
_f("lentil|dried lentil|red lentil|green lentil|brown lentil|split pea|dried bean|dry bean",
   352, 24.6, 63.4, 1.1, 10.7, d=0.81, portion=CUP)
_f("edamame|soybean", 121, 11.9, 8.9, 5.2, 5.2, d=0.66, portion=CUP)
_f("hummus", 166, 7.9, 14.3, 9.6, 6.0, d=1.0, portion=CUP)
_f("soy milk|soymilk|soya milk", 54, 3.3, 6.3, 1.8, 0.6, d=1.03, portion=CUP)

# --- nuts & seeds ------------------------------------------------------------
_f("walnut|walnut half|walnut piece|chopped walnut|black walnut", 654, 15.2, 13.7, 65.2, 6.7, d=0.42, portion=CUP)
_f("pecan|pecan half|pecan piece|chopped pecan", 691, 9.2, 13.9, 72.0, 9.6, d=0.42, portion=CUP)
_f("almond|sliced almond|slivered almond|whole almond|blanched almond|toasted almond", 579, 21.2, 21.6,
   49.9, 12.5, d=0.5, portion=CUP)
_f("cashew|cashew piece|cashew nut|raw cashew", 553, 18.2, 30.2, 43.9, 3.3, d=0.58, portion=CUP)
_f("peanut|roasted peanut|dry roasted peanut|salted peanut|chocolate-covered peanut", 567, 25.8, 16.1,
   49.2, 8.5, d=0.62, portion=CUP)
_f("hazelnut|filbert|pistachio|macadamia|macadamia nut|brazil nut|pine nut|pignoli|nut|mixed nut|"
   "chopped nut", 628, 15.0, 17.0, 60.0, 9.0, d=0.55, portion=CUP)
_f("peanut butter|creamy peanut butter|crunchy peanut butter|smooth peanut butter|natural peanut butter|"
   "almond butter|cashew butter|nut butter|sunflower seed butter", 588, 25.1, 20.0, 50.4, 6.0, d=1.09,
   portion=("cup", 2, "tbsp"))
_f("almond paste|marzipan", 458, 9.0, 47.8, 27.7, 4.8, d=1.0, portion=("oz",))
_f("sesame seed|toasted sesame seed|black sesame seed", 573, 17.7, 23.4, 49.7, 11.8, d=0.6, portion=TBSP)
_f("tahini|sesame paste", 595, 17.0, 21.2, 53.8, 9.3, d=1.0, portion=TBSP)
_f("sunflower seed|roasted sunflower seed|pumpkin seed|pepita", 584, 20.8, 20.0, 51.5, 8.6, d=0.59, portion=CUP)
_f("flax seed|flaxseed|ground flaxseed|flax meal|chia seed|hemp seed", 534, 18.3, 28.9, 42.2, 27.3, d=0.6,
   portion=TBSP)
_f("coconut|flaked coconut|shredded coconut|sweetened flaked coconut|sweetened shredded coconut|"
   "coconut flake", 456, 3.1, 51.8, 32.0, 4.5, d=0.39, portion=CUP)
_f("unsweetened coconut|desiccated coconut|unsweetened shredded coconut", 660, 6.9, 23.7, 64.5, 16.3,
   d=0.39, portion=CUP)
_f("coconut milk|unsweetened coconut milk|canned coconut milk|coconut cream|cream of coconut",
   197, 2.0, 2.8, 21.3, 0.0, d=1.0, pkg=400, portion=("pkg", 3, "cup"))
_f("light coconut milk|lite coconut milk|reduced-fat coconut milk", 70, 0.7, 2.0, 6.9, 0.0, d=1.0,
   pkg=400, portion=("pkg", 3, "cup"))
_f("almond milk|unsweetened almond milk|cashew milk|rice milk|oat milk|hemp milk|coconut beverage",
   30, 0.6, 3.5, 1.4, 0.3, d=1.03, portion=CUP)

# --- vegetables ------------------------------------------------------------
_f("onion|yellow onion|white onion|red onion|sweet onion|spanish onion|vidalia onion|brown onion|"
   "chopped onion|diced onion|pearl onion", 40, 1.1, 9.3, 0.1, 1.7, each=110, d=0.68)
_f("green onion|scallion|spring onion|green onion top", 32, 1.8, 7.3, 0.2, 2.6, each=15, d=0.42)
_f("shallot", 72, 2.5, 16.8, 0.1, 3.2, each=25, d=0.68)
_f("leek", 61, 1.5, 14.2, 0.3, 1.8, each=89, d=0.37)
_f("garlic|garlic clove|fresh garlic clove|minced garlic|garlic paste|chopped garlic|roasted garlic",
   149, 6.4, 33.1, 0.5, 2.1, each=3, d=0.57, units={"clove": 3, "head": 45})
_f("garlic powder|onion powder|granulated garlic|dried minced onion|onion flake|dehydrated onion",
   331, 16.6, 72.7, 0.7, 9.0, d=0.6, portion=TSP)
_f("ginger|fresh ginger|gingerroot|ginger root|grated ginger|minced ginger|crystallized ginger",
   80, 1.8, 17.8, 0.8, 2.0, d=0.6, each=15, portion=TBSP, units={"inch": 10, "piece": 15})
_f("tomato|fresh tomato|ripe tomato|plum tomato|roma tomato|beefsteak tomato|vine tomato|heirloom tomato",
   18, 0.9, 3.9, 0.2, 1.2, each=123, d=0.76)
_f("cherry tomato|grape tomato", 18, 0.9, 3.9, 0.2, 1.2, each=17, d=0.6, portion=("cup", 4, "each"))
_f("canned tomato|diced tomato|crushed tomato|chopped tomato|stewed tomato|whole tomato|tomato puree|"
   "tomatoes and green chilies|tomato and green chily|tomato and green chili|rotel|petite diced tomato|"
   "fire-roasted tomato|canned diced tomato|diced tomatoes with juice", 32, 1.6, 7.0, 0.3, 1.9, d=1.03,
   pkg=411, portion=("pkg", 3, "cup"))
_f("tomato paste", 82, 4.3, 18.9, 0.5, 4.1, d=1.1, pkg=170, portion=("pkg", 1.5, "tbsp"))
_f("tomato sauce|tomato soup", 24, 1.2, 5.3, 0.3, 1.5, d=1.03, pkg=227, portion=("pkg", 3, "cup"))
_f("spaghetti sauce|marinara sauce|marinara|pasta sauce|pizza sauce|prego|ragu", 50, 1.4, 8.0, 1.5, 1.8,
   d=1.03, pkg=680, portion=("pkg", 3, "cup"))
_f("sun-dried tomato|sun dried tomato|sun-dried tomato packed|oil-packed sun-dried tomato", 258, 14.1,
   55.8, 3.0, 12.3, d=0.5, portion=CUP)
_f("ketchup|catsup|tomato ketchup", 101, 1.0, 27.4, 0.1, 0.3, d=1.15, portion=("cup", 1, "tbsp"))
_f("salsa|chipotle salsa|picante sauce|pico de gallo|salsa verde", 36, 1.5, 7.0, 0.2, 1.9, d=1.0, portion=CUP)
_f("carrot|baby carrot|shredded carrot|grated carrot", 41, 0.9, 9.6, 0.2, 2.8, each=61, d=0.54)
_f("celery|celery rib|celery stalk|celery heart|celery leaf", 16, 0.7, 3.0, 0.2, 1.6, each=40, d=0.43,
   units={"rib": 40, "stalk": 40})
_f("bell pepper|green pepper|red pepper|green bell pepper|red bell pepper|yellow bell pepper|"
   "orange bell pepper|sweet pepper|capsicum|yellow pepper|sweet red pepper|roasted red pepper",
   26, 1.0, 6.0, 0.3, 2.1, each=119, d=0.62)
_f("jalapeno|jalapeno pepper|serrano pepper|serrano chile|chili pepper|chile pepper|hot pepper|"
   "green chili|green chile|hot green chili pepper|red chili|thai chile|fresno chile|habanero|"
   "poblano pepper|poblano|chipotle pepper|chipotle chile|anaheim pepper|green chily", 40, 1.9, 8.8,
   0.4, 1.5, each=30, d=0.6)
_f("potato|russet potato|baking potato|new potato|red potato|yukon gold potato|white potato|"
   "idaho potato|red-skinned potato|boiling potato|hash brown|frozen hash brown", 77, 2.0, 17.5, 0.1,
   2.2, each=213, d=0.64)
_f("sweet potato|yam", 86, 1.6, 20.1, 0.1, 3.0, each=130, d=0.56)
_f("mashed potato|prepared mashed potato", 113, 1.9, 17.0, 4.2, 1.5, d=0.89, portion=CUP)
_f("mushroom|fresh mushroom|white mushroom|button mushroom|cremini mushroom|crimini mushroom|"
   "shiitake mushroom|portobello mushroom|baby bella mushroom|sliced mushroom|brown mushroom|"
   "canned mushroom|oyster mushroom", 22, 3.1, 3.3, 0.3, 1.0, each=18, d=0.3, portion=("cup", 4, "oz"))
_f("spinach|baby spinach|spinach leaf|fresh spinach leaf|fresh spinach|chard|swiss chard|collard green|"
   "mustard green|watercress|bok choy|baby bok choy", 23, 2.9, 3.6, 0.4, 2.2, d=0.13, portion=CUP,
   units={"bunch": 340})
_f("frozen spinach|frozen chopped spinach|chopped frozen spinach", 29, 3.6, 4.2, 0.5, 2.9, pkg=283,
   d=0.6, portion=("pkg",))
_f("kale|baby kale", 49, 4.3, 8.8, 0.9, 3.6, d=0.28, portion=CUP, units={"bunch": 200})
_f("lettuce|romaine lettuce|romaine|iceberg lettuce|head romaine lettuce|romaine lettuce leaf|"
   "lettuce leaf|leaf lettuce|butter lettuce|mixed green|salad green|spring mix|arugula|baby arugula|"
   "mesclun|endive|radicchio", 15, 1.2, 3.0, 0.2, 1.5, d=0.2, each=600, portion=("each", 2, "cup"),
   units={"leaf": 10, "head": 600})
_f("broccoli|broccoli floret|broccoli crown", 34, 2.8, 6.6, 0.4, 2.6, d=0.38, each=350,
   portion=("each", 2, "cup"))
_f("cauliflower|cauliflower floret|cauliflower head", 25, 1.9, 5.0, 0.3, 2.0, d=0.45, each=575,
   portion=("each", 2, "cup"))
_f("cabbage|red cabbage|green cabbage|napa cabbage|savoy cabbage|coleslaw mix|shredded cabbage",
   25, 1.3, 5.8, 0.1, 2.5, d=0.38, each=900, portion=("each", 2, "cup"))
_f("brussels sprout", 43, 3.4, 9.0, 0.3, 3.8, d=0.38, each=19, portion=("lb",))
_f("cucumber|english cucumber|seedless cucumber", 15, 0.7, 3.6, 0.1, 0.5, each=300, d=0.55)
_f("zucchini|summer squash|yellow squash|courgette|green zucchini", 17, 1.2, 3.1, 0.3, 1.0, each=196, d=0.53)
_f("eggplant|aubergine|japanese eggplant", 25, 1.0, 5.9, 0.2, 3.0, each=458, d=0.35)
_f("corn|corn kernel|frozen corn|sweet corn|whole kernel corn|corn on the cob|ear of corn|frozen corn kernel|"
   "fresh corn kernel", 86, 3.3, 19.0, 1.4, 2.7, d=0.65, each=90, portion=CUP)
_f("creamed corn|cream-style corn|cream style corn", 72, 1.7, 18.1, 0.4, 1.2, d=1.05, pkg=425,
   portion=("pkg", 3, "cup"))
_f("pea|green pea|frozen pea|sweet pea|petite pea|baby pea", 81, 5.4, 14.5, 0.4, 5.1, d=0.6, portion=CUP)
_f("snow pea|sugar snap pea|snap pea|pea pod", 42, 2.8, 7.6, 0.2, 2.6, d=0.27, portion=CUP)
_f("green bean|fresh green bean|string bean|french bean|haricot vert|cut green bean|frozen green bean",
   31, 1.8, 7.0, 0.2, 2.7, d=0.47, portion=("lb", 4, "cup"))
_f("asparagus|asparagus spear", 20, 2.2, 3.9, 0.1, 2.1, each=16, d=0.57, portion=("lb",))
_f("avocado|ripe avocado|hass avocado", 160, 2.0, 8.5, 14.7, 6.7, each=150, d=0.6)
_f("pumpkin|canned pumpkin|pumpkin puree|solid pack pumpkin|pureed pumpkin", 34, 1.1, 8.1, 0.3, 2.9,
   d=1.03, pkg=425, portion=("pkg", 3, "cup"))
_f("butternut squash|squash|acorn squash|spaghetti squash|winter squash|pureed squash butternut",
   45, 1.0, 11.7, 0.1, 2.0, d=0.59, each=1200, portion=("each", 2, "cup"))
_f("artichoke heart|artichoke|marinated artichoke heart|frozen artichoke heart", 47, 3.3, 10.5, 0.2, 5.4,
   d=0.7, each=30, pkg=240, portion=("pkg", 3, "each"))
_f("olive|black olive|green olive|kalamata olive|ripe olive|pitted olive|sliced black olive|"
   "pimento-stuffed olive", 115, 0.8, 6.3, 10.7, 3.2, each=4, d=0.57, portion=("cup", 4, "each"))
_f("caper", 23, 2.4, 4.9, 0.9, 3.2, d=0.58, portion=TBSP)
_f("pickle|dill pickle|gherkin|pickle relish|sweet pickle relish|relish", 60, 0.5, 14.0, 0.3, 1.2, each=65,
   d=1.0, portion=("each",))
_f("sauerkraut|kimchi", 19, 0.9, 4.3, 0.1, 2.9, d=0.6, portion=CUP)
_f("okra", 33, 1.9, 7.5, 0.2, 3.2, d=0.42, portion=CUP)
_f("beet|beetroot|radish|turnip|parsnip|rutabaga|jicama|daikon", 40, 1.2, 9.0, 0.2, 2.8, each=82, d=0.58)
_f("fennel|fennel bulb", 31, 1.2, 7.3, 0.2, 3.1, each=234, d=0.37)
_f("bean sprout|sprout|alfalfa sprout", 30, 3.0, 5.9, 0.2, 1.8, d=0.44, portion=CUP)
_f("water chestnut|bamboo shoot", 50, 0.9, 12.3, 0.1, 2.5, d=0.6, pkg=140, portion=("pkg", 3, "cup"))
_f("lemongrass|lemon grass", 99, 1.8, 25.3, 0.5, 0.0, each=20, portion=("each",))

# --- herbs, spices, seasonings --------------------------------------------------
_f("parsley|fresh parsley|italian parsley|flat leaf parsley|flat-leaf parsley|curly parsley|cilantro|"
   "fresh cilantro|coriander leaf|basil|fresh basil|basil leaf|fresh basil leaf|mint|mint leaf|"
   "fresh mint|fresh mint leaf|dill|fresh dill|dill leaf|chive|fresh chive|tarragon|fresh tarragon|"
   "fresh thyme|thyme sprig|fresh thyme sprig|fresh thyme leaf|fresh rosemary|rosemary sprig|fresh sage|"
   "sage leaf|fresh sage leaf|fresh oregano|fresh herb|thai basil|fresh coriander|fresh parsley leaf|"
   "of fresh mint", 36, 3.0, 6.3, 0.8, 3.3, d=0.25, each=0.5, portion=("cup", 1, "tbsp"),
   units={"sprig": 1, "bunch": 30, "leaf": 0.5})
_f("dried oregano|oregano|dried basil|dried thyme|thyme|dried rosemary|rosemary|dried parsley|dried dill|"
   "dill weed|dried dill weed|dried tarragon|dry tarragon|dried marjoram|marjoram|dried sage|sage|"
   "rubbed sage|ground sage|bay leaf|italian seasoning|herbes de provence|dried mint|mint powder|"
   "dried oregano leaf|dried basil leaf|dried thyme leaf|oregano leaf|thyme leaf|savory|"
   "cumin|ground cumin|cumin seed|paprika|smoked paprika|sweet paprika|chili powder|chile powder|"
   "ancho chile powder|chipotle powder|curry powder|garam masala|cinnamon|ground cinnamon|cinnamon stick|"
   "nutmeg|ground nutmeg|allspice|ground allspice|clove|ground clove|whole clove|ginger powder|"
   "ground ginger|cayenne|cayenne pepper|ground red pepper|crushed red pepper|crushed red pepper flake|"
   "red pepper flake|chili flake|turmeric|turmeric powder|ground turmeric|coriander|ground coriander|"
   "coriander seed|cardamom|ground cardamom|cardamom pod|black cardamom pod|cardamom seed|"
   "mustard powder|dry mustard|ground mustard|mustard seed|pumpkin pie spice|apple pie spice|"
   "five-spice powder|chinese five spice|black pepper|pepper|ground black pepper|ground pepper|"
   "fresh ground black pepper|fresh ground pepper|cracked black pepper|white pepper|ground white pepper|"
   "peppercorn|black peppercorn|lemon pepper|fennel seed|caraway seed|celery seed|poppy seed|anise|"
   "star anise|anise seed|mace|ground mace|saffron|sumac|za'atar|cajun seasoning|creole seasoning|"
   "taco seasoning|old bay seasoning|poultry seasoning|steak seasoning|greek seasoning|jerk seasoning|"
   "chili seasoning|seasoning|spice|xanthan gum|fenugreek|asafoetida|vanilla bean|juniper berry",
   300, 12.0, 60.0, 10.0, 30.0, d=0.5, each=0.5, portion=TSP)
_f("salt|kosher salt|sea salt|table salt|coarse salt|coarse sea salt|fine sea salt|seasoning salt|"
   "seasoned salt|garlic salt|celery salt|onion salt|salt and pepper|accent seasoning|msg|rock salt|"
   "pink salt|himalayan salt", 0, 0.0, 0.0, 0.0, 0.0, d=1.2, portion=TSP)
_f("baking powder|baking soda|bicarbonate of soda|cream of tartar|salt substitute", 53, 0.0, 27.7, 0.0,
   0.2, d=0.9, portion=TSP)
_f("yeast|active dry yeast|dry yeast|instant yeast|dry active yeast|fast rise yeast|rapid rise yeast|"
   "bread machine yeast|quick-rise yeast|fresh yeast", 325, 40.4, 41.2, 7.6, 26.9, d=0.8, pkg=7,
   portion=("pkg", 2, "tsp"))
_f("vanilla|vanilla extract|pure vanilla extract|vanilla essence|almond extract|lemon extract|"
   "peppermint extract|rum extract|maple extract|imitation vanilla|orange extract|coconut extract",
   288, 0.1, 12.7, 0.1, 0.0, d=0.88, portion=TSP)
_f("gelatin|unflavored gelatin|gelatine|knox gelatin", 335, 85.6, 0.0, 0.1, 0.0, pkg=7, d=0.6, portion=("pkg",))
_f("jello|jell-o|gelatin dessert|flavored gelatin|jello gelatin|sugar-free jello", 381, 7.8, 90.5, 0.0,
   0.0, pkg=85, portion=("pkg",))

# --- sauces, condiments, liquids ------------------------------------------------
_f("water|cold water|warm water|hot water|boiling water|ice water|ice|ice cube|crushed ice|club soda|"
   "soda water|sparkling water|seltzer|fresh water|tap water|lukewarm water|filtered water",
   0, 0.0, 0.0, 0.0, 0.0, d=1.0, portion=CUP)
_f("coffee|brewed coffee|strong coffee|espresso|tea|brewed tea|black tea|green tea|strong brewed coffee",
   1, 0.1, 0.0, 0.0, 0.0, d=1.0, portion=CUP)
_f("instant coffee|instant coffee granule|instant espresso powder|espresso powder|instant coffee powder",
   353, 12.2, 75.4, 0.5, 0.0, d=0.3, portion=TSP)
_f("broth|chicken broth|beef broth|vegetable broth|stock|chicken stock|beef stock|vegetable stock|"
   "low sodium chicken broth|low-sodium chicken broth|fat-free chicken broth|reduced-sodium chicken broth|"
   "low sodium beef broth|low sodium vegetable broth|low sodium chicken|low sodium beef|fish stock|"
   "chicken bouillon|beef bouillon|vegetable bouillon|consomme|beef consomme|turkey broth|bone broth",
   7, 1.0, 0.4, 0.2, 0.0, d=1.0, pkg=411, portion=CUP)
_f("bouillon cube|chicken bouillon cube|beef bouillon cube|vegetable bouillon cube|stock cube|"
   "chicken bouillon granule|beef bouillon granule|bouillon granule|bouillon", 240, 15.0, 20.0, 12.0, 0.0,
   each=4, d=0.8, portion=("each",))
_f("soy sauce|light soy sauce|dark soy sauce|low sodium soy sauce|low-sodium soy sauce|reduced-sodium soy sauce|"
   "tamari|shoyu|soya sauce|low sodium soy|kecap manis", 53, 8.1, 4.9, 0.6, 0.8, d=1.15, portion=("cup", 1, "tbsp"))
_f("worcestershire sauce|worcestershire", 78, 0.0, 19.5, 0.0, 0.0, d=1.1, portion=TBSP)
_f("fish sauce|thai fish sauce|nam pla|nuoc mam", 35, 5.1, 3.6, 0.0, 0.0, d=1.2, portion=TBSP)
_f("oyster sauce", 51, 1.4, 10.9, 0.3, 0.3, d=1.2, portion=TBSP)
_f("hoisin sauce|teriyaki sauce|teriyaki marinade|stir-fry sauce|sweet and sour sauce|plum sauce|"
   "duck sauce|sweet chili sauce", 200, 2.5, 44.0, 2.0, 1.0, d=1.15, portion=TBSP)
_f("barbecue sauce|bbq sauce|steak sauce|a1 steak sauce", 172, 0.8, 40.8, 0.6, 0.9, d=1.1, portion=CUP)
_f("hot sauce|tabasco|tabasco sauce|hot pepper sauce|sriracha|sriracha sauce|louisiana hot sauce|"
   "chili sauce|chili garlic sauce|sambal oelek|red hot sauce|frank's red hot|buffalo sauce",
   40, 1.0, 7.0, 0.6, 0.8, d=1.05, portion=TSP)
_f("vinegar|white vinegar|distilled vinegar|cider vinegar|apple cider vinegar|red wine vinegar|"
   "white wine vinegar|rice vinegar|rice wine vinegar|sherry vinegar|malt vinegar|seasoned rice vinegar",
   21, 0.0, 0.9, 0.0, 0.0, d=1.0, portion=("cup", 1, "tbsp"))
_f("balsamic vinegar|balsamic glaze", 88, 0.5, 17.0, 0.0, 0.0, d=1.06, portion=("cup", 1, "tbsp"))
_f("mustard|dijon mustard|prepared mustard|yellow mustard|spicy brown mustard|whole grain mustard|"
   "grainy mustard|honey mustard|stone ground mustard|prepared yellow mustard|german mustard",
   66, 4.4, 5.8, 4.0, 3.3, d=1.05, portion=TBSP)
_f("lemon juice|fresh lemon juice|lime juice|fresh lime juice|key lime juice|lemon|lime|juice of lemon|"
   "lemon wedge|lemon slice|lime wedge|limes juice of|lemons juice of|lemon juice of", 22, 0.4, 6.9, 0.2,
   0.3, d=1.03, each=45, portion=("cup", 1, "tbsp"))
_f("lemon zest|lemon rind|lemon peel|orange zest|orange rind|orange peel|lime zest|lime rind|lime peel|"
   "grated lemon rind|grated orange rind|fresh lemon rind|lemon rind of|orange zest of|lemon zest of|"
   "orange rind of|lime zest of|citron|candied peel|candied orange peel", 47, 1.5, 16.0, 0.3, 10.6,
   d=0.4, each=6, portion=TSP)
_f("orange|navel orange|mandarin orange|tangerine|clementine|grapefruit|blood orange|satsuma",
   47, 0.9, 11.8, 0.1, 2.4, each=131, d=0.75)
_f("orange juice|apple juice|cranberry juice|pineapple juice|grape juice|apple cider|tangerine juice|"
   "grapefruit juice|fruit juice|lemonade|white grape juice|pomegranate juice|passion fruit pulp",
   45, 0.7, 10.4, 0.2, 0.2, d=1.04, portion=CUP)
_f("cranberry sauce|whole berry cranberry sauce|jellied cranberry sauce", 159, 0.2, 40.4, 0.2, 1.0,
   d=1.15, pkg=397, portion=("pkg", 3, "cup"))
_f("curry paste|green curry paste|red curry paste|thai red curry paste|yellow curry paste|harissa",
   140, 2.5, 15.0, 8.0, 4.0, d=1.0, portion=TBSP)
_f("miso|white miso|red miso|miso paste|shiro miso", 198, 12.8, 25.4, 6.0, 5.4, d=1.15, portion=TBSP)
_f("pesto|basil pesto|pesto sauce|prepared pesto", 450, 5.0, 6.0, 45.0, 1.5, d=1.0, portion=("cup", 1.5, "tbsp"))
_f("salad dressing|italian dressing|vinaigrette|ranch dressing|caesar dressing|kittencals famous caesar|"
   "blue cheese dressing|thousand island dressing|french dressing|balsamic vinaigrette|honey mustard dressing",
   350, 1.0, 10.0, 34.0, 0.0, d=1.0, portion=("cup", 1, "tbsp"))
_f("tartar sauce|aioli|remoulade", 490, 1.0, 13.0, 48.0, 0.0, d=1.0, portion=TBSP)
_f("cream of mushroom soup|condensed cream of mushroom soup|cream of chicken soup|"
   "condensed cream of chicken soup|cream of celery soup|cream of mushroom|cream of chicken|cream of celery|"
   "cream of potato soup|cheddar cheese soup|condensed cream of mushroom|condensed cream of chicken|"
   "gluten-free canned cream", 81, 1.6, 6.5, 5.3, 0.8, d=1.04, pkg=305, portion=("pkg",))
_f("gravy|brown gravy|turkey gravy|chicken gravy|gravy mix", 50, 1.5, 5.0, 2.5, 0.3, d=1.03, portion=CUP)
_f("mirin|sweet rice wine|cooking sherry", 240, 0.2, 43.0, 0.0, 0.0, d=1.1, portion=TBSP)
_f("wine|white wine|red wine|dry white wine|dry red wine|marsala wine|marsala|sherry|dry sherry|"
   "sherry wine|port wine|port|sake|rice wine|chinese rice wine|shaoxing wine|vermouth|dry vermouth|"
   "madeira|champagne|sparkling wine|rose wine|cooking wine|white cooking wine", 83, 0.1, 2.6, 0.0, 0.0,
   d=0.99, portion=CUP)
_f("beer|lager|ale|stout|dark beer|light beer|guinness", 43, 0.5, 3.6, 0.0, 0.0, d=1.0, pkg=355, portion=("pkg",))
_f("rum|vodka|brandy|bourbon|whiskey|whisky|tequila|gin|cognac|dark rum|light rum|white rum|spiced rum|"
   "scotch|banana rum|mango rum|coconut rum|flavored vodka", 231, 0.0, 0.0, 0.0, 0.0, d=0.95, portion=("fl_oz",))
_f("liqueur|kahlua|amaretto|amaretto liqueur|triple sec|grand marnier|cointreau|frangelico|baileys|"
   "irish cream|creme de cacao|creme de menthe|green creme de|white creme de|schnapps|peach schnapps|"
   "coffee liqueur|orange liqueur|chambord|limoncello|blue curacao|sambuca|ouzo|midori|galliano",
   320, 0.0, 35.0, 0.2, 0.0, d=1.05, portion=("fl_oz",))
_f("soda|cola|ginger ale|lemon-lime soda|sprite|7-up|root beer|tonic water|cream soda|coca-cola",
   41, 0.0, 10.6, 0.0, 0.0, d=1.04, pkg=355, portion=CUP)
_f("diet ginger ale|diet soda|diet cola|diet coke|diet sprite|sugar-free soda", 0, 0.0, 0.0, 0.0, 0.0,
   d=1.0, pkg=355, portion=CUP)
_f("liquid smoke|kitchen bouquet|browning sauce|food coloring|red food coloring|green food coloring",
   0, 0.0, 0.0, 0.0, 0.0, d=1.0, portion=TSP)

# --- fruit -------------------------------------------------------------------
_f("apple|tart apple|granny smith apple|green apple|red apple|gala apple|fuji apple|golden delicious apple|"
   "honeycrisp apple|cooking apple", 52, 0.3, 13.8, 0.2, 2.4, each=182, d=0.53)
_f("applesauce|unsweetened applesauce|apple sauce|sweetened applesauce", 42, 0.2, 11.3, 0.1, 1.1, d=1.03,
   portion=CUP)
_f("banana|ripe banana|bananas|overripe banana|mashed banana|frozen banana", 89, 1.1, 22.8, 0.3, 2.6,
   each=118, d=0.95)
_f("strawberry|fresh strawberry|frozen strawberry|sliced strawberry", 32, 0.7, 7.7, 0.3, 2.0, each=12,
   d=0.6, portion=("cup", 6, "each"))
_f("blueberry|fresh blueberry|frozen blueberry|raspberry|fresh raspberry|frozen raspberry|blackberry|"
   "berry|mixed berry|cranberry|fresh cranberry|frozen cranberry|boysenberry|frozen mixed berry",
   50, 1.0, 12.0, 0.4, 4.0, d=0.6, portion=CUP)
_f("pineapple|fresh pineapple|crushed pineapple|pineapple chunk|pineapple tidbit|canned pineapple|"
   "pineapple ring", 50, 0.5, 13.1, 0.1, 1.4, d=0.7, pkg=567, portion=("pkg", 3, "cup"))
_f("mango|mangoes|ripe mango|frozen mango", 60, 0.8, 15.0, 0.4, 1.6, each=200, d=0.7)
_f("peach|pear|plum|nectarine|kiwi|kiwi fruit|apricot|fig|fresh fig|persimmon|papaya|guava|pomegranate",
   50, 0.7, 13.0, 0.2, 2.5, each=150, d=0.65)
_f("grape|red grape|green grape|seedless grape|red seedless grape|green seedless grape|seedless red grape",
   69, 0.7, 18.1, 0.2, 0.9, d=0.64, portion=CUP)
_f("raisin|golden raisin|currant|dried currant|dried cranberry|craisin|sultana|dried cherry|dried blueberry",
   299, 3.1, 79.2, 0.5, 3.7, d=0.62, portion=CUP)
_f("date|pitted date|medjool date|dried fig|prune|dried prune|dried apricot|dried pear half|dried fruit|"
   "dried mango", 282, 2.5, 75.0, 0.4, 8.0, each=8, d=0.62, portion=("cup", 4, "each"))
_f("watermelon|cantaloupe|melon|honeydew|honeydew melon", 30, 0.6, 7.6, 0.2, 0.4, d=0.65, each=1500,
   portion=CUP)
_f("cherry|sweet cherry|maraschino cherry|sour cherry|frozen cherry|cherry pie filling|pie filling",
   100, 0.8, 25.0, 0.2, 1.6, each=5, d=0.65, portion=CUP)
_f("rhubarb", 21, 0.9, 4.5, 0.2, 1.8, d=0.5, portion=CUP)
_f("coconut water", 19, 0.7, 3.7, 0.2, 1.1, d=1.02, portion=CUP)


def _tokens(text: str) -> list[str]:
    import re

    text = text.lower().replace("&quot;", " ")
    words = re.findall(r"[a-z0-9][a-z0-9'%-]*", text)
    return [singularize(w.strip("'-")) for w in words if w.strip("'-")]


_INDEX: dict[str, list[tuple[tuple[str, ...], Food]]] = {}
for _food in FOODS:
    for _alias in _food.aliases:
        _tok = tuple(_tokens(_alias))
        if _tok:
            _INDEX.setdefault(_tok[0], []).append((_tok, _food))


def match_food(name: str) -> Food | None:
    """Longest alias that appears as a contiguous token run; ties go to the rightmost
    (the head noun in English noun phrases is usually last)."""
    tokens = _tokens(name)
    best: tuple[int, int, Food] | None = None
    for i, tok in enumerate(tokens):
        for alias_tokens, food in _INDEX.get(tok, ()):
            n = len(alias_tokens)
            if tuple(tokens[i:i + n]) == alias_tokens:
                key = (n, i)
                if best is None or key > best[:2]:
                    best = (n, i, food)
    return best[2] if best else None


def match_ingredient_nutrition(ingredient_raw: str) -> dict[str, float] | None:
    """Per-100 g nutrients of the best matching food, or None when nothing matches."""
    food = match_food(ingredient_raw)
    if food is None:
        return None
    return dict(zip(NUTRIENT_KEYS, food.per100, strict=True))


def _unit_grams(unit: str, food: Food, parsed: ParsedIngredient) -> tuple[float | None, bool]:
    """Grams for one `unit` of `food`. Second value is True when a fallback was used."""
    if unit in MASS_UNITS_G:
        return MASS_UNITS_G[unit], False
    if unit in VOLUME_UNITS_ML:
        if food.density is None:
            return VOLUME_UNITS_ML[unit] * FALLBACK_DENSITY, True
        return VOLUME_UNITS_ML[unit] * food.density, False
    if unit in CONTAINER_UNITS or unit == "pkg":
        if parsed.size_quantity and parsed.size_unit:
            size_g, assumed = _unit_grams(parsed.size_unit, food, parsed)
            if size_g is not None:
                return parsed.size_quantity * size_g, assumed
        if food.pkg_g:
            return food.pkg_g, False
        if food.each_g:
            return food.each_g, True
        return None, True
    if unit in food.unit_g:
        return food.unit_g[unit], False
    if unit in {"each", "piece"}:
        if parsed.size_quantity and parsed.size_unit and parsed.size_unit != "inch":
            size_g, assumed = _unit_grams(parsed.size_unit, food, parsed)
            if size_g is not None:
                return parsed.size_quantity * size_g, assumed
        if food.each_g:
            return food.each_g, False
        if food.pkg_g:
            return food.pkg_g, True
        return None, True
    if unit in _GENERIC_UNIT_G:
        return _GENERIC_UNIT_G[unit], True
    return None, True


def _portion_unit(food: Food, quantity: float) -> str:
    rule = food.portion
    if len(rule) == 3:
        below, threshold, above = rule
        return below if quantity < threshold else above
    return rule[0]


def _bare_number_unit(food: Food, quantity: float) -> str:
    """Unit for a quantity stored without one (Food.com "portion").

    Besides the per-food rule, large numbers are read as weights: 20 or more of a food
    normally measured by volume is grams (metric recipes: "flour 125"), 4-20 of a canned
    food is the can size in ounces ("kidney beans 15 1/2"), and 5-20 cups is more likely
    tablespoons ("sugar 12").
    """
    unit = _portion_unit(food, quantity)
    by_volume = unit in VOLUME_UNITS_ML or unit == "pkg"
    if by_volume and quantity >= 20:
        return "g"
    if unit == "pkg" and quantity >= 4:
        return "oz"
    if unit == "cup" and quantity >= 5:
        return "tbsp"
    return unit


def estimate_grams(parsed: ParsedIngredient, food: Food) -> tuple[float | None, str]:
    """Return (grams, status). Status is "estimated", "assumed_unit" or "assumed_quantity"."""
    quantity = parsed.quantity_mean
    status = "estimated"
    if quantity is None or quantity <= 0:
        quantity = 1.0
        status = "assumed_quantity"
    unit = parsed.unit
    if unit is None:
        unit = _bare_number_unit(food, quantity)
        if status == "estimated" and unit != "each":
            status = "assumed_unit"
    grams_per_unit, fallback = _unit_grams(unit, food, parsed)
    if grams_per_unit is None:
        return None, "unknown"
    if fallback and status == "estimated":
        status = "assumed_unit"
    grams = min(quantity * grams_per_unit, MAX_GRAMS_PER_INGREDIENT)
    return grams, status


def _zero() -> dict[str, float]:
    return {k: 0.0 for k in NUTRIENT_KEYS}


def _parse_all(ingredients: Iterable[str], ingredient_details: list[dict[str, Any]] | None) -> list[tuple[str, ParsedIngredient]]:
    if ingredient_details:
        out = []
        for d in ingredient_details:
            if not isinstance(d, dict) or not str(d.get("name") or "").strip():
                continue
            out.append((str(d["name"]), parse_ingredient_detail(d)))
        return out
    return [(ing, parse_ingredient(ing)) for ing in ingredients if ing and str(ing).strip()]


def calculate_recipe_nutrition(
    ingredients: list[str],
    servings: int | None = None,
    ingredient_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate total and per-serving macronutrients.

    `ingredients` may be bare names or full lines ("2 cups flour"). When
    `ingredient_details` (records with name/quantity/unit) is given it is used instead, so
    stored quantities are taken into account.

    Ingredients that match no food are listed under coverage["unknown"] and add nothing.
    """
    servings_count = servings if (servings and servings > 0) else DEFAULT_SERVINGS
    totals = _zero()
    breakdown: list[dict[str, Any]] = []
    unknown: list[str] = []
    assumed = 0

    for label, parsed in _parse_all(ingredients, ingredient_details):
        food = match_food(parsed.raw) or (match_food(parsed.name) if parsed.name else None)
        grams: float | None = None
        status = "unknown"
        values = _zero()
        if food is not None:
            grams, status = estimate_grams(parsed, food)
            if grams is not None:
                values = {k: v * grams / 100.0 for k, v in zip(NUTRIENT_KEYS, food.per100, strict=True)}
        if grams is None:
            status = "unknown"
            unknown.append(label)
        elif status != "estimated":
            assumed += 1
        for k in NUTRIENT_KEYS:
            totals[k] += values[k]
        breakdown.append({
            "ingredient": label,
            **{k: round(values[k], 1) for k in NUTRIENT_KEYS},
            "matched_food": food.key if (food is not None and grams is not None) else None,
            "grams": round(grams, 1) if grams is not None else None,
            "status": status,
        })

    total_count = len(breakdown)
    matched = total_count - len(unknown)
    return {
        "servings": servings_count,
        "per_serving": {k: round(totals[k] / servings_count, 1) for k in NUTRIENT_KEYS},
        "total": {k: round(totals[k], 1) for k in NUTRIENT_KEYS},
        "breakdown": breakdown,
        "coverage": {
            "ratio": round(matched / total_count, 3) if total_count else 0.0,
            "matched": matched,
            "total": total_count,
            "unknown": unknown,
            "assumed_quantity_or_unit": assumed,
        },
    }
