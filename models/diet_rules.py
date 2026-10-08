"""
Rule-based diet compatibility tagging.
Real nutrition apps use exactly this hybrid: ML for cuisine, rules for dietary flags.
"""

MEAT_FISH_KEYWORDS = {
    "chicken", "beef", "pork", "bacon", "ham", "turkey", "lamb", "veal",
    "sausage", "salami", "pepperoni", "prosciutto", "duck", "fish", "salmon",
    "tuna", "shrimp", "prawn", "crab", "lobster", "anchovy", "anchovies",
    "squid", "octopus", "clam", "mussel", "oyster", "scallop", "gelatin",
}

DAIRY_EGG_KEYWORDS = {
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "egg", "eggs",
    "mayonnaise", "ghee", "paneer", "mozzarella", "parmesan", "cheddar",
    "ricotta", "custard", "honey",
}

GLUTEN_KEYWORDS = {
    "flour", "wheat", "bread", "pasta", "noodle", "noodles", "spaghetti",
    "barley", "rye", "couscous", "semolina", "breadcrumb", "breadcrumbs",
    "soy sauce", "beer", "malt", "cracker", "tortilla", "pita",
}

HIGH_CARB_KEYWORDS = {
    "rice", "sugar", "flour", "bread", "pasta", "potato", "potatoes",
    "corn", "noodle", "noodles", "honey", "syrup", "banana", "oats",
}


def _contains_any(ingredients_text, keyword_set):
    return any(kw in ingredients_text for kw in keyword_set)


def tag_diet(ingredients):
    """
    ingredients: list of raw ingredient strings
    returns: dict of diet compatibility flags
    """
    text = " ".join(ingredients).lower()

    has_meat_fish = _contains_any(text, MEAT_FISH_KEYWORDS)
    has_dairy_egg = _contains_any(text, DAIRY_EGG_KEYWORDS)
    has_gluten = _contains_any(text, GLUTEN_KEYWORDS)
    high_carb = _contains_any(text, HIGH_CARB_KEYWORDS)

    return {
        "vegetarian": not has_meat_fish,
        "vegan": not has_meat_fish and not has_dairy_egg,
        "gluten_free": not has_gluten,
        "keto_friendly": not high_carb and not has_gluten,
    }
