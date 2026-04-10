"""
Shared lexicons for the ARMADA bias detection pipeline.
Composed as pseudo-human-annotator output for validation.

In production, target/contrast groups are researcher-defined;
frame taxonomy is populated from undirected PPMI discovery (Step 0)
then classified by expert annotators (Step 1).

"""

# ── Target demographic tokens (lemmatized) ────────────────────────────────────
# Covers immigrant/refugee, named ethnic groups, and broader minority framing.
# Intentionally excludes highly polysemous colour/direction words from this set.

TARGET_TOKENS = {
    # immigration / legal status
    "immigrant", "immigrants", "immigration",
    "refugee", "refugees",
    "migrant", "migrants",
    "asylum",
    "undocumented",
    "expat", "expatriate", "expatriates",
    "diaspora",
    "stateless",
    "deportee", "deportees",

    # foreignness framing
    "foreigner", "foreigners",
    "overseas",
    "alien",                 # legal/rhetorical sense

    # ethnic / racial minority (named groups)
    "asian", "chinese", "japanese", "korean", "vietnamese", "filipino",
    "indian", "hindi", "russian", "balkan",
    "arab", "arabic",
    "muslim", "islamic",
    "jewish", "jew",
    "hispanic", "latino", "latina", "latinx",
    "mexican", "mexicano",
    "african", "nigger", "negro", "black",

    # Middle East / North Africa
    "egyptian", "turkish", "iranian", "iraqi", "syrian", "lebanese",
    "palestinian", "yemeni", "libyan", "tunisian", "algerian", "moroccan",
    "jordanian", "bahraini", "emirati", "kuwaiti", "omani", "qatari",

    # Sub-Saharan Africa
    "somali", "ethiopian", "eritrean", "sudanese", "nigerian", "ghanaian",
    "kenyan", "ugandan", "tanzanian", "congolese", "zimbabwean",
    "rwandan", "senegalese", "cameroonian", "malian",

    # South Asia
    "pakistani", "bangladeshi", "nepali", "afghan", "lankan",

    # Southeast Asia
    "thai", "indonesian", "malaysian", "burmese", "cambodian", "laotian",

    # East Asia (supplement)
    "taiwanese", "mongolian", "tibetan",

    # Latin America / Caribbean
    "cuban", "colombian", "venezuelan", "peruvian", "bolivian",
    "ecuadorian", "guatemalan", "honduran", "salvadoran", "nicaraguan",
    "haitian", "jamaican", "dominican", "brazilian",
    "chilean", "argentinian", "uruguayan", "paraguayan",

    # Eastern Europe (non-EU-core / historically marginalized)
    "polish", "romanian", "hungarian", "czech", "slovak", "bulgarian",
    "serbian", "croatian", "bosnian", "albanian", "ukrainian",
    "belarusian", "moldovan", "georgian", "armenian",

    # Roma / stateless / broad
    "romani", "sinti", "kurdish", 

    # political minority
    "soviet", "ussr", "communist",

    # broader minority framing
    "minority", "minorities",
    "ethnic", 
    "nonwhite", "non-white", 
    "indigenous", "aboriginal", "islander", "native",
    "colored",               # historical/SAEE register; flags attitudinal context
    "poc", "brown", "yellow",                # person/people of colour (abbreviated)
    "biracial", "multiracial",
    "marginalized",
}

# ── Contrast (dominant/majority) tokens ───────────────────────────────────────

CONTRAST_TOKENS = {
    # named dominant-group demonyms
    "european", "american", "british", "western",
    "white", "caucasian",
    "anglophone",

    # Western Europe
    "french", "german", "dutch", "belgian", "swiss", "austrian",
    "italian", "spanish", "portuguese",

    # Scandinavia
    "swedish", "norwegian", "danish", "finnish", "icelandic",

    # Anglosphere
    "australian", "canadian", "irish", "scottish",

    # Southern Europe / other historically dominant
    "greek", "israeli",

    # civic / majority framing
    "citizen", "citizens",
    "national",              # ⚠ polysemous
    "local", "locals",       # ⚠ polysemous
    "native",                # ⚠ polysemous (native speaker vs. Native American)
    "domestic",
    "mainstream",
    "majority",
    "native-born",

    # ideological / historical labels used contrastively in corpus
    "settler", "settlers",
    "colonist", "colonists",
    "conservatist",

}

# Context-window disambiguation defaults.
GROUP_CONTEXT_WINDOW = 3
SEMANTIC_CONTEXT_WINDOW = 6

# Minimal modifier maps for ambiguous human nouns like "citizen" or "national".
# These replace the older phrase inventory and avoid singular/plural duplication.
TARGET_NOUN_MODIFIER_MAP = {
    "foreign": "foreign",
}

CONTRAST_NOUN_MODIFIER_MAP = {
    "native-born": "native-born",
    "local": "local",
    "american": "american",
    "british": "british",
    "european": "european",
}

AMBIGUOUS_TARGET_MODIFIERS = {"foreign", "black", "polish"}
AMBIGUOUS_CONTRAST_MODIFIERS = {
    "american", "british", "european", "western",
    "white", "caucasian", "anglophone",
    "local", "native", "domestic", "mainstream",
}
AMBIGUOUS_NOUNS = {"citizen", "national", "native"}
SEMANTIC_DISAMBIGUATION_TOKENS = (
    AMBIGUOUS_TARGET_MODIFIERS
    | AMBIGUOUS_CONTRAST_MODIFIERS
    | {"american", "british", "european", "western", "foreign", "black", "native", "national", "polish"}
)

STRONG_TARGET_TOKENS = TARGET_TOKENS - AMBIGUOUS_TARGET_MODIFIERS
STRONG_CONTRAST_TOKENS = CONTRAST_TOKENS - (
    AMBIGUOUS_CONTRAST_MODIFIERS
    | AMBIGUOUS_NOUNS
    | {"citizens", "nationals", "locals"}
)

# Tokens too broad to trigger the lexical gate on their own.
# Still resolved when found in sentences that entered via other tokens.
GATE_EXCLUDE_TOKENS = {
    "citizen", "citizens",
    "national", "nationals",
    "local", "locals",
    "native",
    "domestic",
    "mainstream",
    "majority",
    "minority", "minorities",
    "native-born",
}

_SEMANTIC_GROUP_RESOLVER = None


def set_semantic_group_resolver(resolver) -> None:
    global _SEMANTIC_GROUP_RESOLVER
    _SEMANTIC_GROUP_RESOLVER = resolver


_MODIFIER_DEPS = {"amod", "compound", "appos", "flat", "npadvmod"}
_GROUP_PREFIX_TOKENS = {"non", "anti", "pro"}


def _normalize_surface(text: str) -> str:
    return (
        text.lower()
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _candidate_forms(form: str):
    seen = set()
    for candidate in (
        form,
        form.rstrip("."),
        form[:-3] + "y" if form.endswith("ies") and len(form) > 3 else None,
        form[:-2] if form.endswith("es") and len(form) > 2 else None,
        form[:-1] if form.endswith("s") and len(form) > 1 else None,
    ):
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _match_inventory(form: str, inventory: set[str]) -> str | None:
    for candidate in _candidate_forms(_normalize_surface(form)):
        if candidate in inventory:
            return candidate
    return None


def _human_noun_form(token) -> str | None:
    return _match_inventory(token.lemma_, HUMAN_NOUNS) or _match_inventory(token.text, HUMAN_NOUNS)


def _inanimate_noun_form(token) -> str | None:
    return _match_inventory(token.lemma_, INANIMATE_NOUNS) or _match_inventory(token.text, INANIMATE_NOUNS)


def _group_base(token) -> str | None:
    inventories = (
        TARGET_TOKENS,
        CONTRAST_TOKENS,
        AMBIGUOUS_TARGET_MODIFIERS,
        AMBIGUOUS_CONTRAST_MODIFIERS,
        AMBIGUOUS_NOUNS,
    )
    for inventory in inventories:
        match = _match_inventory(token.lemma_, inventory) or _match_inventory(token.text, inventory)
        if match:
            return match
    return None


def _group_side(base: str | None) -> str | None:
    if base is None:
        return None
    if base in TARGET_TOKENS or base in AMBIGUOUS_TARGET_MODIFIERS:
        return "minority"
    if base in CONTRAST_TOKENS or base in AMBIGUOUS_CONTRAST_MODIFIERS or base in AMBIGUOUS_NOUNS:
        return "dominant"
    return None


def _has_group_prefix(token, doc) -> bool:
    left = max(0, token.i - 2)
    for idx in range(left, token.i):
        candidate = _normalize_surface(doc[idx].text)
        if candidate in _GROUP_PREFIX_TOKENS:
            return True
    return False


def _same_head_group_modifiers(token, doc) -> list:
    head = token.head
    siblings = []
    for sibling in head.children:
        if sibling.i == token.i or sibling.dep_ not in _MODIFIER_DEPS:
            continue
        if _group_base(sibling):
            siblings.append(sibling)
    return siblings

# ── Mental-state verbs (gates SI) ──

SUBJECTIVE_VERBS = {
    "think", "believe", "feel", "hope", "fear", "decide", "plan",
    "assume", "consider", "wish", "expect", "want", "imagine",
    "prefer", "suspect", "doubt", "insist", "know", "realize",
    "understand", "recognize", "anticipate", "dread", "desire",
    "trust", "distrust", "worry", "wonder", "speculate",
}

# ── Psych-verb stems (object-experiencer verbs) ──
# Participle forms (horrified, delighted...) describe an EXPERIENCER state,
# not a passive patient. Used for (a) AUX chain guard and (b) AttI routing.

NEG_PSYCH_VERB_STEMS = {
    "horrify", "terrify", "frighten", "alarm", "shock", "appall",
    "disgust", "disturb", "distress", "upset", "trouble", "worry",
    "unsettle", "embarrass", "humiliate", "shame", "intimidate",
    "disappoint", "frustrate", "discourage", "depress",
    "bore", "tire", "exhaust", "overwhelm",
    "irritate", "annoy", "offend", "outrage",
    "confuse", "bewilder", "puzzle", "perplex", "baffle",
}

POS_PSYCH_VERB_STEMS = {
    "please", "delight", "excite", "inspire", "impress",
    "fascinate", "captivate", "enthrall", "intrigue",
    "satisfy", "comfort", "reassure", "encourage", "flatter",
    "amaze", "astonish", "astound", "thrill", "charm",
    "move", "touch", "attract",
}

PSYCH_VERB_STEMS = NEG_PSYCH_VERB_STEMS | POS_PSYCH_VERB_STEMS

# ── Attitudinal adjectives (primary, non-participial) ──
# For cop+adj constructions: "immigrants are afraid/proud"

NEG_ATTITUDINAL_ADJ = {
    "afraid", "fearful", "anxious", "desperate", "hopeless", "helpless",
    "ashamed", "miserable", "angry", "furious", "resentful",
    "suspicious", "hostile", "isolated", "vulnerable", "powerless",
    "insecure", "unsafe", "unwelcome", "unwanted", "misunderstood",
}

POS_ATTITUDINAL_ADJ = {
    "proud", "hopeful", "confident", "determined", "grateful",
    "optimistic", "resilient", "empowered", "secure",
    "satisfied", "content", "motivated", "enthusiastic",
    "ambitious", "capable", "respected", "valued", "wonderful", "welcoming",
}

# Combined sets: adj + psych-verb stems (for extraction matching)
NEG_ATTITUDINAL = NEG_ATTITUDINAL_ADJ | NEG_PSYCH_VERB_STEMS
POS_ATTITUDINAL = POS_ATTITUDINAL_ADJ | POS_PSYCH_VERB_STEMS

# ── Classified frame taxonomy ──
# Pseudo-annotator classification of metaphorical collocates.
# In production, populated from undirected PPMI discovery + expert annotation.

CLASSIFIED_FRAMES = {
    "natural_disaster": {
        "sign": -1,
        "terms": {
            "flood", "flooded", "wave", "surge", "tide", "deluge",
            "torrent", "overflow", "drown", "inundate",
        },
    },
    "animal_dehumanization": {
        "sign": -1,
        "terms": {
            "swarm", "horde", "hordes", "flock", "pack", "breed",
            "nest", "infest", "infestation",
        },
    },
    "invasion": {
        "sign": -1,
        "terms": {
            "invade", "overwhelm", "overrun", "pour", "descend",
            "infiltrate", "penetrate", "encroach",
        },
    },
    "threat_burden": {
        "sign": -1,
        "terms": {
            "threat", "threaten", "burden", "strain", "drain",
            "influx", "crisis", "collapse",
        },
    },
    "contribution": {
        "sign": +1,
        "terms": {
            "contribute", "enrich", "strengthen", "benefit",
            "boost", "innovate", "thrive", "prosper",
        },
    },
    "integration": {
        "sign": +1,
        "terms": {
            "welcome", "integrate", "include", "embrace",
            "accept", "empower",
        },
    },
    "building": {
        "sign": +1,
        "terms": {
            "build", "create", "establish", "found", "develop",
            "provide", "launch",
        },
    },
}

# Flat lookups derived from taxonomy
ALL_FRAME_TERMS: set[str] = set()
FRAME_SIGN: dict[str, int] = {}

for _frame_type, _info in CLASSIFIED_FRAMES.items():
    for _term in _info["terms"]:
        ALL_FRAME_TERMS.add(_term)
        FRAME_SIGN[_term] = _info["sign"]


# ── Inanimate head nouns (shared: Layer 2 sentence filter + Step 3 role guard) ─
# Expanded from Step 3's original set to cover common false-positive head nouns
# for polysemous group adjectives (national _park_, black _hole_, etc.).

INANIMATE_NOUNS = {
    # nature / geography / weather
    "park", "hole", "ocean", "sea", "river", "mountain", "desert",
    "forest", "lake", "valley", "island", "beach", "weather",
    "climate", "sky", "soil", "spring", "creek",
    # food / cuisine
    "food", "cheese", "tea", "coffee", "rice", "cuisine", "dish",
    "recipe", "meal", "restaurant", "menu", "spice",
    # objects / technology / infrastructure
    "car", "vehicle", "phone", "computer", "machine", "device",
    "building", "house", "bridge", "road", "highway", "wall",
    "door", "furniture", "equipment", "tool",
    # media / arts / entertainment
    "film", "movie", "show", "game", "sport", "song", "album",
    "book", "novel", "story", "music", "art", "painting",
    "football", "baseball", "basketball", "soccer", "chess",
    # economic / policy / governance
    "policy", "law", "bill", "regulation", "statute",
    "market", "economy", "stock", "bond", "rate", "price",
    "cost", "wage", "salary", "tax", "fund", "budget", "debt",
    "product", "brand", "patent", "trade", "tariff",
    "system", "network", "platform", "software", "algorithm",
    "data", "database", "report", "study", "survey",
    "plan", "program", "project", "grant", "scheme",
    "resource", "service", "supply", "demand",
    "check", "account", "record", "document", "form",
    # geographic / political units (the entity, not its people)
    "city", "country", "area", "region", "border", "territory",
    "zone", "sector", "district", "coast", "continent", "frontier",
    # organizational (non-person)
    "job", "position", "role", "company", "corporation",
    "agency", "department", "office", "institution", "industry",
    # scientific / medical
    "medicine", "drug", "treatment", "therapy", "surgery",
    "species", "gene", "cell", "virus", "disease",
    # abstract / miscellaneous
    "interest", "affair", "standard", "exchange",
    "security", "defense", "defence", "anthem", "flag",
    "dream", "box", "list", "mail", "express",
    "summer", "winter", "bear", "cat",
    "violence", "abuse",
    "school", "university", "college", "church", "hospital",
    "library", "store", "shop", "station", "airport",
    "war", "media", "newspaper",
}

INANIMATE_ENTITY_TYPES = (
    "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART",
    "DATE", "TIME", "MONEY", "QUANTITY", "CARDINAL",
    "ORDINAL", "LAW", "LANGUAGE", "FAC",
)

# ── Human-referent head nouns (Layer 2 positive signal, future use) ────────────

HUMAN_NOUNS = {
    "person", "people", "man", "woman", "men", "women",
    "child", "children", "kid", "boy", "girl",
    "adult", "youth", "elder", "teenager", "baby",
    "community", "population", "group", "family", "household",
    "tribe", "clan", "folk",
    "worker", "employee", "laborer", "labourer", "staff",
    "resident", "inhabitant", "citizen", "voter", "taxpayer",
    "student", "teacher", "professor", "scholar", "researcher",
    "doctor", "nurse", "lawyer", "engineer", "scientist",
    "leader", "official", "politician", "activist", "advocate",
    "soldier", "veteran", "prisoner", "inmate", "detainee",
    "farmer", "merchant", "trader", "entrepreneur",
    "artist", "writer", "musician", "journalist", "reporter",
    "tourist", "traveler", "traveller", "visitor", "speaker",
    "owner", "employer", "manager", "director", "executive",
    "patient", "victim", "survivor", "witness",
    "suspect", "criminal", "offender", "convict",
    "consumer", "customer", "client",
    "member", "participant", "delegate", "representative",
    "neighbor", "neighbour", "colleague", "peer",
    "ancestor", "descendant",
    "immigrant", "refugee", "migrant", "settler", "colonist",
    "foreigner", "native", "exile",
    "minority", "majority",
}


def resolve_group_token(token, doc, context_window: int = GROUP_CONTEXT_WINDOW):
    """
    Resolve a token to a demographic group label with a small local context window.

    Returns:
        (group_type, canonical_lemma) where group_type ∈ {"minority", "dominant"}
        or None when the token should not count as a group mention.
    """
    group_base = _group_base(token)
    lemma = group_base or _normalize_surface(token.lemma_)
    head = token.head
    head_lemma = _normalize_surface(head.lemma_) if head is not None else ""
    head_group = _group_base(head) if head is not None and head != token else None

    left = max(0, token.i - context_window)
    right = min(len(doc), token.i + context_window + 1)
    window = [t for t in doc[left:right] if t.i != token.i]
    window_lemmas = {_normalize_surface(t.lemma_) for t in window}

    is_modifier = token.dep_ in _MODIFIER_DEPS
    human_head = _human_noun_form(head) is not None
    inanimate_head = _inanimate_noun_form(head) is not None or head.ent_type_ in INANIMATE_ENTITY_TYPES
    nounish = token.pos_ in ("NOUN", "PROPN")

    if _has_group_prefix(token, doc):
        if is_modifier and head_group:
            return None

    # Modifier + group-head compounds: "African American", "Asian Americans".
    if group_base and is_modifier and head_group and lemma != head_group:
        side = _group_side(lemma)
        if side == "dominant" and _group_side(head_group) == "minority":
            return None
        return (side or _group_side(head_group), f"{lemma}-{head_group}")

    # Conjoined modifier in a grouped mention: "Asian and Latino Americans".
    if group_base and token.dep_ == "conj" and token.head.dep_ in _MODIFIER_DEPS and head.head != head:
        outer_head = head.head
        outer_group = _group_base(outer_head)
        if outer_group and outer_group != lemma:
            return (_group_side(lemma) or _group_side(outer_group), f"{lemma}-{outer_group}")

    # Suppress ambiguous modifiers when the same human head already has a more
    # specific group modifier, e.g. "local African American population".
    if (
        lemma in AMBIGUOUS_CONTRAST_MODIFIERS
        and is_modifier
        and human_head
        and not inanimate_head
        and _same_head_group_modifiers(token, doc)
    ):
        return None

    # Suppress group heads already represented by a composite modifier, e.g. the
    # "American" in "African American population".
    if is_modifier and human_head and not inanimate_head:
        if any(_group_base(child) and _group_base(child) != lemma for child in token.children):
            return None

    # Citizen/national patterns.
    if lemma in AMBIGUOUS_NOUNS:
        if lemma in {"citizen", "citizens"} and "native" in window_lemmas and "bear" in window_lemmas:
            return ("dominant", "native-born")
        for modifier, canonical in TARGET_NOUN_MODIFIER_MAP.items():
            if modifier in window_lemmas:
                return ("minority", canonical)
        for modifier, canonical in CONTRAST_NOUN_MODIFIER_MAP.items():
            if modifier in window_lemmas:
                return ("dominant", canonical)
        if lemma == "native":
            return None
        if nounish:
            return ("dominant", lemma)
        return None

    # Strong unambiguous tokens remain countable without extra context.
    if lemma in STRONG_TARGET_TOKENS:
        if is_modifier and not head_group and not (human_head and not inanimate_head):
            return None
        return ("minority", lemma)
    if lemma in STRONG_CONTRAST_TOKENS:
        if is_modifier and not head_group and not (human_head and not inanimate_head):
            return None
        return ("dominant", lemma)

    # Ambiguous modifiers require a human-referent head.
    if lemma in AMBIGUOUS_TARGET_MODIFIERS:
        if is_modifier and human_head and not inanimate_head:
            return ("minority", lemma)
        if is_modifier and inanimate_head:
            return None
        if nounish and lemma == "black":
            return ("minority", lemma)
        return None

    if lemma in AMBIGUOUS_CONTRAST_MODIFIERS:
        # "native Arab" should count as Arab, not as dominant.
        if is_modifier and head_lemma in TARGET_TOKENS:
            return None
        if lemma == "native" and head_lemma == "bear" and head.head.lemma_.lower() == "citizen":
            return ("dominant", "native-born")
        if is_modifier and human_head and not inanimate_head:
            return ("dominant", lemma)
        if is_modifier and inanimate_head:
            return None
        if nounish and not is_modifier:
            if any(
                _group_base(child)
                and _group_base(child) != lemma
                and not _has_group_prefix(child, doc)
                for child in token.children
            ):
                return None
            return ("dominant", lemma)
        if _SEMANTIC_GROUP_RESOLVER and lemma in SEMANTIC_DISAMBIGUATION_TOKENS:
            return _SEMANTIC_GROUP_RESOLVER(token, doc)
        return None

    if _SEMANTIC_GROUP_RESOLVER and lemma in SEMANTIC_DISAMBIGUATION_TOKENS:
        return _SEMANTIC_GROUP_RESOLVER(token, doc)

    return None
