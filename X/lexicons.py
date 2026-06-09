"""Shared active lexicons for the ARMADA bias detection pipeline."""

# ── Target demographic tokens (lemmatized) ────────────────────────────────────
# Covers immigrant/refugee, named ethnic groups, and broader minority framing.

TARGET_TOKENS = {
    # immigration / legal status
    "immigrant", "immigrants",
    "refugee", "refugees",
    "migrant", "migrants",
    "asylum", "asylum-seeker", "asylum-seekers",
    "undocumented",
    "expat", "expatriate", "expatriates",
    "diaspora",
    "stateless",
    "deportee", "deportees",

    # foreignness framing
    "foreigner", "foreigners",
    "alien",                 # legal/rhetorical sense

    # ethnic / racial minority (named groups)
    "asian", "chinese", "japanese", "korean", "vietnamese", "filipino",
    "indian", "hindi", "russian", "balkan",
    "arab", "arabic",
    "muslim", "islamic",
    "jewish", "jew", "zionist",
    "hispanic", "latino", "latina", "latinx", "spanish",
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
    "thai", "indonesian", "malay", "malaysian", "burmese", "cambodian", "laotian",

    # East Asia (supplement)
    "taiwanese", "mongolian", "tibetan",

    # Latin America / Caribbean
    "cuban", "colombian", "venezuelan", "peruvian", "bolivian",
    "ecuadorian", "guatemalan", "honduran", "salvadoran", "nicaraguan",
    "haitian", "jamaican", "dominican", "brazilian", "latinx",
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
    "indigenous", "aboriginal", "aborigine", "islander",
    "colored", "dark", "darker",            # historical/SAEE register
    "poc", "brown", "yellow",
    "biracial", "multiracial",
    "marginalized", "underrepresented",

    # compound canonical forms
    "native-american",
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

    # ideological / historical labels used contrastively in corpus
    "settler",
    "colonist",

    # skin-color
    "whiter",
}

# Political / ideological labels are resolved and reported separately from the
# demographic minority/dominant contrast. They may remain useful corpus signals,
# but should not silently become evidence for demographic framing claims.
POLITICAL_GROUP_TOKENS = {"soviet", "ussr", "communist", "conservatist"}

# Spaced/prepositional compounds that need explicit handling because spaCy
# tokenizes them as separate tokens. Each entry is keyed by the *head* lemma
# (the noun spaCy assigns the dependency root of the noun phrase to). The
# value spec describes how to detect the compound around that head and what
# canonical lemma + group side to emit.
COMPOUND_TARGET_HEADS: dict[str, dict] = {
    # "asylum seeker(s)" — compound: asylum -> seeker
    "seeker":  {"compound_lemma": "asylum", "canonical": "asylum-seeker", "side": "minority"},
    "seekers": {"compound_lemma": "asylum", "canonical": "asylum-seeker", "side": "minority"},
    # "people of color" — pobj 'color' under prep 'of' under head 'people'.
    "people":  {"prep_lemma": "of", "prep_pobj_lemma": "color", "canonical": "people-of-color", "side": "minority"},
    # "native American(s)" — compound: native -> American (spaCy: native amod of Americans)
    "american":  {"compound_lemma": "native", "canonical": "native-american", "side": "minority"},
    "americans": {"compound_lemma": "native", "canonical": "native-american", "side": "minority"},
}

# Context-window disambiguation defaults. The rule-based group window stays
# local; the semantic resolver window can be wider with GTE ModernBERT.
GROUP_CONTEXT_WINDOW = 4
SEMANTIC_CONTEXT_WINDOW = 24

# Tokens too broad to trigger the lexical gate on their own.
# Still resolved when found in sentences that entered via other tokens.
GATE_EXCLUDE_TOKENS = {
    "minority", "minorities",
    "mainstream",
}


# When set, resolve_group_token returns None for any canonical lemma not in
# this set.  Populated by run_pipeline from dolma/active_labels.json so that
# Phase 2 mention resolution is restricted to the same token set that Phase 1
# used for extraction (top-8 target + top-8 contrast by corpus frequency).
# None means no restriction (default; used by dim_sanity.py and unit callers).
_ACTIVE_EXTRACTION_TOKENS: "set[str] | None" = None


def set_active_extraction_tokens(tokens: "set[str] | None") -> None:
    """Restrict resolve_group_token to the given canonical-lemma set.

    Pass None to lift the restriction (default behaviour).
    """
    global _ACTIVE_EXTRACTION_TOKENS
    _ACTIVE_EXTRACTION_TOKENS = tokens


_MODIFIER_DEPS = {"amod", "compound", "appos", "flat", "npadvmod"}
_NEGATION_PREFIXES = {"non"}
_STANCE_PREFIXES = {"anti", "pro"}
_ALL_PREFIX_TOKENS = _NEGATION_PREFIXES | _STANCE_PREFIXES


def _normalize_surface(text: str) -> str:
    return (
        text.lower()
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
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
    if token.text.lower() in ("born", "bearing"):
        return None
    return _match_inventory(token.lemma_, INANIMATE_NOUNS) or _match_inventory(token.text, INANIMATE_NOUNS)


def _group_base(token) -> str | None:
    for inventory in (TARGET_TOKENS, CONTRAST_TOKENS):
        match = _match_inventory(token.lemma_, inventory) or _match_inventory(token.text, inventory)
        if match:
            return match
    return None


def _group_side(base: str | None) -> str | None:
    if base is None:
        return None
    if base in POLITICAL_GROUP_TOKENS:
        return "political"
    if base in TARGET_TOKENS:
        return "minority"
    if base in CONTRAST_TOKENS:
        return "dominant"
    return None


def _get_group_prefix(token, doc) -> str | None:
    """Return the actual prefix token text if a group prefix is found, else None."""
    left = max(0, token.i - 2)
    for idx in range(left, token.i):
        candidate = _normalize_surface(doc[idx].text)
        if candidate in _ALL_PREFIX_TOKENS:
            return candidate
    return None


def _same_head_group_modifiers(token, doc) -> list:
    head = token.head
    siblings = []
    for sibling in head.children:
        if sibling.i == token.i or sibling.dep_ not in _MODIFIER_DEPS:
            continue
        if _group_base(sibling):
            siblings.append(sibling)
    return siblings




# ── Inanimate head nouns (active: mention-resolution and role guards) ────────
# Covers common false-positive head nouns for polysemous group adjectives
# (national _park_, black _hole_, etc.).

INANIMATE_NOUNS = {
    # ── Category 1: Nature, geography, weather, and locations ──
    "ocean", "sea", "river", "mountain", "desert", "forest", "lake", "valley", 
    "island", "beach", "coast", "shore", "bay", "gulf", "creek", "harbor", "harbour", 
    "continent", "subcontinent", "frontier", "world", "peninsula", "penisula", 
    "sky", "soil", "mud", "sand", "dirt", "rock", "stone", "clay", "dust", 
    "cloud", "moon", "star", "sun", "space", "weather", "climate", "wind", 
    "rain", "snow", "ice", "storm", "shrub", "tree", "flower", "leaf", "wood", 
    "wildlife", "animal", "bear", "cat", "sheep", "swan", "whale", "species", 
    "breed", "mould", "hole", "fire", "flame", "light", "oil", "smoke",
    "settlement", "side", "campus", "plant",

    # ── Category 2: Infrastructure, buildings, streets, and transport ──
    "building", "house", "home", "reservation", "bridge", "road", "street", 
    "highway", "wall", "door", "gate", "path", "alley", "avenue", "square", 
    "park", "station", "airport", "hospital", "airline", "airways", "store", 
    "shop", "restaurant", "hotel", "cabinet", "furniture", "tower", "embassy", 
    "consulate", "office", "headquarters", "facility", "city", "urban", "town", "capital", 
    "district", "sector", "zone", "province", "state", "nation", "country", 
    "empire", "republic", "kingdom", 
    "territory", "border", "region", "area", "public", "center",

    # ── Category 3: Objects, technology, software, and tools ──
    "car", "vehicle", "computer", "phone", "machine", "device", "equipment", 
    "tool", "instrument", "software", "algorithm", "system", "network", "platform", 
    "database", "data", "tech", "technology", "app", "application", "code", 
    "chip", "tablets", "earphone", "headphone", "printer", "ink", "box", "bottle", 
    "glass", "cup", "pot", "pan", "dish", "plate", "bowl", "fork", "spoon", 
    "knife", "weapon", "gun", "sword", "shield", "armor", "armour", "wire", 
    "rope", "string", "chain", "backpack", "bag", "satchel", "handbag", "case", 
    "chest", "trunk", "barrel", "basket", "pocket", "chalk", "coal", "hood",
    "card", "pack", "package", "paper", "sheet", "passport", "ship", "photograph", 
    "image", "figure", "dollar",

    # ── Category 4: Media, arts, publications, entertainment, and sports ──
    "film", "movie", "game", "sport", "song", "album", "novel", "book", "magazine", 
    "journal", "newspaper", "newsletter", "article", "report", "blog", "column", 
    "podcast", "interview", "survey", "editorial", "archive", "footage", "video", 
    "tv", "television", "series", "broadcast", "show", "media", "music", "art", 
    "painting", "drawing", "sculpture", "comedy", "drama", "tragedy", "opera", 
    "play", "performance", "website", "championship", "tournament", "cup", 
    "trophy", "football", "baseball", "basketball", "soccer", "tennis", "golf", 
    "chess", "band", "orchestra", "choir", "vandal", "cinema", "story", "literature",

    # ── Category 5: Business, economics, finance, and corporate entities ──
    "business", "enterprise", "venture", "market", "economy", "company", "firm", 
    "multinational", "corporation", "brand", "patent", "trade", "tariff", "industry", "product", 
    "stock", "bond", "price", "cost", "wage", "salary", "tax", "fund", "budget", 
    "debt", "account", "bank", "property", "estate", "finance", "commercial", 
    "advertisement", "job", "position", "role", "work", "resource", "service", 
    "supply", "demand",

    # ── Category 6: Politics, laws, documents, and governance ──
    "policy", "law", "bill", "regulation", "statute", "act", "charter", "treaty", "mandate",
    "constitution", "decree", "politics", "government", "regime", "power", 
    "propaganda", "movement", "protest", "revolution", "independence", "war", 
    "conflict", "violence", "abuse", "crime", "embargo", "sanction", "summit", 
    "conference", "meeting", "workshop", "forum", "confrontation", "party", 
    "check", "record", "document", "form", "signature", "program", "project", 
    "grant", "scheme", "issue", "problem", "question", "situation", "force", 
    "troop", "army", "authority", "rule", "colony", "u.s.", "usa", "vote",

    # ── Category 7: Science, mathematics, statistics, and medicine ──
    "medicine", "drug", "treatment", "therapy", "surgery", "disease", "measles", 
    "virus", "cell", "gene", "dna", "engineering", "study", "research", "science", 
    "mathematics", "statistics", "average", "level", "rate", "index", "ratio", 
    "formula", "analysis", "term", "bibliography", "%", "context", "education", 
    "health", "occupation",

    # ── Category 8: Organizations, institutions, societies, and foundations ──
    "association", "alliance", "union", "cooperation", "management", "committee", 
    "council", "board", "agency", "department", "institution", "organization", 
    "academy", "university", "college", "school", "library", "museum", "parliament", 
    "court", "society", "foundation", "initiative", "federation", "coalition", 
    "fellowship", "fellowshop", "commission", "caucus", 
    "language", "linguistics", "dialect", "accent", "grammer",

    # ── Category 9: Religion, abstract concepts, time, and miscellaneous ──
    "faith", "religion", "church", "mosque", "temple", "shrine", "tradition", 
    "culture", "celebration", "festival", "holiday", "wedding", "calligraphy", 
    "calligrapy", "jewelry", "garment", "attire", "clothing", "outfit", "gown", 
    "cloak", "cap", "hat", "coat", "shirt", "pants", "pant", "trouser", "suit", 
    "jacket", "skirt", "robe", "veil", "blanket", "pillow", "towel", "sack", 
    "wrapping", "wallpaper", "fabric", "cloth", "thread", "linen", "silk", 
    "cotton", "velvet", "leather", "plastic", "metal", "iron", "gold", "silver", 
    "copper", "bronze", "steel", "wool", "fur", "feather", "plumage", "hair", 
    "teeth", "tooth", "forehead", "spot", "nail", "tail", "collar", "sash", 
    "tie", "belt", "shoe", "boot", "sock", "glove", "buttercream", "cake", 
    "beer", "wine", "bread", "rice", "tea", "coffee", "cofƒee", "food", 
    "cuisine", "spice", "menu", "recipe", "meal", "dream", "magic", "illusion", 
    "one", "immigration", "period", "interest", "affair", "standard", "security", 
    "defense", "defence", "anthem", "flag", "list", "mail", "summer", "winter", 
    "spring", "autumn", "fall", "friday", "idol", "lie", "joke", "toast", 
    "horn", "accent", "speaker", "cross", "point", "carpentry", "decoration", 
    "cheese", "life", "lifestyle", "philosophy", "style", "topics", "privilege", 
    "history", "heritage", "identity", "race", "ancestry", "supremacy", "neighborhood", 
    "right", "experience", "skin", "citizenship", "nationality", "civilization", 
    "character", "class", "value", "name", "origin", "influence", "background", 
    "suffering", "way", "conquest", "root", "access", "ethnicity", "oppression", 
    "preference", "condition", "color", "racism", "attack", "body", "campaign", 
    "ideal", "presence", "invasion", "historical",
}

INANIMATE_ENTITY_TYPES = (
    "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART",
    "DATE", "TIME", "MONEY", "QUANTITY", "CARDINAL",
    "ORDINAL", "LAW", "LANGUAGE", "FAC",
)

# ── Human-referent head nouns (active: modifier/head disambiguation) ──────────

HUMAN_NOUNS = {
    # man family
    "chairman", "gentleman", "lady",
    "craftsman", "businessman", "entrepreneur", "oligarch",
    "person", "personnel", "people", "man", "woman", "men", "women",
    "child", "children", "kid", "kids", "boy", "boys", "girl", "girls", "guy", "guys",
    "mother", "father", "parents", "brother", "sister",
    "adult", "youth", "elder", "teenager", "baby",
    "community", "population", "group", "family", "household",
    "tribe", "clan", "folk",
    "slave",  "enslaved",
    "worker", "employee", "laborer", "labourer", "staff",
    "resident", "inhabitant", "citizen", "voter", "taxpayer",
    "student", "teacher", "professor", "scholar", "researcher",
    "doctor", "nurse", "lawyer", "engineer", "scientist",
    "leader", "official", "politician", "activist", "advocate",
    "soldier", "veteran", "prisoner", "inmate", "detainee",
    "fighter", "villager",
    "farmer", "merchant", "trader", "entrepreneur",
    "artist", "writer", "musician",
    "journalist", "reporter", "investigator",
    "tourist", "traveler", "traveller", "visitor",
    "owner", "employer", "manager", "director", "executive",
    "patient", "victim", "survivor", "witness",
    "suspect", "criminal", "offender", "convict",
    "consumer", "customer", "client", "shopper", "buyer",
    "member", "participant", "delegate", "representative",
    "neighbor", "neighbour", "colleague", "peer",
    "ancestor", "descendant",
    "immigrant", "refugee", "migrant", "settler", "colonist",
    "foreigner", "native", "exile",
    "seeker", "seekers",
    "minority", "majority",
    # service / occupation roles missing from earlier coverage
    "guard", "officer", "policeman", "police", "enforcer",
    "informant", "agent", "clerk", "secretary",
    "ambassador", "minister", "president",
    "spokesman", "spokeswoman", "representative",
    "chef", "cook", "waiter", "waitress",
    "maid", "servant", "butler",
    "driver", "pilot", "captain",
    "athlete", "player", "coach",
    "actor", "actress", "singer", "dancer", "rapper",
    "influencer", "blogger", "broadcaster",
    "barber", "stylist", "barista", "janitor",
    "guardian", "host", "guest",
    "fan", "follower", "supporter",
    "boss", "subordinate",
    "couple", "spouse", "lover", "partner",
    "friend", "enemy",
    "founder", "creator",
    "candidate", "nominee",
    "buyer", "seller",
    "ally", "adversary",
    "protester", "revolutionary",
    "astronaut",
    "bishop", "pope", "priest", "nun", "pastor", "preacher",
    # additional human-referent nouns (covers gaps from mark.md)
    "audience", "anthropologist",
    # Newly identified human referents
    "counterpart", "male", "female", "author", "explorer", "husband",
    "individual", "civilian", "teen", "bride", "cop", "cowboy", "human",
    "millennial", "genz", "boomer",
    # descriptive identity
    "racist", "nationalist", "conservativist", "democrat", "republican",
    "leftist", "socialist", "capitalist",
}


def _is_in_hyphenated_chain_with_minority(token, doc) -> bool:
    """Return True if a token is part of a hyphenated chain containing a minority group."""
    # Scan left
    i = token.i
    while i > 0:
        prev_token = doc[i - 1]
        if prev_token.text in ("-", "\u2011", "\u2012", "\u2013", "\u2014"):
            if i - 2 >= 0:
                left_token = doc[i - 2]
                left_base = _group_base(left_token)
                if left_base is not None and _group_side(left_base) == "minority":
                    return True
                i = i - 2
            else:
                break
        else:
            left_token = prev_token
            left_base = _group_base(left_token)
            if left_base is not None and _group_side(left_base) == "minority":
                return True
            break

    # Scan right
    i = token.i
    while i < len(doc) - 1:
        next_token = doc[i + 1]
        if next_token.text in ("-", "\u2011", "\u2012", "\u2013", "\u2014"):
            if i + 2 < len(doc):
                right_token = doc[i + 2]
                right_base = _group_base(right_token)
                if right_base is not None and _group_side(right_base) == "minority":
                    return True
                i = i + 2
            else:
                break
        else:
            right_token = next_token
            right_base = _group_base(right_token)
            if right_base is not None and _group_side(right_base) == "minority":
                return True
            break

    return False


def _resolve_group_token_raw(token, doc):
    group_base_val = _group_base(token)
    lemma = group_base_val or _normalize_surface(token.lemma_)
    head = token.head
    head_group = _group_base(head) if head is not None and head != token else None

    is_modifier = token.dep_ in _MODIFIER_DEPS
    inanimate_head = (
        (
            _inanimate_noun_form(head) is not None
            or head.ent_type_ in INANIMATE_ENTITY_TYPES
        )
        and _group_base(head) is None
    ) if head is not None else False

    # ── Spaced/prepositional compound resolution ─────────────────────────────
    token_lemma_norm = _normalize_surface(token.lemma_)
    compound_spec = COMPOUND_TARGET_HEADS.get(token_lemma_norm)
    if compound_spec is not None:
        # Compound modifier: head + child whose lemma matches compound_lemma.
        compound_lemma = compound_spec.get("compound_lemma")
        if compound_lemma is not None:
            for child in token.children:
                if (
                    child.dep_ in _MODIFIER_DEPS
                    and _normalize_surface(child.lemma_) == compound_lemma
                ):
                    return (compound_spec["side"], compound_spec["canonical"])
        # Prepositional modifier: head -> prep -> pobj.
        prep_lemma = compound_spec.get("prep_lemma")
        prep_pobj_lemma = compound_spec.get("prep_pobj_lemma")
        if prep_lemma is not None and prep_pobj_lemma is not None:
            for child in token.children:
                if child.dep_ == "prep" and _normalize_surface(child.lemma_) == prep_lemma:
                    for grandchild in child.children:
                        if (
                            grandchild.dep_ == "pobj"
                            and _normalize_surface(grandchild.lemma_) == prep_pobj_lemma
                        ):
                            return (compound_spec["side"], compound_spec["canonical"])

    # Suppress the modifier child of a recognized compound head so it does not
    # also fire on its own (e.g., the 'asylum' in 'asylum seeker').
    if (
        token.head is not None
        and token.dep_ in _MODIFIER_DEPS
        and _normalize_surface(token.head.lemma_) in COMPOUND_TARGET_HEADS
    ):
        head_spec = COMPOUND_TARGET_HEADS[_normalize_surface(token.head.lemma_)]
        if _normalize_surface(token.lemma_) == head_spec.get("compound_lemma"):
            return None
    # Same idea for the pobj branch (the 'color' in 'people of color').
    if (
        token.dep_ == "pobj"
        and token.head is not None
        and token.head.dep_ == "prep"
        and token.head.head is not None
        and _normalize_surface(token.head.head.lemma_) in COMPOUND_TARGET_HEADS
    ):
        head_spec = COMPOUND_TARGET_HEADS[_normalize_surface(token.head.head.lemma_)]
        if (
            _normalize_surface(token.head.lemma_) == head_spec.get("prep_lemma")
            and _normalize_surface(token.lemma_) == head_spec.get("prep_pobj_lemma")
        ):
            return None

    # ── Prefix suppression ───────────────────────────────────────────────────
    prefix = _get_group_prefix(token, doc)
    if prefix is not None:
        if prefix in _NEGATION_PREFIXES:
            # non-X ≠ X: suppress unconditionally
            return None
        # anti-/pro-: stance prefix, suppress only when modifier of another
        # demographic head (let the head carry the mention).
        if is_modifier and head_group:
            return None

    # ── X speaking Y: a language-modifier of a demographic head should not
    # double-count as a separate group mention.
    if (
        is_modifier
        and head is not None
        and _normalize_surface(head.lemma_) in {"speak", "speaking"}
        and head.head is not None
        and _group_base(head.head) is not None
    ):
        return None

    # ── Hyphenated chain compound demographic rule ───────────────────────────
    # If a dominant group token is part of a hyphenated chain (or directly adjacent)
    # containing at least one minority group token, we suppress the dominant one
    # to keep the minority one (e.g., "African American", "Palestinian-Lebanese-American"
    # -> keep the minority constituents).
    # Except for "native American" (handled separately).
    if group_base_val and _group_side(lemma) == "dominant":
        if _is_in_hyphenated_chain_with_minority(token, doc):
            return None

    # ── Sibling-modifier suppression: when two modifiers of the same
    # head both resolve to demographic groups, suppress the dominant-side
    # one and keep the minority-side.
    if group_base_val and is_modifier and head is not None:
        own_side = _group_side(lemma)
        if own_side == "dominant":
            for sibling in head.children:
                if sibling.i == token.i or sibling.dep_ not in _MODIFIER_DEPS:
                    continue
                sib_base = _group_base(sibling)
                if sib_base is None or sib_base == lemma:
                    continue
                if _get_group_prefix(sibling, doc) is not None:
                    continue
                if _group_side(sib_base) == "minority":
                    return None

    # ── Inanimate head suppression ───────────────────────────────────────────
    # If the token modifies a confirmed inanimate head, suppress.
    # "German law", "black hole", "American government" → None.
    if is_modifier and inanimate_head:
        return None

    # ── Keyword extraction: emit if group-resolvable ─────────────────────────
    side = _group_side(lemma)
    if side is not None and side != "political":
        return (side, lemma)

    return None


def resolve_group_token(token, doc, context_window: int = GROUP_CONTEXT_WINDOW):
    """
    Resolve a token to a demographic group label.

    Keyword-extraction approach: any token in the active set whose head is not
    a confirmed inanimate noun resolves to its group side. No human-head
    requirement, no spaCy-POS gating beyond inanimate suppression.

    Returns:
        (group_type, canonical_lemma) where group_type ∈ {"minority", "dominant"}
        or None when the token should not count as a group mention.

    When _ACTIVE_EXTRACTION_TOKENS is set (populated by run_pipeline from
    dolma/active_labels.json), tokens whose base lemma or compound canonical
    is not in that set are suppressed immediately, restricting Phase 2
    mention resolution to the same labels Phase 1 extracted.
    """
    res = _resolve_group_token_raw(token, doc)
    if res is not None:
        side, canonical = res
        if _ACTIVE_EXTRACTION_TOKENS is not None and canonical not in _ACTIVE_EXTRACTION_TOKENS:
            return None
        return (side, canonical)
    return None
