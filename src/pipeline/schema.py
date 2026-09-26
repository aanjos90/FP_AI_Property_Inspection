"""The fixed lists and word-lists the rest of the pipeline shares"""

# The parts of a room that the report describes
SECTIONS = ["door", "floor", "wall", "window", "ceiling", "electrical", "lighting", "other"]

# How each section is named in the report and in the questions. The keys are the same as SECTIONS.
SECTION_NAMES = {
    "door": "door",
    "floor": "floor",
    "wall": "walls",
    "window": "window",
    "ceiling": "ceiling",
    "electrical": "electrical fittings",
    "lighting": "light fixtures",
    "other": "other fixtures and fittings",
}

# Room folder names are one word
ROOM_DISPLAY_NAMES = {
    "livingroom": "living room",
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
}


def room_display_name(room: str) -> str:
    """Return the everyday name of a room"""
    return ROOM_DISPLAY_NAMES.get(room, room)


# What the report says about a condition that came from the photos
REPORT_LEVEL_WORDS = ["new or like-new", "good condition", "worn", "damaged"]


def condition_wording(condition: dict) -> str:
    """The words that stand for a condition in the report and in the questions"""
    if condition["source"] == "vision" and "level" in condition:
        return REPORT_LEVEL_WORDS[condition["level"]]
    return condition["value"] or ""


# If a sentence contains one of these words, it is filed under that section.
# One sentence can belong to several sections like "the walls and the ceiling"
SECTION_KEYWORDS = {
    "door": ["door", "lock"],
    "floor": ["floor", "baseboard"],
    "wall": ["wall"],
    "window": ["window", "grill", "latch"],
    "ceiling": ["ceiling"],
    "electrical": ["electrical", "switch", "outlet", "breaker", "faceplate", "wiring", "distribution board"],
    "lighting": ["lighting", "led", "bulb", "light fixture", "ceiling light"],
    "other": [
        "plumbing", "sanitary", "cabinet", "vanity", "mdf", "shelf", "mirror", "antenna",
        "panel", "granite", "countertop", "holder", "towel", "basin", "drain", "shower",
        "toilet", "faucet", "valve", "microwave", "cooktop", "enclosure", "fixture", "marble",
    ],
}

# door and wall are ambiguous words - "a cabinet door", "a cable along the wall"
AMBIGUOUS_SECTIONS = ["door", "wall"]

# CLIP recognises these kinds of things
# This says which report section each one belongs to
COMPONENT_TO_SECTION = {
    "door": "door",
    "floor": "floor",
    "wall": "wall",
    "window": "window",
    "ceiling": "ceiling",
    "electrical fitting": "electrical",
    "light fixture": "lighting",
    "baseboard": "floor",
    "fixture": "other",
    "plumbing fitting": "other",
    "sanitary ware": "other",
    "shower enclosure": "other",
    "cabinetry": "other",
    "railing": "other",
}


""" Condition labels for property components
These are used to generate prompts for CLIP model
Conditions from best to worst: new, good, worn, damaged
Inferred from actual property inspection reports with the help of Claude
"""
CONDITION_LABELS = {
    "door": [
        "new or like-new door with fresh smooth paint or varnish and an intact frame",
        "door in good condition with slightly faded varnish and minor scuffs",
        "worn door with faded varnish, scratches, stains, nail holes and paint splashes",
        "damaged door with broken frame, cracked panel, holes, peeling laminate or rust",
    ],
    "floor": [
        "new shiny ceramic tile or laminate flooring, clean and intact",
        "clean tiled floor in good condition with slight wear or dust",
        "worn floor with scratches, stains, dirty grout and paint splashes",
        "damaged flooring with chipped, cracked or broken tiles, cement patches or rotten boards",
    ],
    "wall": [
        "freshly painted smooth wall or clean new wall tiles with no holes",
        "painted or tiled wall in good condition with minor marks",
        "wall with faded paint, scuffs, stains, dirty grout or filled screw holes",
        "damaged wall with many drilled holes, cracks, peeling paint or broken tiles",
    ],
    "window": [
        "new window frame with clean, clear and intact glass",
        "window in good condition with minor frame marks or slightly dirty glass",
        "worn window with faded paint, paint splashes, dirty glass and dirty tracks",
        "damaged window with broken or cracked glass, rusty frame or missing handles",
    ],
    "ceiling": [
        "freshly painted smooth white ceiling or new clean PVC panel ceiling",
        "white ceiling in good condition with minor marks",
        "worn ceiling with stains, discolouration or paint marks along the edges",
        "damaged ceiling with holes, cracks, broken panels, peeling paint or water stains",
    ],
    "electrical fitting": [
        "new clean white wall switch and socket plates",
        "switch and socket plates in good condition with slight wear",
        "old yellowed switch or socket plates with dirt and paint stains",
        "damaged socket or switch with missing or cracked cover plate and exposed wires",
    ],
    "light fixture": [
        "new ceiling light fixture or new LED panel light",
        "clean white ceiling light fixture in good condition",
        "old ceiling light with yellowed cover, dirt or paint stains",
        "broken light fixture or bare bulb socket hanging from exposed wires",
    ],
    "baseboard": [
        "new or like-new baseboard with fresh paint, straight and intact",
        "baseboard in good condition with minor scuffs",
        "worn baseboard with scratches, dirt marks and faded or chipped paint",
        "damaged baseboard with cracks, missing pieces, loose sections or water damage",
    ],
    "fixture": [
        "new wall fittings or bathroom accessories, firmly fixed",
        "towel rail, curtain rod or accessories in good condition",
        "old wall accessories with stains, rust spots or loosely fixed",
        "broken or missing wall accessories with bare mounting holes",
    ],
    "plumbing fitting": [
        "new shiny chrome tap or valve",
        "tap or valve in good condition with minor wear",
        "worn tap or valve with oxidation, limescale or stains",
        "broken tap, rusted valve or open pipe outlet with no fitting",
    ],
    "sanitary ware": [
        "new clean white toilet, washbasin or laundry sink",
        "white toilet, washbasin or sink in good condition with minor marks",
        "stained toilet, washbasin or sink with dirty bowl, yellow marks or worn seat",
        "damaged toilet, washbasin or sink with chips, cracks or a broken or missing seat",
    ],
    "shower enclosure": [
        "new shower enclosure with clear glass and a clean frame",
        "shower enclosure in good condition with slightly dirty glass",
        "worn shower enclosure with limescale, dirty panels and a worn frame",
        "damaged shower enclosure with cracked or broken panels or peeling frame",
    ],
    "cabinetry": [
        "new MDF cabinets with intact doors and a clean countertop",
        "cabinets or wardrobe in good condition with minor wear",
        "worn cabinets with stains, scratches, dirty interior and misaligned doors",
        "damaged cabinets with swollen water-damaged MDF, chipped edges or broken drawers",
    ],
    "railing": [
        "new freshly painted metal railing or handrail",
        "metal railing in good condition with minor marks",
        "worn railing with faded paint and dirty stains",
        "damaged railing with rust, peeling paint or broken sections",
    ],
}
