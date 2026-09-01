"""
Supplementary Knowledge — additional entries from Wikipedia and educational sources.

Covers: food science, sports, transportation, space exploration, famous people,
everyday science, animals, plants, materials science, environmental science,
psychology, linguistics, and more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SuppEntry:
    id: str
    domain: str
    topic: str
    fact: str
    answer: str
    confidence: float
    source: str
    category: str
    reasoning: str = ""
    related: list[str] = field(default_factory=list)


class SupplementaryKnowledge:
    """Additional knowledge entries beyond the base 500."""

    def __init__(self) -> None:
        self._entries: list[SuppEntry] = []
        self._load_food_science()
        self._load_sports()
        self._load_transportation()
        self._load_space_exploration()
        self._load_famous_people()
        self._load_everyday_science()
        self._load_animals()
        self._load_plants()
        self._load_materials_science()
        self._load_psychology()
        self._load_linguistics()
        self._load_additional_physics()
        self._load_additional_biology()
        self._load_additional_history()
        self._load_additional_geography()
        self._load_additional_math()
        self._load_additional_cs()
        self._load_additional_health()

    def get_all(self) -> list[SuppEntry]:
        return self._entries

    # ══════════════════════════════════════════════════════════════
    # FOOD SCIENCE
    # ══════════════════════════════════════════════════════════════
    def _load_food_science(self) -> None:
        d = "food_science"
        s = "USDA / Food science sources"
        e = self._entries.extend
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"fs_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1
            return entry

        n("Protein", "Protein is made of amino acids; essential amino acids must come from food", "yes", "definition")
        n("Fiber", "Dietary fiber aids digestion; found in fruits, vegetables, and whole grains", "yes", "fact")
        n("Gluten", "Gluten is a protein in wheat, barley, and rye that gives bread its elasticity", "yes", "definition")
        n("Fermentation Foods", "Fermented foods include yogurt, kimchi, sauerkraut, kombucha, and miso", "yes", "classification")
        n("Maillard Reaction", "The Maillard reaction browns food when amino acids and sugars are heated", "yes", "process")
        n("Food Preservation", "Methods: canning, freezing, drying, pickling, pasteurization, irradiation", "yes", "classification")
        n("Food Groups", "Major food groups: grains, protein, dairy, fruits, vegetables", "yes", "classification")
        n("Water Boiling Point", "Water boils at 100°C (212°F) at sea level", "100", "fact")
        n("Water Freezing Point", "Water freezes at 0°C (32°F) at sea level", "0", "fact")
        n("Calorie Definition", "One food calorie (kilocalorie) = energy to raise 1 kg water by 1°C", "yes", "definition")
        n("Vitamin C", "Vitamin C prevents scurvy; found in citrus fruits, peppers, and strawberries", "yes", "fact")
        n("Omega-3 Fatty Acids", "Omega-3s are essential fats found in fish, walnuts, and flaxseed", "yes", "fact")

    # ══════════════════════════════════════════════════════════════
    # SPORTS
    # ══════════════════════════════════════════════════════════════
    def _load_sports(self) -> None:
        d = "sports"
        s = "Olympic Committee / Sports science"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"sp_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Olympic Games Origin", "The ancient Olympic Games began in 776 BCE in Olympia, Greece", "776 BCE", "event")
        n("Modern Olympics", "The modern Olympic Games were revived in 1896 by Pierre de Coubertin", "1896", "event")
        n("Olympic Rings", "The five Olympic rings represent the five inhabited continents", "yes", "symbol")
        n("Soccer Field", "A standard soccer field is 100-110m long and 64-75m wide", "yes", "fact")
        n("Basketball Invented", "Basketball was invented by James Naismith in 1891 in Springfield, Massachusetts", "1891", "event")
        n("Marathon Distance", "A marathon is 42.195 kilometers (26.2 miles)", "42.195", "fact")
        n("Swimming Strokes", "Competitive swimming strokes: freestyle, backstroke, breaststroke, butterfly", "yes", "classification")
        n("World Cup", "The FIFA World Cup is held every 4 years; Brazil has won the most (5 titles)", "5", "fact")
        n("Tennis Scoring", "Tennis scoring: 15, 30, 40, game; a set requires 6 games with 2-game lead", "yes", "rules")
        n("Cricket Terms", "Cricket: a century is 100 runs, a wicket is when a batsman is dismissed", "yes", "definition")
        n("Athletics Events", "Track and field: sprinting, middle-distance, long-distance, jumping, throwing", "yes", "classification")
        n("Golf Basics", "Golf: 18 holes, par is the expected strokes, birdie is 1 under par", "18", "fact")

    # ══════════════════════════════════════════════════════════════
    # TRANSPORTATION
    # ══════════════════════════════════════════════════════════════
    def _load_transportation(self) -> None:
        d = "transportation"
        s = "Various sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"tr_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Wright Brothers", "The Wright Brothers made the first powered flight on December 17, 1903", "1903", "event")
        n("Speed of Sound", "The speed of sound in air is about 343 m/s (1,235 km/h)", "343", "fact")
        n("Speed of Sound Mach", "Mach 1 = speed of sound; Mach 2 = twice the speed of sound", "yes", "definition")
        n("Electric Cars", "Electric cars use batteries and motors; Tesla, Nissan Leaf are popular models", "yes", "fact")
        n("Railways", "The first public railway opened in 1825 in Stockton and Darlington, England", "1825", "event")
        n("GPS System", "GPS uses 24+ satellites to provide location accuracy within a few meters", "24+", "fact")
        n("Aviation Altitude", "Commercial airplanes cruise at 30,000-40,000 feet (9,000-12,000 m)", "30000-40000", "fact")
        n("Bicycle Efficiency", "Bicycles are the most energy-efficient form of transportation", "yes", "fact")
        n("Container Shipping", "Container shipping revolutionized global trade in the 1950s", "yes", "fact")
        n("Hyperloop", "Hyperloop proposes pods traveling through tubes at over 1,000 km/h", "1000+", "concept")

    # ══════════════════════════════════════════════════════════════
    # SPACE EXPLORATION
    # ══════════════════════════════════════════════════════════════
    def _load_space_exploration(self) -> None:
        d = "space_exploration"
        s = "NASA / ESA / Roscosmos"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"se_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("First Satellite", "Sputnik 1 was the first artificial satellite, launched by the USSR in 1957", "1957", "event")
        n("First Human in Space", "Yuri Gagarin was the first human in space on April 12, 1961", "1961", "event")
        n("Moon Landing", "Apollo 11 landed on the Moon on July 20, 1969; Neil Armstrong was first to walk", "1969", "event")
        n("International Space Station", "The ISS has been continuously occupied since November 2000", "2000", "fact")
        n("Mars Rovers", "Mars rovers: Sojourner, Spirit, Opportunity, Curiosity, Perseverance", "yes", "classification")
        n("James Webb Telescope", "JWST launched in 2021, orbits at L2, observes infrared light from the early universe", "2021", "fact")
        n("Voyager 1", "Voyager 1 launched in 1977, entered interstellar space in 2012, still transmitting", "yes", "fact")
        n("Space Shuttle", "The Space Shuttle flew from 1981 to 2011, completing 135 missions", "135", "fact")
        n("Hubble Telescope", "The Hubble Space Telescope has been orbiting Earth since 1990", "1990", "fact")
        n("Rocket Propulsion", "Rockets work by Newton's Third Law: exhaust pushes down, rocket pushes up", "yes", "principle")
        n("Escape Velocity", "Earth's escape velocity is about 11.2 km/s", "11.2", "fact")
        n("International Space Treaty", "The Outer Space Treaty (1967) prohibits weapons of mass destruction in space", "1967", "fact")
        n("SpaceX", "SpaceX, founded by Elon Musk in 2002, developed reusable rockets (Falcon 9)", "2002", "fact")

    # ══════════════════════════════════════════════════════════════
    # FAMOUS PEOPLE
    # ══════════════════════════════════════════════════════════════
    def _load_famous_people(self) -> None:
        d = "famous_people"
        s = "Various biographical sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="entity", conf=0.95):
            entry = SuppEntry(f"fp_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Isaac Newton", "Newton (1643-1727) developed laws of motion, calculus, and gravity", "yes")
        n("Marie Curie", "Marie Curie (1867-1934) won Nobel Prizes in Physics and Chemistry for radioactivity research", "yes")
        n("Charles Darwin", "Darwin (1809-1882) wrote On the Origin of Species, establishing evolution by natural selection", "yes")
        n("Nikola Tesla", "Tesla (1856-1943) pioneered alternating current (AC) electrical systems", "yes")
        n("Thomas Edison", "Edison (1847-1931) invented the phonograph and practical incandescent light bulb", "yes")
        n("Ada Lovelace", "Ada Lovelace (1815-1852) wrote the first computer algorithm for Babbage's Analytical Engine", "yes")
        n("Alan Turing", "Turing (1912-1954) formalized computation with the Turing machine and helped crack Enigma", "yes")
        n("Albert Schweitzer", "Schweitzer (1875-1965) was a theologian, musician, and physician who won the Nobel Peace Prize", "yes")
        n("Rosalind Franklin", "Franklin (1920-1958) captured X-ray images crucial to discovering DNA's structure", "yes")
        n("Nelson Mandela", "Mandela (1918-2013) was South Africa's first Black president and anti-apartheid leader", "yes")
        n("Mahatma Gandhi", "Gandhi (1869-1948) led India's independence movement through nonviolent civil disobedience", "yes")
        n("Martin Luther King Jr.", "MLK (1929-1968) led the American civil rights movement; gave I Have a Dream speech", "yes")
        n("Winston Churchill", "Churchill (1874-1965) led Britain through WWII; won Nobel Prize in Literature", "yes")
        n("Abraham Lincoln", "Lincoln (1809-1865) was the 16th US president who preserved the Union and abolished slavery", "yes")
        n("Cleopatra", "Cleopatra (69-30 BCE) was the last active pharaoh of ancient Egypt", "yes")

    # ══════════════════════════════════════════════════════════════
    # EVERYDAY SCIENCE
    # ══════════════════════════════════════════════════════════════
    def _load_everyday_science(self) -> None:
        d = "everyday_science"
        s = "Various educational sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"es_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Rainbow Colors", "Rainbow colors (ROYGBIV): red, orange, yellow, green, blue, indigo, violet", "yes", "fact")
        n("Why Sky Blue", "The sky appears blue because shorter blue wavelengths scatter more than other colors", "yes", "cause")
        n("Why Sun Yellow", "The Sun appears yellow because Earth's atmosphere scatters blue light", "yes", "cause")
        n("Magnetism", "Magnets have north and south poles; they attract铁 and nickel", "yes", "fact")
        n("Static Electricity", "Static electricity builds up when materials gain or lose electrons through friction", "yes", "process")
        n("Rain Formation", "Rain forms when water vapor condenses around particles in clouds and becomes heavy enough to fall", "yes", "process")
        n("Snowflake Shape", "Snowflakes have hexagonal (6-sided) symmetry due to the molecular structure of ice", "6", "fact")
        n("Thunder and Lightning", "Lightning heats air to 30,000°C causing a shockwave we hear as thunder", "30000", "fact")
        n("Echo", "An echo is sound reflected off a surface, arriving at least 0.1 seconds after the original sound", "yes", "definition")
        n("Mirror Reflection", "Plane mirrors produce virtual images that are the same size and laterally inverted", "yes", "fact")
        n("Pendulum Clock", "A pendulum clock uses the regular swing of a pendulum to keep time", "yes", "process")
        n("Refrigerator", "A refrigerator removes heat from inside using a refrigerant cycle (compression/expansion)", "yes", "process")
        n("Microwave Oven", "Microwaves heat food by causing water molecules to vibrate rapidly", "yes", "process")
        n("Camera Lens", "A camera lens focuses light onto a sensor using refraction through curved glass elements", "yes", "process")
        n("LED Lights", "LEDs produce light through electroluminescence in semiconductor materials", "yes", "process")

    # ══════════════════════════════════════════════════════════════
    # ANIMALS
    # ══════════════════════════════════════════════════════════════
    def _load_animals(self) -> None:
        d = "animals"
        s = "Various zoological sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"an_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Animal Kingdom", "The animal kingdom has over 1.5 million known species", "1.5 million", "fact")
        n("Mammals", "Mammals are warm-blooded vertebrates that nurse young with milk", "yes", "definition")
        n("Birds", "Birds are warm-blooded vertebrates with feathers and most can fly", "yes", "definition")
        n("Reptiles", "Reptiles are cold-blooded vertebrates with scales (snakes, lizards, turtles, crocodiles)", "yes", "definition")
        n("Amphibians", "Amphibians live in water and on land; they undergo metamorphosis (frogs, salamanders)", "yes", "definition")
        n("Insects", "Insects have 6 legs, 3 body segments, and most have wings", "6", "fact")
        n("Fish", "Fish are cold-blooded vertebrates that live in water and breathe through gills", "yes", "definition")
        n("Whale Heart", "A blue whale's heart is about the size of a small car", "yes", "fact")
        n("Cheetah Speed", "Cheetahs can run up to 112 km/h (70 mph), the fastest land animal", "112", "fact")
        n("Octopus Intelligence", "Octopuses have 8 arms, 3 hearts, blue blood, and can solve complex puzzles", "yes", "fact")
        n("Dolphin Communication", "Dolphins use clicks, whistles, and body language to communicate", "yes", "fact")
        n("Ant Colony", "Ant colonies can contain millions of individuals working as a superorganism", "yes", "fact")
        n("Camel Adaptation", "Camels can survive weeks without water and store fat in their humps", "yes", "fact")
        n("Bat Echolocation", "Bats use echolocation (sonar) to navigate and find prey in the dark", "yes", "fact")
        n("Penguin Species", "There are 18 species of penguins, all found in the Southern Hemisphere", "18", "fact")
        n("Shark Cartilage", "Sharks have skeletons made of cartilage, not bone", "yes", "fact")
        n("Spider Silk", "Spider silk is stronger than steel by weight and highly elastic", "yes", "fact")
        n("Butterfly Metamorphosis", "Butterflies undergo complete metamorphosis: egg, larva, pupa, adult", "yes", "process")

    # ══════════════════════════════════════════════════════════════
    # PLANTS
    # ══════════════════════════════════════════════════════════════
    def _load_plants(self) -> None:
        d = "plants"
        s = "Botanical sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"pl_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Plant Classification", "Plants: mosses, ferns, conifers, flowering plants (angiosperms)", "yes", "classification")
        n("Flowering Plants", "Flowering plants (angiosperms) are the most diverse group with ~300,000 species", "300000", "fact")
        n("Tree Rings", "Tree rings indicate age; wider rings mean better growing conditions", "yes", "fact")
        n("Cactus Water", "Cacti store water in their stems and have spines instead of leaves to reduce water loss", "yes", "fact")
        n("Seeds", "Seeds contain an embryo, food supply, and protective coat", "yes", "definition")
        n("Pollination", "Pollination transfers pollen; done by wind, insects, birds, bats", "yes", "process")
        n("Nitrogen Fixation", "Some plants (legumes) host bacteria that convert atmospheric nitrogen to usable forms", "yes", "process")
        n("Bamboo Growth", "Bamboo can grow up to 91 cm (35 inches) per day, the fastest growing plant", "91", "fact")
        n("Largest Flower", "Rafflesia arnoldii is the largest flower, up to 1 meter in diameter", "1 meter", "fact")
        n("Oldest Tree", "The oldest known tree is a bristlecone pine over 5,000 years old", "5000+", "fact")
        n("Moss", "Mosses are non-vascular plants that absorb water directly through their leaves", "yes", "fact")
        n("Fern Reproduction", "Ferns reproduce via spores found on the undersides of their fronds", "yes", "process")

    # ══════════════════════════════════════════════════════════════
    # MATERIALS SCIENCE
    # ══════════════════════════════════════════════════════════════
    def _load_materials_science(self) -> None:
        d = "materials_science"
        s = "Materials science sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"ms_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Steel", "Steel is an alloy of iron and carbon (0.2-2% carbon)", "yes", "definition")
        n("Aluminum", "Aluminum is the most abundant metal in Earth's crust", "yes", "fact")
        n("Concrete", "Concrete is a mixture of cement, water, and aggregate; strongest in compression", "yes", "fact")
        n("Glass", "Glass is made by heating sand (silicon dioxide) to very high temperatures", "yes", "process")
        n("Plastics", "Plastics are polymers derived from petroleum; types include PVC, PET, HDPE", "yes", "classification")
        n("Carbon Fiber", "Carbon fiber is 5 times stronger than steel and much lighter", "5x", "fact")
        n("Ceramics", "Ceramics are hard, heat-resistant materials made from clay or other minerals", "yes", "definition")
        n("Rubber", "Natural rubber comes from latex sap of rubber trees; synthetic rubber is petroleum-based", "yes", "fact")
        n("Graphene", "Graphene is a single layer of carbon atoms, stronger than steel and highly conductive", "yes", "fact")
        n("Superconductors", "Superconductors conduct electricity with zero resistance at very low temperatures", "yes", "definition")
        n("Semiconductors", "Semiconductors (silicon, germanium) are the basis of modern electronics", "yes", "definition")
        n("Memory Metal", "Nitinol (nickel-titanium) returns to its original shape when heated", "yes", "fact")

    # ══════════════════════════════════════════════════════════════
    # PSYCHOLOGY
    # ══════════════════════════════════════════════════════════════
    def _load_psychology(self) -> None:
        d = "psychology"
        s = "APA / Psychology textbooks"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="definition", conf=0.95):
            entry = SuppEntry(f"py_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Classical Conditioning", "Pavlov's dogs: pairing a bell with food caused dogs to salivate at the bell alone", "yes", "concept")
        n("Operant Conditioning", "Skinner's operant conditioning: behavior is shaped by rewards and punishments", "yes", "concept")
        n("Memory Models", "Memory models: sensory memory, short-term memory, long-term memory", "yes", "classification")
        n("Cognitive Biases", "Common biases: confirmation bias, anchoring, availability heuristic, framing effect", "yes", "classification")
        n("Fight or Flight", "The fight-or-flight response releases adrenaline and cortisol for immediate energy", "yes", "process")
        n("Sleep Stages", "Sleep stages: NREM (light, deep) and REM (dreaming, memory consolidation)", "yes", "classification")
        n("IQ Definition", "IQ measures cognitive ability relative to the population average (mean = 100, SD = 15)", "100", "definition")
        n("Maslow's Hierarchy", "Maslow's hierarchy: physiological, safety, love, esteem, self-actualization", "yes", "model")
        n("Attachment Theory", "Attachment styles: secure, anxious-ambivalent, avoidant, disorganized", "yes", "classification")
        n("Stress Response", "Hans Selye's General Adaptation Syndrome: alarm, resistance, exhaustion", "yes", "model")
        n("Dopamine", "Dopamine is a neurotransmitter involved in pleasure, motivation, and reward", "yes", "definition")
        n("Neuroplasticity", "The brain can reorganize itself by forming new neural connections throughout life", "yes", "concept")
        n("Placebo Effect", "The placebo effect: inactive treatments can produce real improvements in health", "yes", "phenomenon")
        n("Amygdala Function", "The amygdala processes emotions, especially fear and threat detection", "yes", "function")
        n("Prefrontal Cortex", "The prefrontal cortex handles decision-making, planning, and impulse control", "yes", "function")

    # ══════════════════════════════════════════════════════════════
    # LINGUISTICS
    # ══════════════════════════════════════════════════════════════
    def _load_linguistics(self) -> None:
        d = "linguistics"
        s = "Linguistics sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="definition", conf=0.95):
            entry = SuppEntry(f"lg_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Language Families", "Major language families: Indo-European, Sino-Tibetan, Afro-Asiatic, Niger-Congo", "yes", "classification")
        n("Most Spoken Language", "Mandarin Chinese is the most spoken native language (~920 million)", "Mandarin", "fact")
        n("English Origin", "English is a Germanic language with heavy French and Latin influence", "yes", "fact")
        n("Phonemes", "Phonemes are the smallest units of sound in a language", "yes", "definition")
        n("Morphology", "Morphology studies the internal structure of words (prefixes, roots, suffixes)", "yes", "definition")
        n("Syntax", "Syntax is the study of sentence structure and word order", "yes", "definition")
        n("Semantics", "Semantics studies meaning in language", "yes", "definition")
        n("Pragmatics", "Pragmatics studies how context affects meaning (implicature, speech acts)", "yes", "definition")
        n("Writing Systems", "Writing systems: alphabetic, syllabic, logographic, featural", "yes", "classification")
        n("Universal Grammar", "Chomsky proposed that all languages share a universal grammar innate to humans", "yes", "theory")
        n("Language Acquisition", "Children typically acquire language in stages: babbling, one-word, two-word, complex sentences", "yes", "process")
        n("Dead Languages", "Dead languages with no native speakers include Latin, Ancient Greek, Sanskrit", "yes", "classification")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL PHYSICS
    # ══════════════════════════════════════════════════════════════
    def _load_additional_physics(self) -> None:
        d = "physics"
        s = "Physics textbooks"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"phx_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Speed of Gravity", "Gravitational waves travel at the speed of light (299,792,458 m/s)", "299792458", "fact")
        n("Electron Mass", "Electron mass is approximately 9.109 x 10^-31 kg", "9.109e-31", "constant")
        n("Proton Mass", "Proton mass is approximately 1.673 x 10^-27 kg", "1.673e-27", "constant")
        n("Avogadro's Number", "Avogadro's number: 6.022 x 10^23 particles per mole", "6.022e23", "constant")
        n("Gravitational Constant", "G = 6.674 x 10^-11 N⋅m²/kg²", "6.674e-11", "constant")
        n("Speed of Light Squared", "c² = 8.988 x 10^16 m²/s² (used in E=mc²)", "8.988e16", "constant")
        n("Planck Length", "Planck length is about 1.616 x 10^-35 m, the smallest meaningful length", "1.616e-35", "constant")
        n("Cosmic Speed Limit", "Nothing with mass can exceed the speed of light in vacuum", "yes", "law")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL BIOLOGY
    # ══════════════════════════════════════════════════════════════
    def _load_additional_biology(self) -> None:
        d = "biology"
        s = "Biology textbooks"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"bix_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Human Genome", "The human genome has about 3 billion base pairs and ~20,000-25,000 genes", "3 billion", "fact")
        n("Cell Division", "Mitosis produces 2 identical daughter cells; meiosis produces 4 unique gametes", "yes", "process")
        n("Stem Cells", "Stem cells can differentiate into many different cell types", "yes", "definition")
        n("Cancer Genes", "Oncogenes promote cell growth; tumor suppressor genes inhibit it", "yes", "definition")
        n("Blood Types", "Blood types: A, B, AB, O; Rh positive or negative (8 types total)", "8", "fact")
        n("White Blood Cell Types", "WBC types: neutrophils, lymphocytes, monocytes, eosinophils, basophils", "yes", "classification")
        n("Neuron Structure", "Neurons have a cell body, dendrites (receive), and axon (sends signals)", "yes", "structure")
        n("Synapse", "Synapses are gaps between neurons where neurotransmitters传递 signals", "yes", "definition")
        n("Hormones", "Major hormones: insulin, adrenaline, cortisol, estrogen, testosterone, thyroid hormones", "yes", "classification")
        n("Blood Vessels", "Blood vessels: arteries (away from heart), veins (to heart), capillaries (exchange)", "yes", "classification")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL HISTORY
    # ══════════════════════════════════════════════════════════════
    def _load_additional_history(self) -> None:
        d = "history"
        s = "History textbooks"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="event", conf=0.95):
            entry = SuppEntry(f"hsx_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Printing Press", "Gutenberg invented the printing press around 1440, revolutionizing information spread", "1440")
        n("Magnus Carta", "The Magna Carta (1215) limited the power of the English king and established rule of law", "1215")
        n("Scientific Revolution", "The Scientific Revolution (16th-17th century) transformed natural philosophy", "16th-17th century")
        n("Enlightenment", "The Enlightenment (17th-18th century) emphasized reason, individualism, and science", "17th-18th century")
        n("Abolition of Slavery", "Slavery was abolished at different times: UK 1833, US 1865, Brazil 1888", "yes", "event")
        n("Women's Suffrage", "Women gained voting rights: New Zealand 1893, UK 1918, US 1920, France 1944", "yes", "event")
        n("Decolonization", "Decolonization occurred after WWII: India 1947, Africa 1950s-1960s", "yes", "event")
        n("Silk Road", "The Silk Road connected China to the Mediterranean, facilitating trade for centuries", "yes", "fact")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL GEOGRAPHY
    # ══════════════════════════════════════════════════════════════
    def _load_additional_geography(self) -> None:
        d = "geography"
        s = "World Atlas / Geographic sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"gex_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Russia Capital", "The capital of Russia is Moscow", "Moscow")
        n("South Korea Capital", "The capital of South Korea is Seoul", "Seoul")
        n("Italy Capital", "The capital of Italy is Rome", "Rome")
        n("Spain Capital", "The capital of Spain is Madrid", "Madrid")
        n("Mexico Capital", "The capital of Mexico is Mexico City", "Mexico City")
        n("Argentina Capital", "The capital of Argentina is Buenos Aires", "Buenos Aires")
        n("Nigeria Capital", "The capital of Nigeria is Abuja", "Abuja")
        n("Thailand Capital", "The capital of Thailand is Bangkok", "Bangkok")
        n("Turkey Capital", "The capital of Turkey is Ankara", "Ankara")
        n("Indonesia Capital", "The capital of Indonesia is Jakarta", "Jakarta")
        n("World Population", "World population is approximately 8 billion people (2023)", "8 billion")
        n("Largest Island", "Greenland is the largest island at 2.17 million km²", "2.17 million km²")
        n("Most Populated City", "Tokyo is the most populated metropolitan area with ~37 million", "37 million")
        n("Lakes", "The 5 Great Lakes are: Superior, Michigan, Huron, Erie, Ontario", "yes", "classification")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL MATH
    # ══════════════════════════════════════════════════════════════
    def _load_additional_math(self) -> None:
        d = "mathematics"
        s = "Math textbooks"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="formula", conf=0.95):
            entry = SuppEntry(f"mtx_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Pythagorean Triples", "Pythagorean triples: (3,4,5), (5,12,13), (8,15,17), (7,24,25)", "yes", "fact")
        n("Area of Triangle", "Area of triangle = (base × height) / 2", "A = bh/2", "formula")
        n("Area of Rectangle", "Area of rectangle = length × width", "A = lw", "formula")
        n("Volume of Cylinder", "Volume of cylinder = πr²h", "V = πr²h", "formula")
        n("Volume of Cone", "Volume of cone = (1/3)πr²h", "V = (1/3)πr²h", "formula")
        n("Volume of Cube", "Volume of cube = s³", "V = s³", "formula")
        n("Surface Area Sphere", "Surface area of sphere = 4πr²", "SA = 4πr²", "formula")
        n("Law of Cosines", "c² = a² + b² - 2ab⋅cos(C)", "c² = a² + b² - 2ab⋅cos(C)", "formula")
        n("Sine Rule", "a/sin(A) = b/sin(B) = c/sin(C)", "yes", "formula")
        n("Geometric Series", "Sum of geometric series: S = a/(1-r) for |r| < 1", "S = a/(1-r)", "formula")
        n("Combinations", "C(n,k) = n! / (k!(n-k)!)", "C(n,k) = n! / (k!(n-k)!)", "formula")
        n("Permutations", "P(n,k) = n! / (n-k)!", "P(n,k) = n! / (n-k)!", "formula")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL CS
    # ══════════════════════════════════════════════════════════════
    def _load_additional_cs(self) -> None:
        d = "computer_science"
        s = "CS textbooks"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="definition", conf=0.95):
            entry = SuppEntry(f"csx_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Bit and Byte", "A bit is 0 or 1; a byte is 8 bits; 1 KB = 1024 bytes", "8", "fact")
        n("IPv4 Addresses", "IPv4 uses 32-bit addresses (about 4.3 billion unique addresses)", "4.3 billion", "fact")
        n("IPv6 Addresses", "IPv6 uses 128-bit addresses, providing vastly more unique addresses", "128", "fact")
        n("HTML Tags", "HTML tags define structure: <html>, <head>, <body>, <div>, <p>, <a>, <img>", "yes", "classification")
        n("CSS Selectors", "CSS selectors: element, class (.name), ID (#name), attribute, pseudo-class", "yes", "classification")
        n("Git Commands", "Git basics: init, add, commit, push, pull, branch, merge, checkout", "yes", "classification")
        n("SQL Joins", "SQL joins: INNER, LEFT, RIGHT, FULL, CROSS", "yes", "classification")
        n("REST Principles", "REST: stateless, cacheable, uniform interface, layered system, client-server", "yes", "principles")
        n("JSON Format", "JSON: JavaScript Object Notation, lightweight data format with key-value pairs", "yes", "definition")
        n("Lambda Functions", "Lambda functions are anonymous functions defined inline: lambda x: x+1", "yes", "definition")
        n("Regular Expressions", "Regex patterns match text: . * + ? [] () ^ $ \\d \\w", "yes", "notation")
        n("Binary Search Tree", "BST: left child < parent < right child; enables O(log n) search", "yes", "structure")

    # ══════════════════════════════════════════════════════════════
    # ADDITIONAL HEALTH
    # ══════════════════════════════════════════════════════════════
    def _load_additional_health(self) -> None:
        d = "health"
        s = "Health sources"
        self._i = len(self._entries)

        def n(topic, fact, answer, cat="fact", conf=0.95):
            entry = SuppEntry(f"hlx_{self._i}", d, topic, fact, answer, conf, s, cat)
            self._entries.append(entry)
            self._i += 1

        n("Body Temperature", "Normal body temperature is about 37°C (98.6°F)", "37", "fact")
        n("Heart Rate", "Resting heart rate for adults is 60-100 beats per minute", "60-100", "fact")
        n("Respiratory Rate", "Normal breathing rate is 12-20 breaths per minute", "12-20", "fact")
        n("Blood Pressure", "Normal blood pressure is about 120/80 mmHg", "120/80", "fact")
        n("Human Height", "Average human height varies by country; global average is about 170 cm", "170", "fact")
        n("Brain Cells", "The human brain has approximately 86 billion neurons", "86 billion", "fact")
        n("Bone Count", "Adults have 206 bones; babies have about 270 that fuse during growth", "206", "fact")
        n("Muscle Count", "The human body has over 600 muscles", "600+", "fact")
        n("Blood Volume", "The average adult has about 5 liters of blood", "5 liters", "fact")
        n("Digestion Time", "Food takes 24-72 hours to pass through the digestive system", "24-72", "fact")
        n("Sleep Cycles", "A complete sleep cycle lasts about 90 minutes", "90", "fact")
        n("DNA Length", "If uncoiled, DNA in one cell would stretch about 2 meters", "2 meters", "fact")
