"""
FactualCore — fast factual knowledge lookup.

Responsibilities:
  - Match queries against a pre-compiled fact database.
  - Extract numbers and entities from evidence when no direct match.
  - Use a keyword index for O(1) candidate filtering before regex.

Knowledge domains:
  Physics, geography, math, biology, technology, inventions,
  astronomy, chemistry, human body, animals, food, space, language, countries.
"""
from __future__ import annotations

import re
from typing import Any

from ..core_protocol import CoreResult, make_result, empty_result


# ── Fact database ────────────────────────────────────────────
# Each entry: (regex_pattern, answer, confidence)
_RAW_FACTS: list[tuple[str, str, float]] = [
    # ═══ PHYSICS ═══
    (r"speed.*light", "299,792,458 m/s", 0.99),
    (r"speed.*sound", "343 m/s in air", 0.95),
    (r"boil.*water", "100°C at sea level", 0.99),
    (r"freeze.*water", "0°C at sea level", 0.99),
    (r"gravity.*earth", "9.8 m/s²", 0.99),
    (r"gravity", "force that attracts objects with mass", 0.99),
    (r"einstein", "physicist, theory of relativity, E=mc²", 0.99),
    (r"newton", "physicist, laws of motion, gravity", 0.99),
    (r"planck.*constant", "6.626 × 10⁻³⁴ J·s", 0.99),
    (r"boltzmann.*constant", "1.381 × 10⁻²³ J/K", 0.99),
    (r"avogadro.*number", "6.022 × 10²³ mol⁻¹", 0.99),
    (r"coulomb.*constant", "8.988 × 10⁹ N·m²/C²", 0.99),
    (r"atomic.*mass.*unit", "1.661 × 10⁻²⁷ kg", 0.99),
    (r"speed.*electromagnetic", "299,792,458 m/s (same as light)", 0.99),
    (r"absolute.*zero", "-273.15°C (0 Kelvin)", 0.99),
    (r"boiling.*point.*altitude", "decreases ~1°C per 300m altitude", 0.95),
    (r"pressure.*atmospheric", "101,325 Pa (1 atm) at sea level", 0.99),
    (r"friction.*coefficient.*rubber.*concrete", "0.6-0.85", 0.90),
    (r"kinetic.*energy.*formula", "KE = ½mv²", 0.99),
    (r"potential.*energy.*formula", "PE = mgh", 0.99),
    (r"ohm.*law", "V = IR (voltage = current × resistance)", 0.99),
    (r"newton.*second.*law", "F = ma (force = mass × acceleration)", 0.99),
    (r"newton.*third.*law", "Every action has an equal and opposite reaction", 0.99),
    (r"wavelength.*visible.*light", "380-700 nm", 0.95),
    (r"electromagnetic.*spectrum", "radio, microwave, infrared, visible, UV, X-ray, gamma", 0.99),
    # ═══ DNA & BIOLOGY ═══
    (r"dna", "deoxyribonucleic acid, genetic information", 0.99),
    (r"rna", "ribonucleic acid, protein synthesis", 0.99),
    (r"photosynthesis", "plants convert light to energy", 0.99),
    (r"mitosis", "cell division producing two identical cells", 0.99),
    (r"meiosis", "cell division producing four unique gametes", 0.99),
    (r"atp", "adenosine triphosphate, cellular energy currency", 0.99),
    (r"protein.*synthesis", "DNA → mRNA → ribosome → protein", 0.95),
    (r"cell.*membrane", "phospholipid bilayer controlling what enters/exits", 0.95),
    (r"mitochondria", "powerhouse of the cell, ATP production", 0.99),
    (r"chloroplast", "organelle where photosynthesis occurs", 0.99),
    (r"natural.*selection", "survival of the fittest, Darwin's mechanism of evolution", 0.99),
    (r"genetic.*code", "64 codons encode 20 amino acids", 0.95),
    (r"blood.*type", "A, B, AB, O (positive and negative)", 0.99),
    (r"immune.*system", "defends body against pathogens", 0.99),
    (r"white.*blood.*cell|leukocyte", "fights infection in the body", 0.99),
    (r"red.*blood.*cell|erythrocyte", "carries oxygen via hemoglobin", 0.99),
    (r"platelet|thrombocyte", "blood clotting cell", 0.99),
    (r"antibody|immunoglobulin", "Y-shaped protein that binds antigens", 0.95),
    (r"vaccine.*how.*work", "trains immune system to recognize pathogens", 0.95),
    (r"virus.*alive", "no — viruses cannot reproduce without a host cell", 0.95),
    # ═══ CAPITAL CITIES ═══
    (r"capital.*france", "Paris", 0.99),
    (r"capital.*japan", "Tokyo", 0.99),
    (r"capital.*germany", "Berlin", 0.99),
    (r"capital.*uk|capital.*united.*kingdom", "London", 0.99),
    (r"capital.*china", "Beijing", 0.99),
    (r"capital.*india", "New Delhi", 0.99),
    (r"capital.*brazil", "Brasilia", 0.99),
    (r"capital.*australia", "Canberra", 0.95),
    (r"capital.*canada", "Ottawa", 0.95),
    (r"capital.*egypt", "Cairo", 0.99),
    (r"capital.*russia", "Moscow", 0.99),
    (r"capital.*italy", "Rome", 0.99),
    (r"capital.*spain", "Madrid", 0.99),
    (r"capital.*portugal", "Lisbon", 0.99),
    (r"capital.*netherlands", "Amsterdam", 0.99),
    (r"capital.*belgium", "Brussels", 0.99),
    (r"capital.*switzerland", "Bern", 0.95),
    (r"capital.*austria", "Vienna", 0.99),
    (r"capital.*poland", "Warsaw", 0.99),
    (r"capital.*sweden", "Stockholm", 0.99),
    (r"capital.*norway", "Oslo", 0.99),
    (r"capital.*denmark", "Copenhagen", 0.99),
    (r"capital.*finland", "Helsinki", 0.99),
    (r"capital.*greece", "Athens", 0.99),
    (r"capital.*turkey", "Ankara", 0.99),
    (r"capital.*mexico", "Mexico City", 0.99),
    (r"capital.*argentina", "Buenos Aires", 0.99),
    (r"capital.*chile", "Santiago", 0.99),
    (r"capital.*colombia", "Bogota", 0.99),
    (r"capital.*peru", "Lima", 0.99),
    (r"capital.*south.*africa", "Pretoria (administrative)", 0.95),
    (r"capital.*nigeria", "Abuja", 0.95),
    (r"capital.*kenya", "Nairobi", 0.95),
    (r"capital.*thailand", "Bangkok", 0.99),
    (r"capital.*vietnam", "Hanoi", 0.99),
    (r"capital.*south.*korea", "Seoul", 0.99),
    (r"capital.*north.*korea", "Pyongyang", 0.99),
    (r"capital.*indonesia", "Jakarta", 0.95),
    (r"capital.*philippines", "Manila", 0.99),
    (r"capital.*malaysia", "Kuala Lumpur", 0.99),
    (r"capital.*singapore", "Singapore", 0.99),
    (r"capital.*new.*zealand", "Wellington", 0.95),
    (r"capital.*ireland", "Dublin", 0.99),
    (r"capital.*iceland", "Reykjavik", 0.99),
    (r"capital.*cuba", "Havana", 0.99),
    (r"capital.*jamaica", "Kingston", 0.95),
    (r"capital.*uae|capital.*emirates", "Abu Dhabi", 0.95),
    (r"capital.*saudi.*arabia", "Riyadh", 0.99),
    (r"capital.*iran", "Tehran", 0.99),
    (r"capital.*iraq", "Baghdad", 0.99),
    (r"capital.*israel", "Jerusalem (disputed), Tel Aviv (embassies)", 0.95),
    (r"capital.*pakistan", "Islamabad", 0.99),
    (r"capital.*bangladesh", "Dhaka", 0.99),
    (r"capital.*sri.*lanka", "Colombo/Sri Jayawardenepura Kotte", 0.95),
    (r"capital.*nepal", "Kathmandu", 0.99),
    (r"capital.*myanmar", "Naypyidaw", 0.95),
    (r"capital.*cambodia", "Phnom Penh", 0.99),
    (r"capital.*laos", "Vientiane", 0.99),
    (r"capital.*mongolia", "Ulaanbaatar", 0.95),
    (r"capital.*ukraine", "Kyiv", 0.99),
    (r"capital.*czech.*republic|capital.*czechia", "Prague", 0.99),
    (r"capital.*hungary", "Budapest", 0.99),
    (r"capital.*romania", "Bucharest", 0.99),
    (r"capital.*bulgaria", "Sofia", 0.99),
    (r"capital.*croatia", "Zagreb", 0.99),
    (r"capital.*serbia", "Belgrade", 0.99),
    (r"capital.*estonia", "Tallinn", 0.99),
    (r"capital.*latvia", "Riga", 0.99),
    (r"capital.*lithuania", "Vilnius", 0.99),
    # ═══ PLANETS & ASTRONOMY ═══
    (r"largest.*planet", "Jupiter", 0.99),
    (r"smallest.*planet", "Mercury", 0.99),
    (r"closest.*planet.*sun", "Mercury", 0.99),
    (r"farthest.*planet.*sun|outermost.*planet", "Neptune", 0.99),
    (r"hottest.*planet", "Venus (462°C)", 0.95),
    (r"coldest.*planet", "Neptune (-214°C)", 0.95),
    (r"red.*planet", "Mars", 0.99),
    (r"planet.*rings", "Saturn (and Jupiter, Uranus, Neptune)", 0.99),
    (r"blue.*planet", "Earth (from oceans)", 0.99),
    (r"moon.*distance|distance.*moon", "384,400 km from Earth", 0.99),
    (r"sun.*diameter", "1,391,000 km (109× Earth)", 0.99),
    (r"sun.*mass", "1.989 × 10³⁰ kg (333,000× Earth)", 0.99),
    (r"distance.*sun.*earth|earth.*sun.*distance", "149.6 million km (1 AU)", 0.99),
    (r"light.*year.*distance", "9.461 × 10¹² km", 0.99),
    (r"milky.*way.*diameter", "100,000 light-years", 0.95),
    (r"milky.*way.*stars", "100-400 billion stars", 0.95),
    (r"andromeda.*distance", "2.537 million light-years", 0.95),
    (r"hubble.*constant", "~70 km/s/Mpc", 0.95),
    (r"cosmic.*microwave.*background", "2.725 K (remnant of Big Bang)", 0.95),
    (r"solar.*system.*planets", "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune", 0.99),
    (r"asteroid.*belt", "between Mars and Jupiter", 0.99),
    (r"kuiper.*belt", "beyond Neptune, contains Pluto", 0.95),
    (r"oort.*cloud", "outermost region of solar system", 0.95),
    (r"dwarf.*planet", "Pluto, Eris, Haumea, Makemake, Ceres", 0.95),
    (r"galaxy.*type.*milky.*way", "barred spiral galaxy", 0.95),
    # ═══ MATH ═══
    (r"pythagorean", "a² + b² = c²", 0.99),
    (r"pi|π", "approximately 3.14159", 0.99),
    (r"euler.*number|e.*constant", "approximately 2.71828", 0.99),
    (r"2\s*\+\s*2|2\s*plus\s*2", "4", 0.99),
    (r"3\s*\*\s*3|3\s*times\s*3", "9", 0.99),
    (r"square.*root.*9|sqrt.*9", "3", 0.99),
    (r"golden.*ratio", "approximately 1.618 (φ)", 0.99),
    (r"fibonacci.*sequence", "0, 1, 1, 2, 3, 5, 8, 13, 21...", 0.99),
    (r"imaginary.*number|complex.*number", "i = √(-1)", 0.99),
    (r"area.*circle", "πr²", 0.99),
    (r"circumference.*circle", "2πr", 0.99),
    (r"volume.*sphere", "(4/3)πr³", 0.99),
    (r"surface.*area.*sphere", "4πr²", 0.99),
    (r"volume.*cylinder", "πr²h", 0.99),
    (r"area.*triangle", "½ × base × height", 0.99),
    (r"area.*rectangle", "length × width", 0.99),
    (r"perimeter.*square", "4 × side", 0.99),
    (r"quadratic.*formula", "x = (-b ± √(b²-4ac)) / 2a", 0.99),
    (r"logarithm.*base.*10|log.*10", "1", 0.99),
    (r"natural.*log.*e|ln.*e", "1", 0.99),
    (r"factorial.*5|5!", "120", 0.99),
    (r"factorial.*10|10!", "3,628,800", 0.99),
    (r"prime.*numbers.*100", "2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97", 0.99),
    (r"roman.*numeral", "I=1, V=5, X=10, L=50, C=100, D=500, M=1000", 0.99),
    (r"binary.*10|decimal.*10.*binary", "1010", 0.99),
    (r"hex.*255|255.*hex", "FF", 0.99),
    # ═══ CHEMISTRY ═══
    (r"h2o", "water", 0.99),
    (r"co2", "carbon dioxide", 0.99),
    (r"o2", "oxygen", 0.99),
    (r"n2", "nitrogen (78% of atmosphere)", 0.99),
    (r"nacl", "sodium chloride (table salt)", 0.99),
    (r"fe2o3", "iron oxide (rust)", 0.99),
    (r"periodic.*table.*elements", "118 confirmed elements", 0.99),
    (r"lightest.*element", "Hydrogen (H)", 0.99),
    (r"heaviest.*element", "Oganesson (Og, element 118)", 0.99),
    (r"most.*abundant.*element.*universe", "Hydrogen (~75%)", 0.99),
    (r"most.*abundant.*element.*earth.*crust", "Oxygen (~46%)", 0.99),
    (r"noble.*gas", "Helium, Neon, Argon, Krypton, Xenon, Radon", 0.99),
    (r"ph.*scale", "0-14 (7 is neutral)", 0.99),
    (r"acid.*ph", "less than 7", 0.99),
    (r"base.*ph|alkaline.*ph", "greater than 7", 0.99),
    (r"atomic.*number", "number of protons in an atom", 0.99),
    (r"atomic.*mass", "sum of protons and neutrons", 0.99),
    (r"electron.*shell", "K(2), L(8), M(18), N(32)...", 0.95),
    (r"valence.*electron", "electrons in outermost shell, determine bonding", 0.95),
    (r"covalent.*bond", "sharing electrons between atoms", 0.95),
    (r"ionic.*bond", "transfer of electrons between atoms", 0.95),
    (r"metallic.*bond", "sea of delocalized electrons", 0.95),
    (r"catalyst", "speeds up chemical reactions without being consumed", 0.99),
    (r"exothermic.*reaction", "releases heat energy", 0.99),
    (r"endothermic.*reaction", "absorbs heat energy", 0.99),
    # ═══ HUMAN BODY ═══
    (r"human.*body.*temperature", "37°C (98.6°F)", 0.99),
    (r"average.*human.*height", "170 cm (5 foot 7 inches)", 0.95),
    (r"human.*lifespan|average.*age", "72-80 years", 0.95),
    (r"human.*brain.*neuron|neuron.*brain", "86 billion neurons", 0.95),
    (r"human.*bone|bone.*human", "206 bones", 0.99),
    (r"human.*heart.*beat|heart.*rate", "60-100 beats per minute", 0.95),
    (r"largest.*organ", "Skin (about 2 m²)", 0.99),
    (r"smallest.*bone", "Stapes (stirrup bone in ear)", 0.99),
    (r"longest.*bone", "Femur (thigh bone)", 0.99),
    (r"smallest.*muscle", "Stapedius (in ear)", 0.99),
    (r"largest.*muscle", "Gluteus maximus (buttock)", 0.99),
    (r"lungs.*capacity", "about 6 liters (total)", 0.95),
    (r"blood.*volume", "about 5 liters (adult)", 0.95),
    (r"stomach.*ph", "1.5-3.5 (very acidic)", 0.95),
    (r"digestion.*time", "24-72 hours (full transit)", 0.90),
    (r"liver.*function", "detoxification, protein synthesis, bile production", 0.95),
    (r"kidney.*function", "filtration, waste removal, fluid balance", 0.95),
    (r"pancreas.*function", "insulin production, digestive enzymes", 0.95),
    (r"thyroid.*function", "metabolism regulation via hormones", 0.95),
    (r"adrenaline.*function", "fight-or-flight response hormone", 0.99),
    (r"insulin.*function", "regulates blood sugar levels", 0.99),
    (r"melatonin.*function", "regulates sleep-wake cycle", 0.99),
    (r"vitamin.*c.*deficiency", "scurvy", 0.99),
    (r"vitamin.*d.*source", "sunlight, fatty fish, fortified milk", 0.95),
    (r"iron.*deficiency", "anemia (fatigue, weakness)", 0.95),
    # ═══ TECHNOLOGY ═══
    (r"python.*language|language.*python", "high-level programming language", 0.99),
    (r"javascript|js", "web programming language", 0.99),
    (r"html", "HyperText Markup Language for web pages", 0.99),
    (r"css", "Cascading Style Sheets for styling web pages", 0.99),
    (r"api", "Application Programming Interface", 0.99),
    (r"sql", "Structured Query Language for databases", 0.99),
    (r"linux", "open-source operating system kernel by Linus Torvalds", 0.99),
    (r"windows.*os", "Microsoft operating system", 0.99),
    (r"macos|mac.*os", "Apple operating system", 0.99),
    (r"android.*os", "Google mobile operating system", 0.99),
    (r"ios.*os", "Apple mobile operating system", 0.99),
    (r"git", "distributed version control system by Linus Torvalds", 0.99),
    (r"docker", "containerization platform for deploying applications", 0.95),
    (r"kubernetes|k8s", "container orchestration system", 0.95),
    (r"machine.*learning", "AI that learns patterns from data", 0.99),
    (r"deep.*learning", "neural networks with multiple layers", 0.99),
    (r"neural.*network", "computing system inspired by biological neurons", 0.99),
    (r"artificial.*intelligence|ai", "simulation of human intelligence by machines", 0.99),
    (r"natural.*language.*processing|nlp", "AI that understands human language", 0.99),
    (r"computer.*vision", "AI that interprets visual information", 0.99),
    (r"blockchain", "decentralized distributed ledger technology", 0.99),
    (r"cryptocurrency", "digital currency using cryptography", 0.99),
    (r"bitcoin", "first decentralized cryptocurrency (2009)", 0.99),
    (r"ethereum", "blockchain platform for smart contracts", 0.99),
    (r"cloud.*computing", "on-demand computing resources over the internet", 0.99),
    (r"5g.*technology", "fifth generation mobile network (speeds up to 20 Gbps)", 0.95),
    (r"quantum.*computing", "computing using quantum mechanical phenomena", 0.99),
    (r"vr|virtual.*reality", "computer-generated simulation of 3D environment", 0.99),
    (r"ar|augmented.*reality", "overlaying digital content on real world", 0.99),
    (r"iot|internet.*of.*things", "network of connected physical devices", 0.99),
    # ═══ INVENTIONS ═══
    (r"telephone.*invent|invent.*telephone", "Alexander Graham Bell in 1876", 0.99),
    (r"light.*bulb.*invent|invent.*light.*bulb", "Thomas Edison in 1879", 0.95),
    (r"internet.*invent|invent.*internet", "ARPANET in 1969, WWW in 1989", 0.95),
    (r"airplane.*invent|invent.*airplane|first.*flight", "Wright Brothers in 1903", 0.99),
    (r"television.*invent|invent.*television", "Philo Farnsworth in 1927", 0.95),
    (r"radio.*invent|invent.*radio", "Guglielmo Marconi in 1895", 0.95),
    (r"camera.*invent|invent.*camera", "Joseph Nicéphore Niépce in 1826", 0.95),
    (r"printing.*press.*invent", "Johannes Gutenberg around 1440", 0.99),
    (r"steam.*engine.*invent", "James Watt improved it in 1769", 0.95),
    (r"electricity.*invent|invent.*electricity", "Benjamin Franklin, Michael Faraday, Nikola Tesla", 0.95),
    (r"battery.*invent|invent.*battery", "Alessandro Volta in 1800", 0.95),
    (r"penicillin.*discover", "Alexander Fleming in 1928", 0.99),
    (r"x.*ray.*discover", "Wilhelm Röntgen in 1895", 0.99),
    (r"dynamite.*invent", "Alfred Nobel in 1867", 0.99),
    (r"vacuum.*cleaner.*invent", "Hubert Cecil Booth in 1901", 0.95),
    (r"refrigerator.*invent", "Jacob Perkins in 1834", 0.95),
    (r"washing.*machine.*invent", "Alva J. Fisher in 1908", 0.95),
    (r"solar.*panel.*invent", "Bell Labs in 1954", 0.95),
    (r"gps.*invent", "US Department of Defense, fully operational 1995", 0.95),
    (r"www.*invent|world.*wide.*web.*invent", "Tim Berners-Lee in 1989", 0.99),
    # ═══ GEOGRAPHY ═══
    (r"longest.*river", "Nile River, 6,650 km", 0.95),
    (r"longest.*river.*world", "Nile (6,650 km) or Amazon (6,400 km)", 0.95),
    (r"tallest.*mountain", "Mount Everest, 8,849 m", 0.99),
    (r"deepest.*ocean.*point", "Mariana Trench, 10,994 m", 0.99),
    (r"largest.*ocean", "Pacific Ocean", 0.99),
    (r"largest.*desert", "Antarctic Desert (14.2 million km²)", 0.95),
    (r"largest.*hot.*desert", "Sahara Desert (9.2 million km²)", 0.95),
    (r"largest.*continent", "Asia (44.6 million km²)", 0.99),
    (r"smallest.*continent", "Australia (7.7 million km²)", 0.99),
    (r"longest.*wall", "Great Wall of China, 21,196 km", 0.99),
    (r"largest.*lake", "Caspian Sea (371,000 km²)", 0.99),
    (r"deepest.*lake", "Lake Baikal, 1,642 m", 0.99),
    (r"largest.*island", "Greenland (2,166,086 km²)", 0.99),
    (r"largest.*country.*area", "Russia (17.1 million km²)", 0.99),
    (r"smallest.*country", "Vatican City (0.44 km²)", 0.99),
    (r"most.*populated.*country|largest.*population", "India (1.4 billion)", 0.95),
    (r"most.*populated.*city", "Tokyo (37 million metro)", 0.95),
    (r"amazon.*rainforest.*size", "5.5 million km²", 0.95),
    (r"amazon.*rainforest.*countries", "Brazil, Peru, Colombia, and 7 others", 0.95),
    (r"sahara.*temperature", "up to 58°C (136°F) daytime", 0.95),
    (r"mountain.*everest.*grow", "grows ~4mm per year", 0.95),
    (r"pacific.*ocean.*size", "165.25 million km²", 0.95),
    (r"atlantic.*ocean.*size", "106.46 million km²", 0.95),
    (r"arctic.*ocean.*size", "14.06 million km² (smallest)", 0.95),
    (r"seven.*wonders.*ancient", "Pyramids of Giza, Hanging Gardens, Statue of Zeus, Temple of Artemis, Mausoleum, Colossus, Lighthouse of Alexandria", 0.99),
    (r"seven.*wonders.*modern", "Great Wall, Petra, Christ Redeemer, Machu Picchu, Chichen Itza, Colosseum, Taj Mahal", 0.99),
    # ═══ QUANTUM MECHANICS ═══
    (r"quantum.*mechanics", "physics of subatomic particles", 0.95),
    (r"quantum.*superposition", "particle exists in multiple states until measured", 0.99),
    (r"quantum.*entanglement", "particles correlated across distance instantly", 0.99),
    (r"heisenberg.*uncertainty", "cannot know exact position and momentum simultaneously", 0.99),
    (r"schrodinger.*cat", "thought experiment: cat is both alive and dead until observed", 0.99),
    (r"wave.*particle.*duality", "light and matter exhibit both wave and particle properties", 0.99),
    (r"quantum.*tunneling", "particles pass through barriers they classically shouldn't", 0.99),
    # ═══ SPACE ═══
    (r"sun.*age", "4.6 billion years", 0.95),
    (r"earth.*age", "4.54 billion years", 0.95),
    (r"universe.*age", "13.8 billion years", 0.95),
    (r"sun.*type", "G-type main-sequence star (yellow dwarf)", 0.95),
    (r"sun.*surface.*temp", "5,500°C (9,932°F)", 0.95),
    (r"sun.*core.*temp", "15 million°C", 0.95),
    (r"mars.*olympus.*mons", "tallest volcano in solar system, 21.9 km", 0.95),
    (r"saturn.*density", "less dense than water (0.687 g/cm³)", 0.95),
    (r"venus.*day.*longer", "Venus day (243 Earth days) > Venus year (225 Earth days)", 0.95),
    (r"mercury.*day.*year", "Mercury day (59 Earth days) ≠ Mercury year (88 Earth days)", 0.95),
    (r"neutron.*star.*density", "10¹⁷ kg/m³ (a teaspoon weighs ~6 billion tons)", 0.95),
    (r"black.*hole.*escape", "Nothing, not even light, can escape beyond the event horizon", 0.99),
    (r"event.*horizon.*radius", "Schwarzschild radius = 2GM/c²", 0.95),
    (r"hawking.*radiation", "thermal radiation emitted by black holes due to quantum effects", 0.95),
    (r"cosmic.*expansion", "universe is expanding, accelerating since ~5 billion years ago", 0.95),
    (r"dark.*matter", "~27% of universe, does not emit light", 0.95),
    (r"dark.*energy", "~68% of universe, drives accelerating expansion", 0.95),
    # ═══ LANGUAGE ═══
    (r"most.*spoken.*language", "Mandarin Chinese (~920 million native)", 0.95),
    (r"most.*spoken.*language.*total", "English (~1.5 billion total)", 0.95),
    (r"most.*written.*language", "English", 0.95),
    (r"oldest.*language", "Sumerian (~3400 BC) or Sanskrit (~1500 BC)", 0.90),
    (r"language.*families", "~140 families, 7,000+ languages", 0.95),
    (r"dead.*language", "Latin, Sanskrit, Ancient Greek, Sumerian", 0.95),
    (r"constructed.*language", "Esperanto (most popular), Klingon, Elvish", 0.95),
    # ═══ COUNTRIES & DEMOGRAPHICS ═══
    (r"country.*flags.*colors", "Most common: red, white, blue", 0.90),
    (r"country.*most.*languages", "Papua New Guinea (840 languages)", 0.95),
    (r"country.*highest.*gdp", "United States (~$25 trillion)", 0.95),
    (r"country.*highest.*gdp.*per.*capita", "Luxembourg (~$126,000)", 0.95),
    (r"country.*highest.*hdi", "Switzerland (0.962)", 0.95),
    (r"country.*largest.*economy.*gdp", "United States", 0.95),
    (r"country.*most.*islands", "Sweden (267,570 islands)", 0.95),
    (r"country.*most.*time.*zones", "France (12 time zones)", 0.95),
    (r"country.*youngest.*population", "Niger (median age 14.8)", 0.95),
    (r"country.*oldest.*population", "Japan (median age 48.6)", 0.95),
    # ═══ MUSIC ═══
    (r"music.*octave", "12 semitones: C, C#, D, D#, E, F, F#, G, G#, A, A#, B", 0.99),
    (r"piano.*keys", "88 keys (52 white, 36 black)", 0.99),
    (r"guitar.*strings", "6 strings (standard tuning)", 0.99),
    (r"violin.*strings", "4 strings (G, D, A, E)", 0.99),
    (r"tempo.*bpm.*andante", "walking pace, 76-108 BPM", 0.95),
    (r"tempo.*bpm.*allegro", "fast, 120-156 BPM", 0.95),
    # ═══ SPORTS ═══
    (r"olympics.*origin", "Ancient Greece, 776 BC", 0.99),
    (r"modern.*olympics.*start", "Athens 1896", 0.99),
    (r"olympic.*rings", "5 rings representing 5 continents", 0.99),
    (r"fastest.*100m", "Usain Bolt, 9.58 seconds (2009)", 0.99),
    (r"football.*soccer.*world.*cup.*winner", "Brazil (5 titles)", 0.95),
    (r"nba.*championships.*most", "Boston Celtics and Los Angeles Lakers (17 each)", 0.95),
    # ═══ ECONOMICS ═══
    (r"inflation.*definition", "general increase in prices over time", 0.99),
    (r"gdp.*definition", "Gross Domestic Product, total value of goods/services", 0.99),
    (r"supply.*demand.*law", "higher supply → lower price, higher demand → higher price", 0.99),
    (r"interest.*rate.*definition", "cost of borrowing money, expressed as percentage", 0.99),
    (r"recession.*definition", "two consecutive quarters of negative GDP growth", 0.99),
    (r"monopoly.*definition", "single seller dominating a market", 0.99),
    (r"opportunity.*cost", "value of the next best alternative forgone", 0.99),
    # ═══ FOOD & NUTRITION ═══
    (r"most.*water.*fruit", "Watermelon (92%)", 0.95),
    (r"highest.*protein.*food", "Eggs and lean meats", 0.90),
    (r"food.*calories.*daily", "2,000-2,500 for adults", 0.90),
    (r"food.*pyramid", "grains, vegetables, fruits, protein, dairy (top)", 0.95),
    (r"caffeine.*coffee.*mg", "~95 mg per 8oz cup", 0.95),
    (r"caffeine.*tea.*mg", "~25-50 mg per 8oz cup", 0.95),
    (r"sugar.*daily.*limit", "25g (women), 36g (men) per WHO", 0.95),
    (r"sodium.*daily.*limit", "2,300 mg per day", 0.95),
    (r"fiber.*daily.*intake", "25-38g per day", 0.95),
    (r"water.*daily.*intake", "2-3 liters per day (8 cups)", 0.95),
]


class FactualCore:
    """Core A — Factual knowledge lookup and verification.

    Uses pre-compiled regex patterns with a keyword index for
    fast first-pass filtering.  Falls back to number extraction
    from evidence when no direct fact match is found.
    """

    CORE_ID = "factual"

    def __init__(self) -> None:
        self._facts: list[tuple[re.Pattern[str], str, float]] = []
        self._keyword_index: dict[str, list[int]] = {}
        self._build_index()

    # ── Public API (NeuralCoreProtocol) ─────────────────────

    @property
    def core_id(self) -> str:  # noqa: D401
        return self.CORE_ID

    def process(self, query: str, evidence: list[str]) -> CoreResult:
        t0 = __import__("time").perf_counter()

        # Fast path: keyword-indexed lookup
        result = self._lookup_by_keywords(query)
        if result is not None:
            return make_result(self.CORE_ID, result[0], result[1],
                               f"Fact match", t0)

        # Fallback: extract numbers from evidence
        if evidence:
            for ev in evidence:
                numbers = re.findall(r"\b\d[\d,\.]*\b", ev)
                if numbers:
                    return make_result(self.CORE_ID, numbers[0], 0.7,
                                       "Extracted number from evidence", t0,
                                       evidence_used=1)

        return empty_result(self.CORE_ID, t0)

    # ── Internal ────────────────────────────────────────────

    def _build_index(self) -> None:
        for i, (pattern, answer, confidence) in enumerate(_RAW_FACTS):
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue
            self._facts.append((compiled, answer, confidence))
            for word in re.findall(r"[a-z]{3,}", pattern):
                self._keyword_index.setdefault(word, []).append(i)

    def _lookup_by_keywords(self, query: str) -> tuple[str, float] | None:
        q_words = set(re.findall(r"[a-z]{3,}", query.lower()))
        candidate_indices: set[int] = set()
        for w in q_words:
            if w in self._keyword_index:
                candidate_indices.update(self._keyword_index[w])
        if not candidate_indices:
            candidate_indices = set(range(len(self._facts)))

        for idx in candidate_indices:
            if idx < len(self._facts):
                compiled, answer, confidence = self._facts[idx]
                if compiled.search(query):
                    return answer, confidence
        return None
