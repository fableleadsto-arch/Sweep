"""
Knowledge Training Module — structured knowledge from authoritative sources.

Sources:
- Khan Academy (physics, biology, chemistry, math, history)
- Britannica (encyclopedia facts)
- NASA (space science, earth science)
- USGS (geology, geography)
- NIH/WHO (biology, health)
- Wolfram Alpha (mathematical facts)

This module provides structured knowledge that can be loaded into:
- WorldKnowledge (entity properties)
- SemanticMemory (topic knowledge)
- CommonSense (default rules)
- GeneralIntelligence (reasoning facts)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeEntry:
    """A single knowledge entry from an authoritative source."""
    id: str
    domain: str
    topic: str
    fact: str
    answer: str
    confidence: float
    source: str
    category: str  # "entity", "relation", "process", "law", "event", "formula"
    reasoning: str = ""
    related: list[str] = field(default_factory=list)


class KnowledgeTrainer:
    """
    Comprehensive knowledge base from authoritative sources.

    Provides 500+ structured knowledge entries across 12 domains.
    Each entry includes: source, confidence, reasoning, and relationships.
    """

    def __init__(self) -> None:
        self._entries: list[KnowledgeEntry] = []
        self._load_physics()
        self._load_biology()
        self._load_chemistry()
        self._load_earth_science()
        self._load_astronomy()
        self._load_mathematics()
        self._load_history()
        self._load_geography()
        self._load_technology()
        self._load_health()
        self._load_economics()
        self._load_ecology()
        # New domains
        self._load_literature()
        self._load_philosophy()
        self._load_computer_science()
        self._load_music_arts()
        self._load_sociology()
        self._load_law_politics()
        # Load supplementary knowledge
        self._load_supplementary()

    def get_all(self) -> list[KnowledgeEntry]:
        """Return all knowledge entries."""
        return self._entries

    def get_by_domain(self, domain: str) -> list[KnowledgeEntry]:
        """Return entries for a specific domain."""
        return [e for e in self._entries if e.domain == domain]

    def get_by_category(self, category: str) -> list[KnowledgeEntry]:
        """Return entries for a specific category."""
        return [e for e in self._entries if e.category == category]

    def to_world_knowledge(self) -> list[dict[str, Any]]:
        """Convert entries to WorldKnowledge entity format."""
        entities = []
        for entry in self._entries:
            if entry.category == "entity":
                entities.append({
                    "name": entry.topic,
                    "domain": entry.domain,
                    "fact": entry.fact,
                    "answer": entry.answer,
                    "confidence": entry.confidence,
                    "source": entry.source,
                })
        return entities

    def to_semantic_memory(self) -> list[dict[str, Any]]:
        """Convert entries to semantic memory format."""
        memories = []
        for entry in self._entries:
            memories.append({
                "topic": entry.topic,
                "fact": entry.fact,
                "domain": entry.domain,
                "confidence": entry.confidence,
                "source": entry.source,
            })
        return memories

    # ══════════════════════════════════════════════════════════════
    # PHYSICS — Khan Academy, Feynman Lectures
    # ══════════════════════════════════════════════════════════════

    def _load_physics(self) -> None:
        d = "physics"
        s = "Khan Academy Physics / Feynman Lectures"

        # Newton's Laws
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Newton's First Law",
                          "An object at rest stays at rest, and an object in motion stays in motion unless acted upon by a force",
                          "yes", 0.99, s, "law",
                          "Also called the law of inertia"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Newton's Second Law",
                          "Force equals mass times acceleration (F = ma)",
                          "F = ma", 0.99, s, "formula"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Newton's Third Law",
                          "For every action, there is an equal and opposite reaction",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Gravitational Force",
                          "F = G * m1 * m2 / r^2",
                          "F = G * m1 * m2 / r^2", 0.99, s, "formula"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Weight",
                          "Weight is the force of gravity on an object (W = mg)",
                          "W = mg", 0.99, s, "formula"),
        ])

        # Energy
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Kinetic Energy",
                          "KE = 1/2 * m * v^2",
                          "KE = 1/2 * m * v^2", 0.99, s, "formula"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Potential Energy",
                          "PE = m * g * h",
                          "PE = m * g * h", 0.99, s, "formula"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Conservation of Energy",
                          "Energy cannot be created or destroyed, only converted",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Power",
                          "Power is the rate of energy transfer (P = W/t)",
                          "P = W/t", 0.99, s, "formula"),
        ])

        # Waves and Optics
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Speed of Light",
                          "The speed of light in vacuum is approximately 299,792,458 m/s",
                          "299792458", 0.99, s, "fact"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Sound Speed",
                          "Sound travels at approximately 343 m/s in air at 20°C",
                          "343", 0.95, s, "fact"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Frequency and Pitch",
                          "Higher frequency sound waves have higher pitch",
                          "yes", 0.99, s, "relation"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Reflection Law",
                          "Angle of incidence equals angle of reflection",
                          "yes", 0.99, s, "law"),
        ])

        # Thermodynamics
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Heat Transfer",
                          "Heat flows from hot to cold objects",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Thermal Expansion",
                          "Most materials expand when heated",
                          "yes", 0.95, s, "fact"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Absolute Zero",
                          "Absolute zero is -273.15°C (0 Kelvin), the lowest possible temperature",
                          "-273.15", 0.99, s, "fact"),
        ])

        # Electromagnetism
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Electric Current",
                          "Electric current is the flow of electric charge",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Ohm's Law",
                          "V = IR (Voltage = Current × Resistance)",
                          "V = IR", 0.99, s, "formula"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Magnetic Poles",
                          "Like poles repel, opposite poles attract",
                          "yes", 0.99, s, "law"),
        ])

        # Quantum Mechanics
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Planck's Constant",
                          "Planck's constant h = 6.626 x 10^-34 J·s, relates photon energy to frequency",
                          "6.626e-34", 0.99, s, "constant"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Photoelectric Effect",
                          "Light can eject electrons from metal surfaces; photon energy must exceed work function",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Wave-Particle Duality",
                          "Particles like electrons exhibit both wave and particle properties",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Heisenberg Uncertainty Principle",
                          "You cannot simultaneously know exact position and momentum of a particle",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Quantum Superposition",
                          "A quantum system can exist in multiple states simultaneously until measured",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Quantum Entanglement",
                          "Two particles can be correlated such that measuring one instantly affects the other",
                          "yes", 0.99, s, "phenomenon"),
        ])

        # Relativity
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Special Relativity",
                          "E = mc², the speed of light is constant for all observers, time dilates at high speeds",
                          "E = mc²", 0.99, s, "theory"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Time Dilation",
                          "Time passes slower for objects moving close to the speed of light",
                          "yes", 0.99, s, "phenomenon"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Length Contraction",
                          "Objects moving at high speeds appear shorter in the direction of motion",
                          "yes", 0.99, s, "phenomenon"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "General Relativity",
                          "Gravity is the curvature of spacetime caused by mass and energy",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Gravitational Waves",
                          "Ripples in spacetime caused by accelerating massive objects",
                          "yes", 0.99, s, "phenomenon"),
        ])

        # Nuclear Physics
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Radioactive Decay",
                          "Unstable atoms emit radiation (alpha, beta, gamma) to become stable",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Half-Life",
                          "Half-life is the time for half of a radioactive sample to decay",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Nuclear Fission",
                          "Splitting heavy atomic nuclei releases energy (used in nuclear reactors)",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Nuclear Fusion",
                          "Combining light nuclei releases energy (powers the Sun)",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Atomic Mass Unit",
                          "1 amu = 1.6605 x 10^-27 kg, approximately the mass of a proton",
                          "1.6605e-27", 0.99, s, "constant"),
        ])

        # Modern Physics
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Standard Model",
                          "The Standard Model describes fundamental particles and forces (excluding gravity)",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Higgs Boson",
                          "The Higgs boson gives particles mass; discovered at CERN in 2012",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Boltzmann Constant",
                          "k_B = 1.381 x 10^-23 J/K, relates temperature to average kinetic energy",
                          "1.381e-23", 0.99, s, "constant"),
        ])

        # Thermodynamics Laws
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Zeroth Law of Thermodynamics",
                          "If two systems are each in thermal equilibrium with a third, they are in equilibrium with each other",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "First Law of Thermodynamics",
                          "Energy cannot be created or destroyed (conservation of energy)",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Second Law of Thermodynamics",
                          "Entropy of an isolated system always increases",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Third Law of Thermodynamics",
                          "As temperature approaches absolute zero, entropy approaches a minimum",
                          "yes", 0.99, s, "law"),
        ])

        # Fluid Mechanics
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Bernoulli's Principle",
                          "In a flowing fluid, increased speed occurs with decreased pressure",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Archimedes' Principle",
                          "A body submerged in fluid experiences an upward buoyant force equal to weight of displaced fluid",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Pascal's Law",
                          "Pressure applied to an enclosed fluid is transmitted undiminished throughout the fluid",
                          "yes", 0.99, s, "law"),
        ])

        # Electrical
        self._entries.extend([
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Coulomb's Law",
                          "F = k*q1*q2/r^2, electric force between two charges",
                          "F = k*q1*q2/r^2", 0.99, s, "formula"),
            KnowledgeEntry(f"phys_{len(self._entries)}", d, "Kirchhoff's Laws",
                          "Current entering a junction equals current leaving; voltage around a loop sums to zero",
                          "yes", 0.99, s, "laws"),
        ])

    # ══════════════════════════════════════════════════════════════
    # BIOLOGY — Khan Academy Biology, Campbell Biology
    # ══════════════════════════════════════════════════════════════

    def _load_biology(self) -> None:
        d = "biology"
        s = "Khan Academy Biology / Campbell Biology"

        # Cell Biology
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Cell Theory",
                          "All living things are made of cells, cells are the basic unit of life, all cells come from pre-existing cells",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "DNA Function",
                          "DNA carries genetic information that determines traits",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Protein Synthesis",
                          "DNA → RNA → Protein (central dogma of molecular biology)",
                          "DNA → RNA → Protein", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Mitochondria",
                          "Mitochondria are the powerhouse of the cell, producing ATP through cellular respiration",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Photosynthesis",
                          "6CO2 + 6H2O + light → C6H12O6 + 6O2",
                          "6CO2 + 6H2O + light → C6H12O6 + 6O2", 0.99, s, "formula"),
        ])

        # Genetics
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Mendel's Laws",
                          "Traits are inherited through discrete units (genes), dominant alleles mask recessive ones",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Alleles",
                          "Different versions of a gene are called alleles",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Dominant vs Recessive",
                          "Dominant traits appear when at least one dominant allele is present; recessive traits appear only with two recessive alleles",
                          "yes", 0.99, s, "relation"),
        ])

        # Evolution
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Natural Selection",
                          "Organisms with favorable traits are more likely to survive and reproduce",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Evolution",
                          "Evolution is change in allele frequencies in a population over time",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Common Ancestry",
                          "All life on Earth shares a common ancestor",
                          "yes", 0.95, s, "theory"),
        ])

        # Human Biology
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Circulatory System",
                          "The heart pumps blood through arteries and veins to deliver oxygen and nutrients",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Nervous System",
                          "The nervous system uses electrical signals to coordinate body functions",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Immune System",
                          "The immune system defends against pathogens using white blood cells and antibodies",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Digestive System",
                          "Food is broken down mechanically and chemically to extract nutrients",
                          "yes", 0.99, s, "process"),
        ])

        # Ecology
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Food Chain",
                          "Energy flows from producers to consumers in a food chain",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Ecosystem",
                          "An ecosystem includes all living organisms and their physical environment interacting as a system",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Biodiversity",
                          "Biodiversity is the variety of life in a particular habitat or ecosystem",
                          "yes", 0.99, s, "definition"),
        ])

        # Microbiology
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Bacteria Structure",
                          "Bacteria are prokaryotes with cell walls, ribosomes, and circular DNA",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Virus Structure",
                          "Viruses are non-living particles with genetic material (DNA or RNA) in a protein coat",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Bacterial Reproduction",
                          "Bacteria reproduce asexually through binary fission",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Antibiotic Resistance",
                          "Bacteria can evolve resistance to antibiotics through natural selection",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Fermentation",
                          "Fermentation converts sugars to alcohol or acids without oxygen",
                          "yes", 0.99, s, "process"),
        ])

        # Botany
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Plant Cell Structure",
                          "Plant cells have cell walls, chloroplasts, and large central vacuoles",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Transpiration",
                          "Transpiration is water loss through plant leaves that drives water uptake",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Xylem and Phloem",
                          "Xylem transports water upward; phloem transports sugars throughout the plant",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Phototropism",
                          "Plants grow toward light due to the hormone auxin",
                          "yes", 0.99, s, "process"),
        ])

        # Human Anatomy
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Skeletal System",
                          "The human body has 206 bones that provide structure, protection, and movement",
                          "206", 0.99, s, "fact"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Muscular System",
                          "There are over 600 muscles in the human body that enable movement",
                          "600+", 0.99, s, "fact"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Brain Structure",
                          "The brain has cerebrum, cerebellum, and brainstem; it controls body functions",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Kidney Function",
                          "Kidneys filter blood, remove waste, and regulate fluid and electrolyte balance",
                          "yes", 0.99, s, "function"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Lung Function",
                          "Lungs exchange oxygen and carbon dioxide through alveoli",
                          "yes", 0.99, s, "function"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Liver Function",
                          "The liver detoxifies blood, produces bile, and stores glycogen",
                          "yes", 0.99, s, "function"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "White Blood Cells",
                          "White blood cells (leukocytes) fight infections as part of the immune system",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Red Blood Cells",
                          "Red blood cells carry oxygen using hemoglobin; they have no nucleus",
                          "yes", 0.99, s, "entity"),
        ])

        # Taxonomy
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Linnaean Taxonomy",
                          "Classification: Domain > Kingdom > Phylum > Class > Order > Family > Genus > Species",
                          "yes", 0.99, s, "system"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Three Domains of Life",
                          "All life is classified into three domains: Bacteria, Archaea, Eukarya",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Six Kingdoms",
                          "The six kingdoms are Bacteria, Archaea, Protista, Fungi, Plantae, Animalia",
                          "yes", 0.99, s, "classification"),
        ])

        # Biochemistry
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "ATP",
                          "ATP (adenosine triphosphate) is the energy currency of the cell",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Enzyme Function",
                          "Enzymes are biological catalysts that speed up chemical reactions",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Lipids",
                          "Lipids include fats, phospholipids, and steroids; they store energy and form membranes",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Carbohydrates",
                          "Carbohydrates are sugars and starches that provide quick energy",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Cell Membrane",
                          "The cell membrane is a phospholipid bilayer that controls what enters and exits the cell",
                          "yes", 0.99, s, "structure"),
        ])

        # Ecology deeper
        self._entries.extend([
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Symbiosis",
                          "Symbiosis is a close relationship between species (mutualism, commensalism, parasitism)",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"bio_{len(self._entries)}", d, "Biogeochemical Cycles",
                          "Matter cycles through ecosystems via water, carbon, nitrogen, and phosphorus cycles",
                          "yes", 0.99, s, "process"),
        ])

    # ══════════════════════════════════════════════════════════════
    # CHEMISTRY — Khan Academy Chemistry
    # ══════════════════════════════════════════════════════════════

    def _load_chemistry(self) -> None:
        d = "chemistry"
        s = "Khan Academy Chemistry"

        # Atomic Structure
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Atomic Number",
                          "The atomic number equals the number of protons in an atom",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Mass Number",
                          "Mass number = protons + neutrons",
                          "yes", 0.99, s, "formula"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Electron Configuration",
                          "Electrons fill orbitals in order of increasing energy",
                          "yes", 0.99, s, "rule"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Valence Electrons",
                          "Valence electrons are in the outermost shell and determine chemical properties",
                          "yes", 0.99, s, "definition"),
        ])

        # Periodic Table
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Periodic Table Organization",
                          "Elements are arranged by increasing atomic number, with similar properties in groups",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Metals Properties",
                          "Metals are conductive, malleable, ductile, and shiny",
                          "yes", 0.99, s, "properties"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Nonmetals Properties",
                          "Nonmetals are generally poor conductors and can be gases, liquids, or brittle solids",
                          "yes", 0.99, s, "properties"),
        ])

        # Chemical Reactions
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Chemical Reaction",
                          "A chemical reaction transforms reactants into products by breaking and forming bonds",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Balancing Equations",
                          "Chemical equations must be balanced (same number of atoms on both sides)",
                          "yes", 0.99, s, "rule"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Oxidation",
                          "Oxidation is loss of electrons; reduction is gain of electrons (OIL RIG)",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Acids and Bases",
                          "Acids donate H+ ions; bases accept H+ ions. They neutralize each other to form water and salt",
                          "yes", 0.99, s, "process"),
        ])

        # States of Matter
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "States of Matter",
                          "Matter exists as solid, liquid, gas, or plasma",
                          "solid, liquid, gas, plasma", 0.99, s, "classification"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Phase Changes",
                          "Melting (solid→liquid), boiling (liquid→gas), condensation (gas→liquid), freezing (liquid→solid)",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Density",
                          "Density = mass/volume. Objects less dense than water float; more dense objects sink",
                          "yes", 0.99, s, "relation"),
        ])

        # Organic Chemistry
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Carbon Bonding",
                          "Carbon forms 4 covalent bonds and can create chains, rings, and complex structures",
                          "yes", 0.99, s, "property"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Hydrocarbons",
                          "Hydrocarbons are organic compounds containing only carbon and hydrogen",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Functional Groups",
                          "Functional groups (hydroxyl, carboxyl, amino) determine chemical properties of organic molecules",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Polymers",
                          "Polymers are large molecules made of repeating monomer units",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Isomers",
                          "Isomers are molecules with the same formula but different structures",
                          "yes", 0.99, s, "definition"),
        ])

        # Chemical Bonds
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Ionic Bonds",
                          "Ionic bonds form when electrons are transferred between atoms",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Covalent Bonds",
                          "Covalent bonds form when atoms share electrons",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Metallic Bonds",
                          "Metallic bonds involve a sea of delocalized electrons shared among metal atoms",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Hydrogen Bonds",
                          "Hydrogen bonds are weak attractions between partial charges on different molecules",
                          "yes", 0.99, s, "definition"),
        ])

        # Chemical Kinetics
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Reaction Rate",
                          "Reaction rate depends on temperature, concentration, surface area, and catalysts",
                          "yes", 0.99, s, "factors"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Catalyst",
                          "A catalyst speeds up a chemical reaction without being consumed",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Activation Energy",
                          "Activation energy is the minimum energy needed to start a chemical reaction",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Le Chatelier's Principle",
                          "If a system at equilibrium is disturbed, it shifts to counteract the disturbance",
                          "yes", 0.99, s, "law"),
        ])

        # Solutions
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Solutions",
                          "A solution is a homogeneous mixture of solute dissolved in solvent",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Solubility",
                          "Solubility is the maximum amount of solute that dissolves in a given amount of solvent",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Molarity",
                          "Molarity = moles of solute / liters of solution",
                          "yes", 0.99, s, "formula"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "pH Scale",
                          "pH ranges from 0 (acidic) to 14 (basic); 7 is neutral. pH = -log[H+]",
                          "yes", 0.99, s, "scale"),
        ])

        # Biochemistry
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Amino Acids",
                          "There are 20 standard amino acids that form proteins",
                          "20", 0.99, s, "fact"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Nucleotides",
                          "Nucleotides are the building blocks of DNA and RNA",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Carbohydrate Structure",
                          "Simple sugars (monosaccharides) combine to form disaccharides and polysaccharides",
                          "yes", 0.99, s, "structure"),
        ])

        # Materials
        self._entries.extend([
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Crystal Structure",
                          "Crystals have atoms arranged in repeating three-dimensional patterns",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"chem_{len(self._entries)}", d, "Allotropes",
                          "Allotropes are different forms of the same element (e.g., diamond and graphite are carbon)",
                          "yes", 0.99, s, "definition"),
        ])

    # ══════════════════════════════════════════════════════════════
    # EARTH SCIENCE — NASA, USGS
    # ══════════════════════════════════════════════════════════════

    def _load_earth_science(self) -> None:
        d = "earth_science"
        s = "NASA / USGS"

        self._entries.extend([
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Earth's Layers",
                          "Earth has four layers: crust, mantle, outer core, inner core",
                          "crust, mantle, outer core, inner core", 0.99, s, "structure"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Plate Tectonics",
                          "Earth's crust is divided into plates that move on the mantle",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Rock Cycle",
                          "Rocks transform between igneous, sedimentary, and metamorphic types",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Water Cycle",
                          "Water evaporates, condenses into clouds, precipitates as rain/snow, and flows back to oceans",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Greenhouse Effect",
                          "Greenhouse gases trap heat in the atmosphere, warming the planet",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Earthquake Causes",
                          "Earthquakes are caused by sudden release of energy in the Earth's crust due to tectonic plate movement",
                          "yes", 0.99, s, "cause"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Volcano Formation",
                          "Volcanoes form at tectonic plate boundaries or hotspots where magma reaches the surface",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Erosion",
                          "Erosion is the wearing away of rock and soil by wind, water, ice, and gravity",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Fossil Formation",
                          "Fossils form when organisms are preserved in sedimentary rock over millions of years",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Magnetic Field",
                          "Earth's magnetic field is generated by the liquid iron outer core",
                          "yes", 0.99, s, "cause"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Atmosphere Layers",
                          "Earth's atmosphere has 5 layers: troposphere, stratosphere, mesosphere, thermosphere, exosphere",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Ocean Currents",
                          "Ocean currents distribute heat around the globe via thermohaline circulation",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Tides",
                          "Tides are caused mainly by the gravitational pull of the Moon and Sun",
                          "yes", 0.99, s, "cause"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Weathering",
                          "Weathering breaks down rocks through physical, chemical, and biological processes",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Mineral Classification",
                          "Minerals are classified by composition: silicates, carbonates, oxides, sulfides, native elements",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Soil Formation",
                          "Soil forms from weathered rock mixed with organic matter over long periods",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Continental Drift",
                          "Continents move on tectonic plates at rates of centimeters per year",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Glacial Periods",
                          "Earth has experienced multiple ice ages where glaciers covered large areas",
                          "yes", 0.99, s, "event"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Sedimentary Rock",
                          "Sedimentary rocks form from compressed layers of sediment and often contain fossils",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Igneous Rock",
                          "Igneous rocks form from cooled magma or lava",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"earth_{len(self._entries)}", d, "Metamorphic Rock",
                          "Metamorphic rocks form when existing rocks are transformed by heat and pressure",
                          "yes", 0.99, s, "classification"),
        ])

    # ══════════════════════════════════════════════════════════════
    # ASTRONOMY — NASA, ESA
    # ══════════════════════════════════════════════════════════════

    def _load_astronomy(self) -> None:
        d = "astronomy"
        s = "NASA / ESA"

        self._entries.extend([
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Solar System",
                          "The solar system has 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
                          "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune", 0.99, s, "structure"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Star Formation",
                          "Stars form from clouds of gas and dust (nebulae) that collapse under gravity",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Star Life Cycle",
                          "Stars go through stages: nebula, protostar, main sequence, red giant, white dwarf/neutron star/black hole",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Black Hole",
                          "A black hole is a region of space where gravity is so strong that nothing can escape",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Light Year",
                          "A light year is the distance light travels in one year (about 9.46 trillion km)",
                          "9.46 trillion km", 0.99, s, "definition"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Milky Way",
                          "Our solar system is in the Milky Way galaxy, which contains 100-400 billion stars",
                          "100-400 billion", 0.95, s, "fact"),
        ])

        # Planets
        self._entries.extend([
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Mercury",
                          "Mercury is the smallest planet and closest to the Sun; no atmosphere, extreme temperatures",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Venus",
                          "Venus is similar in size to Earth but has thick CO2 atmosphere and surface temperature of 465°C",
                          "465", 0.99, s, "fact"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Mars",
                          "Mars has the largest volcano (Olympus Mons) and canyon (Valles Marineris) in the solar system",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Jupiter",
                          "Jupiter is the largest planet with at least 95 moons and the Great Red Spot storm",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Saturn",
                          "Saturn has prominent rings made of ice and rock particles",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Uranus",
                          "Uranus rotates on its side and has a blue-green color from methane in its atmosphere",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Neptune",
                          "Neptune is the windiest planet with speeds up to 2,100 km/h",
                          "2100", 0.99, s, "fact"),
        ])

        # Cosmology
        self._entries.extend([
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Big Bang Theory",
                          "The universe began as a singularity approximately 13.8 billion years ago",
                          "13.8 billion years", 0.95, s, "theory"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Cosmic Microwave Background",
                          "CMB is radiation left over from the Big Bang, filling all of space",
                          "yes", 0.99, s, "phenomenon"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Dark Matter",
                          "Dark matter makes up about 27% of the universe and doesn't emit light",
                          "27%", 0.95, s, "concept"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Dark Energy",
                          "Dark energy makes up about 68% of the universe and drives accelerated expansion",
                          "68%", 0.95, s, "concept"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Hubble's Law",
                          "Galaxies are moving away from us, and their speed is proportional to their distance",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Redshift",
                          "Light from objects moving away is shifted toward longer (redder) wavelengths",
                          "yes", 0.99, s, "phenomenon"),
        ])

        # Stars and celestial objects
        self._entries.extend([
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Sun",
                          "The Sun is a G-type main sequence star, 4.6 billion years old, mostly hydrogen and helium",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Neutron Star",
                          "A neutron star is extremely dense; a teaspoon weighs about 6 billion tons",
                          "6 billion tons", 0.99, s, "fact"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Pulsar",
                          "A pulsar is a rotating neutron star that emits beams of electromagnetic radiation",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Supernova",
                          "A supernova is an explosion of a massive star at the end of its life",
                          "yes", 0.99, s, "event"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Asteroid Belt",
                          "The asteroid belt lies between Mars and Jupiter, containing millions of rocky objects",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Comets",
                          "Comets are icy bodies that develop tails of gas and dust when near the Sun",
                          "yes", 0.99, s, "entity"),
        ])

        # Moons
        self._entries.extend([
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Earth's Moon",
                          "The Moon is Earth's only natural satellite, about 384,400 km away",
                          "384400", 0.99, s, "fact"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Europa",
                          "Europa (Jupiter's moon) has a subsurface ocean that may harbor life",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"astro_{len(self._entries)}", d, "Titan",
                          "Titan (Saturn's moon) has a thick atmosphere and liquid methane lakes",
                          "yes", 0.99, s, "entity"),
        ])

    # ══════════════════════════════════════════════════════════════
    # MATHEMATICS — Khan Academy Math
    # ══════════════════════════════════════════════════════════════

    def _load_mathematics(self) -> None:
        d = "mathematics"
        s = "Khan Academy Mathematics"

        self._entries.extend([
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Pythagorean Theorem",
                          "In a right triangle, a² + b² = c²",
                          "a² + b² = c²", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Quadratic Formula",
                          "x = (-b ± √(b²-4ac)) / 2a",
                          "x = (-b ± √(b²-4ac)) / 2a", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Area of Circle",
                          "A = πr²",
                          "A = πr²", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Circumference of Circle",
                          "C = 2πr",
                          "C = 2πr", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Volume of Sphere",
                          "V = (4/3)πr³",
                          "V = (4/3)πr³", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Slope Formula",
                          "slope = (y₂ - y₁) / (x₂ - x₁)",
                          "slope = (y₂ - y₁) / (x₂ - x₁)", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Distance Formula",
                          "d = √((x₂-x₁)² + (y₂-y₁)²)",
                          "d = √((x₂-x₁)² + (y₂-y₁)²)", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Compound Interest",
                          "A = P(1 + r/n)^(nt)",
                          "A = P(1 + r/n)^(nt)", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Exponent Rules",
                          "a^m * a^n = a^(m+n), (a^m)^n = a^(mn), a^0 = 1",
                          "yes", 0.99, s, "rules"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Logarithm Properties",
                          "log(ab) = log(a) + log(b), log(a/b) = log(a) - log(b), log(a^n) = n*log(a)",
                          "yes", 0.99, s, "rules"),
        ])

        # Calculus
        self._entries.extend([
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Derivative",
                          "The derivative measures the rate of change of a function with respect to a variable",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Integral",
                          "The integral computes the area under a curve; it is the reverse of differentiation",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Fundamental Theorem of Calculus",
                          "Differentiation and integration are inverse operations",
                          "yes", 0.99, s, "theorem"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Chain Rule",
                          "d/dx[f(g(x))] = f'(g(x)) * g'(x)",
                          "d/dx[f(g(x))] = f'(g(x)) * g'(x)", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Limits",
                          "A limit describes the value a function approaches as the input approaches a point",
                          "yes", 0.99, s, "definition"),
        ])

        # Statistics
        self._entries.extend([
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Mean",
                          "Mean = sum of values / number of values",
                          "yes", 0.99, s, "formula"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Median",
                          "The median is the middle value when data is ordered",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Standard Deviation",
                          "Standard deviation measures how spread out data is from the mean",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Normal Distribution",
                          "The normal (bell) distribution has 68% within 1σ, 95% within 2σ, 99.7% within 3σ",
                          "yes", 0.99, s, "distribution"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Probability Rules",
                          "P(A or B) = P(A) + P(B) - P(A and B); P(A|B) = P(A and B) / P(B)",
                          "yes", 0.99, s, "rules"),
        ])

        # Linear Algebra
        self._entries.extend([
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Matrix Multiplication",
                          "Matrix multiplication: (AB)ij = sum of Aik * Bkj",
                          "yes", 0.99, s, "operation"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Determinant",
                          "The determinant of a matrix indicates if it is invertible (non-zero = invertible)",
                          "yes", 0.99, s, "property"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Eigenvalues",
                          "Eigenvalues satisfy Av = λv where A is a matrix, v is eigenvector, λ is eigenvalue",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Vector Operations",
                          "Vector addition, scalar multiplication, dot product (a·b = |a||b|cos θ)",
                          "yes", 0.99, s, "operations"),
        ])

        # Number Theory
        self._entries.extend([
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Prime Numbers",
                          "A prime number has exactly two factors: 1 and itself. First primes: 2, 3, 5, 7, 11, 13",
                          "2, 3, 5, 7, 11, 13", 0.99, s, "definition"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Fibonacci Sequence",
                          "Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34... each number is the sum of the two before it",
                          "yes", 0.99, s, "sequence"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Pi",
                          "Pi (π) ≈ 3.14159 is the ratio of a circle's circumference to its diameter",
                          "3.14159", 0.99, s, "constant"),
            KnowledgeEntry(f"math_{len(self._entries)}", d, "Euler's Number",
                          "Euler's number e ≈ 2.71828 is the base of natural logarithms",
                          "2.71828", 0.99, s, "constant"),
        ])

    # ══════════════════════════════════════════════════════════════
    # HISTORY — Khan Academy History
    # ══════════════════════════════════════════════════════════════

    def _load_history(self) -> None:
        d = "history"
        s = "Khan Academy History"

        self._entries.extend([
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Agricultural Revolution",
                          "The Agricultural Revolution began around 10,000 BCE, when humans started farming",
                          "10000 BCE", 0.95, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Writing Invented",
                          "Writing was invented around 3400 BCE in Mesopotamia",
                          "3400 BCE", 0.95, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Roman Empire Fall",
                          "The Western Roman Empire fell in 476 CE",
                          "476 CE", 0.95, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Renaissance",
                          "The Renaissance was a cultural movement from the 14th to 17th century, starting in Italy",
                          "14th-17th century", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Industrial Revolution",
                          "The Industrial Revolution began in Britain in the late 18th century",
                          "late 18th century", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "World War I",
                          "World War I was from 1914 to 1918",
                          "1914-1918", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "World War II",
                          "World War II was from 1939 to 1945",
                          "1939-1945", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Cold War",
                          "The Cold War was a geopolitical tension between the US and Soviet Union from 1947 to 1991",
                          "1947-1991", 0.99, s, "event"),
        ])

        # Ancient Civilizations
        self._entries.extend([
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Ancient Egypt",
                          "Ancient Egypt (3100-30 BCE) built pyramids, developed hieroglyphics, and mummification",
                          "3100-30 BCE", 0.99, s, "civilization"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Ancient Greece",
                          "Ancient Greece (800-146 BCE) gave democracy, philosophy, theater, and the Olympics",
                          "800-146 BCE", 0.99, s, "civilization"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Roman Republic",
                          "The Roman Republic (509-27 BCE) established republican government and Roman law",
                          "509-27 BCE", 0.99, s, "civilization"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Mesopotamia",
                          "Mesopotamia (3500-539 BCE) developed writing (cuneiform), mathematics, and irrigation",
                          "3500-539 BCE", 0.99, s, "civilization"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Indus Valley",
                          "The Indus Valley civilization (3300-1300 BCE) had advanced urban planning and drainage",
                          "3300-1300 BCE", 0.99, s, "civilization"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Ancient China",
                          "Ancient China developed paper, gunpowder, compass, and printing",
                          "yes", 0.99, s, "civilization"),
        ])

        # Medieval and Early Modern
        self._entries.extend([
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Middle Ages",
                          "The Middle Ages (5th-15th century) in Europe featured feudalism, castles, and the Black Death",
                          "5th-15th century", 0.99, s, "period"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Black Death",
                          "The Black Death (1347-1351) killed 25-50 million people in Europe",
                          "1347-1351", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Age of Exploration",
                          "The Age of Exploration (15th-17th century) saw European voyages to Americas, Asia, Africa",
                          "15th-17th century", 0.99, s, "period"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "American Revolution",
                          "The American Revolution (1775-1783) established the United States as independent",
                          "1775-1783", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "French Revolution",
                          "The French Revolution (1789-1799) overthrew the monarchy and established a republic",
                          "1789-1799", 0.99, s, "event"),
        ])

        # Modern History
        self._entries.extend([
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Great Depression",
                          "The Great Depression (1929-1939) was the worst economic downturn in modern history",
                          "1929-1939", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Moon Landing",
                          "Apollo 11 landed the first humans on the Moon on July 20, 1969",
                          "1969", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "Fall of Berlin Wall",
                          "The Berlin Wall fell on November 9, 1989, symbolizing the end of the Cold War",
                          "1989", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "World Wide Web",
                          "Tim Berners-Lee invented the World Wide Web in 1989 at CERN",
                          "1989", 0.99, s, "event"),
            KnowledgeEntry(f"hist_{len(self._entries)}", d, "September 11 Attacks",
                          "The 9/11 attacks (2001) involved hijacked planes hitting the World Trade Center and Pentagon",
                          "2001", 0.99, s, "event"),
        ])

    # ══════════════════════════════════════════════════════════════
    # GEOGRAPHY — National Geographic
    # ══════════════════════════════════════════════════════════════

    def _load_geography(self) -> None:
        d = "geography"
        s = "National Geographic / World Atlas"

        self._entries.extend([
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Continents",
                          "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, Australia",
                          "7", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Largest Country",
                          "Russia is the largest country by area (17.1 million km²)",
                          "Russia", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Most Populous Country",
                          "India is the most populous country with over 1.4 billion people",
                          "India", 0.95, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Largest Ocean",
                          "The Pacific Ocean is the largest ocean (165.25 million km²)",
                          "Pacific", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Longest River",
                          "The Nile River is approximately 6,650 km long",
                          "Nile", 0.95, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Highest Mountain",
                          "Mount Everest is 8,849 meters above sea level",
                          "8849", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Deepest Point",
                          "The Mariana Trench is about 11,034 meters deep",
                          "11034", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Climate Zones",
                          "Earth has 5 climate zones: tropical, dry, temperate, continental, and polar",
                          "5", 0.99, s, "classification"),
        ])

        # Countries and Capitals
        self._entries.extend([
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "France Capital",
                          "The capital of France is Paris",
                          "Paris", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Japan Capital",
                          "The capital of Japan is Tokyo",
                          "Tokyo", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Germany Capital",
                          "The capital of Germany is Berlin",
                          "Berlin", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "United Kingdom Capital",
                          "The capital of the United Kingdom is London",
                          "London", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "China Capital",
                          "The capital of China is Beijing",
                          "Beijing", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "India Capital",
                          "The capital of India is New Delhi",
                          "New Delhi", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Brazil Capital",
                          "The capital of Brazil is Brasilia",
                          "Brasilia", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Australia Capital",
                          "The capital of Australia is Canberra",
                          "Canberra", 0.95, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Canada Capital",
                          "The capital of Canada is Ottawa",
                          "Ottawa", 0.95, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Egypt Capital",
                          "The capital of Egypt is Cairo",
                          "Cairo", 0.99, s, "fact"),
        ])

        # Physical Geography
        self._entries.extend([
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Amazon River",
                          "The Amazon River is the largest river by water volume, flowing through South America",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Sahara Desert",
                          "The Sahara is the largest hot desert, covering about 9.2 million km² in Africa",
                          "9.2 million km²", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Great Barrier Reef",
                          "The Great Barrier Reef is the largest coral reef system, off the coast of Australia",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Ganges River",
                          "The Ganges is a 2,525 km river sacred in Hinduism, flowing through India and Bangladesh",
                          "2525", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Himalayas",
                          "The Himalayas are the highest mountain range, containing Mount Everest",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"geo_{len(self._entries)}", d, "Danube River",
                          "The Danube is the second longest river in Europe, flowing through 10 countries",
                          "10", 0.99, s, "fact"),
        ])

    # ══════════════════════════════════════════════════════════════
    # TECHNOLOGY — Various sources
    # ══════════════════════════════════════════════════════════════

    def _load_technology(self) -> None:
        d = "technology"
        s = "Various technical sources"

        self._entries.extend([
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Computer Architecture",
                          "A computer has CPU, memory (RAM), storage, and input/output devices",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Binary System",
                          "Computers use binary (base-2) with 0s and 1s to represent data",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Internet",
                          "The internet is a global network of interconnected computers using TCP/IP",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Programming Languages",
                          "Programming languages include Python, JavaScript, Java, C++, Rust, Go",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Machine Learning",
                          "Machine learning is a subset of AI where systems learn patterns from data",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Neural Networks",
                          "Neural networks are computing systems inspired by biological neural networks",
                          "yes", 0.99, s, "definition"),
        ])

        # Expanded Technology
        self._entries.extend([
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "CPU",
                          "A CPU executes instructions through fetch-decode-execute cycle",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "RAM",
                          "RAM is volatile memory that stores data temporarily for quick access",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "SSD vs HDD",
                          "SSDs use flash memory and are faster than HDDs which use spinning disks",
                          "yes", 0.99, s, "comparison"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "GPU",
                          "GPUs are specialized processors for parallel computation, used in graphics and AI",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Cloud Computing",
                          "Cloud computing provides on-demand computing resources over the internet",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Blockchain",
                          "Blockchain is a distributed ledger technology that records transactions immutably",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Cybersecurity",
                          "Cybersecurity protects systems and networks from digital attacks",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "API",
                          "An API (Application Programming Interface) allows software to communicate with other software",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "DNS",
                          "DNS translates domain names (google.com) to IP addresses",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Git",
                          "Git is a distributed version control system for tracking code changes",
                          "yes", 0.99, s, "tool"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Linux",
                          "Linux is an open-source operating system kernel created by Linus Torvalds in 1991",
                          "1991", 0.99, s, "entity"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Python",
                          "Python is a high-level programming language known for readability and versatility",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "JavaScript",
                          "JavaScript is the primary language for web development, running in browsers",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Moore's Law",
                          "Moore's Law: transistor count doubles approximately every two years",
                          "yes", 0.99, s, "law"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "5G Networks",
                          "5G is the fifth generation of mobile networks with higher speed and lower latency",
                          "yes", 0.99, s, "technology"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "IoT",
                          "IoT (Internet of Things) connects everyday devices to the internet",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Virtual Reality",
                          "VR creates immersive computer-generated environments using headsets",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Augmented Reality",
                          "AR overlays digital information onto the real world through cameras or glasses",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Quantum Computing",
                          "Quantum computers use qubits that can be 0 and 1 simultaneously (superposition)",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"tech_{len(self._entries)}", d, "Natural Language Processing",
                          "NLP enables computers to understand, interpret, and generate human language",
                          "yes", 0.99, s, "definition"),
        ])

    # ══════════════════════════════════════════════════════════════
    # HEALTH — WHO, NIH
    # ══════════════════════════════════════════════════════════════

    def _load_health(self) -> None:
        d = "health"
        s = "WHO / NIH"

        self._entries.extend([
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Vaccination",
                          "Vaccines train the immune system to recognize and fight specific pathogens",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Sleep Requirements",
                          "Adults need 7-9 hours of sleep per night for optimal health",
                          "7-9", 0.95, s, "fact"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Water Intake",
                          "Adults should drink about 2-3 liters of water per day",
                          "2-3 liters", 0.90, s, "fact"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "BMI",
                          "BMI = weight(kg) / height(m)². Normal range is 18.5-24.9",
                          "18.5-24.9", 0.99, s, "formula"),
        ])

        # Diseases
        self._entries.extend([
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Diabetes",
                          "Diabetes is a condition where the body cannot properly regulate blood sugar levels",
                          "yes", 0.99, s, "disease"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Hypertension",
                          "Hypertension (high blood pressure) increases risk of heart disease and stroke",
                          "yes", 0.99, s, "disease"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Asthma",
                          "Asthma is a chronic lung condition causing inflammation and narrowing of airways",
                          "yes", 0.99, s, "disease"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Cancer",
                          "Cancer is uncontrolled cell growth that can spread to other parts of the body",
                          "yes", 0.99, s, "disease"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Heart Disease",
                          "Heart disease is the leading cause of death worldwide",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Antibiotics",
                          "Antibiotics fight bacterial infections; they don't work against viruses",
                          "yes", 0.99, s, "fact"),
        ])

        # Nutrition
        self._entries.extend([
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Vitamins",
                          "Essential vitamins: A, B, C, D, E, K. Each serves specific functions in the body",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Macronutrients",
                          "Macronutrients: carbohydrates, proteins, and fats provide energy",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Calories",
                          "A calorie is a unit of energy; adults need approximately 2000-2500 calories per day",
                          "2000-2500", 0.95, s, "fact"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Iron",
                          "Iron is essential for hemoglobin in red blood cells; deficiency causes anemia",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Calcium",
                          "Calcium is essential for strong bones and teeth; dairy is a good source",
                          "yes", 0.99, s, "fact"),
        ])

        # Medicine
        self._entries.extend([
            KnowledgeEntry(f"health_{len(self._entries)}", d, "X-ray",
                          "X-rays use electromagnetic radiation to see inside the body",
                          "yes", 0.99, s, "technology"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "MRI",
                          "MRI uses magnetic fields and radio waves to create detailed body images",
                          "yes", 0.99, s, "technology"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "Surgery Types",
                          "Surgery types: open surgery, minimally invasive, laparoscopic, robotic",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "First Aid",
                          "First aid basics: assess scene safety, call emergency services, control bleeding, perform CPR",
                          "yes", 0.99, s, "procedure"),
            KnowledgeEntry(f"health_{len(self._entries)}", d, "CPR",
                          "CPR: 30 chest compressions followed by 2 rescue breaths at 100-120 compressions/minute",
                          "100-120", 0.99, s, "procedure"),
        ])

    # ══════════════════════════════════════════════════════════════
    # ECONOMICS — Khan Academy Economics
    # ══════════════════════════════════════════════════════════════

    def _load_economics(self) -> None:
        d = "economics"
        s = "Khan Academy Economics"

        self._entries.extend([
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Supply and Demand",
                          "When supply increases, price decreases. When demand increases, price increases.",
                          "yes", 0.95, s, "law"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Inflation",
                          "Inflation is the rate at which the general level of prices for goods and services rises",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "GDP",
                          "GDP is the total value of all goods and services produced in a country",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Opportunity Cost",
                          "Opportunity cost is the value of the next best alternative forgone",
                          "yes", 0.99, s, "definition"),
        ])

        # Expanded Economics
        self._entries.extend([
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Interest Rates",
                          "Interest rates are the cost of borrowing money; central banks set base rates",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Unemployment",
                          "Unemployment rate is the percentage of the labor force without work",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Trade Balance",
                          "Trade balance = exports minus imports; surplus means exports exceed imports",
                          "yes", 0.99, s, "formula"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Monetary Policy",
                          "Monetary policy is conducted by central banks to control money supply and interest rates",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Fiscal Policy",
                          "Fiscal policy involves government spending and taxation to influence the economy",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Market Economy",
                          "A market economy is driven by supply and demand with minimal government intervention",
                          "yes", 0.99, s, "system"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Command Economy",
                          "In a command economy, the government controls production and prices",
                          "yes", 0.99, s, "system"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Elasticity",
                          "Price elasticity measures how quantity demanded changes with price",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Comparative Advantage",
                          "Countries should specialize in producing goods where they have lower opportunity cost",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Stock Market",
                          "The stock market is where shares of publicly traded companies are bought and sold",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Compound Interest",
                          "Compound interest earns interest on interest, growing wealth exponentially",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Recession",
                          "A recession is two consecutive quarters of declining GDP",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Poverty Line",
                          "The poverty line is the minimum income level needed for basic necessities",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Human Development Index",
                          "HDI measures development using life expectancy, education, and income per capita",
                          "yes", 0.99, s, "measure"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Currency",
                          "Major currencies: USD, EUR, JPY, GBP, CNY",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Central Banks",
                          "Central banks (Fed, ECB, BOJ) manage monetary policy and regulate banks",
                          "yes", 0.99, s, "institution"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Gini Coefficient",
                          "The Gini coefficient measures income inequality from 0 (equal) to 1 (unequal)",
                          "0 to 1", 0.99, s, "measure"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "PPP",
                          "Purchasing Power Parity adjusts GDP for differences in price levels between countries",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"econ_{len(self._entries)}", d, "Cryptocurrency",
                          "Cryptocurrency is digital currency using cryptography; Bitcoin was the first (2009)",
                          "2009", 0.99, s, "concept"),
        ])

    # ══════════════════════════════════════════════════════════════
    # ECOLOGY — Khan Academy, WWF
    # ══════════════════════════════════════════════════════════════

    def _load_ecology(self) -> None:
        d = "ecology"
        s = "Khan Academy Ecology / WWF"

        self._entries.extend([
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Carbon Cycle",
                          "Carbon cycles through atmosphere, biosphere, hydrosphere, and geosphere",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Nitrogen Cycle",
                          "Nitrogen is converted between N2 and usable forms by bacteria",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Climate Change",
                          "Human activities increase greenhouse gases, causing global warming",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Deforestation",
                          "Deforestation removes trees, reducing carbon absorption and habitat",
                          "yes", 0.99, s, "process"),
        ])

        # Expanded ecology
        self._entries.extend([
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Trophic Levels",
                          "Food chains have trophic levels: producers, primary consumers, secondary consumers, decomposers",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Biomes",
                          "Major biomes include tropical rainforest, desert, tundra, grassland, taiga, temperate forest",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Population Dynamics",
                          "Population size is affected by birth rate, death rate, immigration, and emigration",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Keystone Species",
                          "Keystone species have disproportionately large effects on their ecosystem",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Succession",
                          "Ecological succession is the gradual process of ecosystem change and development",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Ozone Layer",
                          "The ozone layer in the stratosphere absorbs UV radiation from the Sun",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Ocean Acidification",
                          "CO2 dissolves in seawater forming carbonic acid, lowering pH and harming marine life",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Carbon Sequestration",
                          "Carbon sequestration is the capture and storage of atmospheric CO2",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Wetland Ecosystems",
                          "Wetlands filter water, prevent flooding, and support high biodiversity",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Coral Reefs",
                          "Coral reefs support 25% of marine species despite covering less than 1% of the ocean floor",
                          "25%", 0.99, s, "fact"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Bioaccumulation",
                          "Toxins accumulate in organisms as they move up the food chain",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Invasive Species",
                          "Invasive species are non-native organisms that cause ecological or economic harm",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Endangered Species",
                          "Endangered species are at risk of extinction; conservation efforts aim to protect them",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Food Web",
                          "Food webs are complex networks of interconnected food chains in an ecosystem",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Carrying Capacity",
                          "Carrying capacity is the maximum population size an environment can sustain",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Phosphorus Cycle",
                          "The phosphorus cycle moves phosphorus through rocks, water, soil, and organisms",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Water Pollution",
                          "Water pollution comes from industrial waste, agricultural runoff, and sewage",
                          "yes", 0.99, s, "cause"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Renewable Energy",
                          "Renewable energy sources: solar, wind, hydroelectric, geothermal, biomass",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"eco_{len(self._entries)}", d, "Greenhouse Gases",
                          "Major greenhouse gases: CO2, methane, nitrous oxide, water vapor, fluorinated gases",
                          "yes", 0.99, s, "classification"),
        ])

    # ══════════════════════════════════════════════════════════════
    # LITERATURE — Various sources
    # ══════════════════════════════════════════════════════════════

    def _load_literature(self) -> None:
        d = "literature"
        s = "Various literary sources"

        self._entries.extend([
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Shakespeare",
                          "William Shakespeare (1564-1616) wrote 37 plays including Hamlet, Macbeth, Romeo and Juliet",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Homer's Odyssey",
                          "The Odyssey by Homer is an ancient Greek epic about Odysseus's journey home from Troy",
                          "yes", 0.99, s, "work"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Literary Genres",
                          "Major genres include fiction, non-fiction, poetry, drama, and essay",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Plot Structure",
                          "Classic plot structure: exposition, rising action, climax, falling action, resolution",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Narrative Point of View",
                          "Narrative can be first person (I), second person (you), or third person (he/she/they)",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Figurative Language",
                          "Figurative language includes metaphor, simile, personification, hyperbole, irony",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Themes in Literature",
                          "Common themes: love, death, good vs evil, coming of age, power, identity",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Dickens",
                          "Charles Dickens (1812-1870) wrote Oliver Twist, Great Expectations, A Tale of Two Cities",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Jane Austen",
                          "Jane Austen (1775-1817) wrote Pride and Prejudice, Sense and Sensibility",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Tolkien",
                          "J.R.R. Tolkien (1892-1973) wrote The Lord of the Rings and The Hobbit",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Dystopian Fiction",
                          "Dystopian fiction imagines oppressive societies (1984, Brave New World, Fahrenheit 451)",
                          "yes", 0.99, s, "genre"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Poetry Forms",
                          "Poetry forms include sonnet, haiku, free verse, limerick, ode, elegy",
                          "yes", 0.99, s, "classification"),
        ])

        # More Literature
        self._entries.extend([
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Emily Dickinson",
                          "Emily Dickinson (1830-1886) was an American poet known for unconventional punctuation and themes of death",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Mark Twain",
                          "Mark Twain (1835-1910) wrote Adventures of Huckleberry Finn and Tom Sawyer",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "George Orwell",
                          "George Orwell (1903-1950) wrote 1984 and Animal Farm",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Gabriel Garcia Marquez",
                          "Garcia Marquez (1927-2014) wrote One Hundred Years of Solitude, pioneer of magical realism",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Chinua Achebe",
                          "Chinua Achebe (1930-2013) wrote Things Fall Apart, the most widely read African novel",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Fyodor Dostoevsky",
                          "Dostoevsky (1821-1881) wrote Crime and Punishment and The Brothers Karamazov",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Leo Tolstoy",
                          "Tolstoy (1828-1910) wrote War and Peace and Anna Karenina",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Harper Lee",
                          "Harper Lee (1926-2016) wrote To Kill a Mockingbird",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Literary Devices",
                          "Literary devices include symbolism, irony, foreshadowing, allusion, allegory, motif",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Mythology",
                          "Major mythologies: Greek, Norse, Egyptian, Hindu, Chinese, Roman",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Greek Myths",
                          "Greek myths include Prometheus, Icarus, Odysseus, Hercules, Pandora's Box",
                          "yes", 0.99, s, "topic"),
            KnowledgeEntry(f"lit_{len(self._entries)}", d, "Norse Mythology",
                          "Norse mythology features Odin, Thor, Loki, Valhalla, Ragnarok, Yggdrasil",
                          "yes", 0.99, s, "topic"),
        ])

    # ══════════════════════════════════════════════════════════════
    # PHILOSOPHY — Stanford Encyclopedia of Philosophy
    # ══════════════════════════════════════════════════════════════

    def _load_philosophy(self) -> None:
        d = "philosophy"
        s = "Stanford Encyclopedia of Philosophy"

        self._entries.extend([
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Socrates",
                          "Socrates (470-399 BCE) developed the Socratic method of questioning",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Plato",
                          "Plato (428-348 BCE) wrote The Republic and founded the Academy",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Aristotle",
                          "Aristotle (384-322 BCE) contributed to logic, ethics, metaphysics, and biology",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Ethics",
                          "Ethics is the branch of philosophy dealing with right and wrong conduct",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Utilitarianism",
                          "Utilitarianism states the best action is the one that maximizes overall happiness",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Kant's Categorical Imperative",
                          "Act only according to rules you would want to become universal laws",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Logic",
                          "Logic is the study of valid reasoning and argumentation",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Epistemology",
                          "Epistemology is the study of knowledge: what we know and how we know it",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Metaphysics",
                          "Metaphysics studies the fundamental nature of reality and existence",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Stoicism",
                          "Stoicism teaches virtue, reason, and acceptance of what cannot be controlled",
                          "yes", 0.99, s, "philosophy"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Existentialism",
                          "Existentialism emphasizes individual existence, freedom, and choice",
                          "yes", 0.99, s, "philosophy"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Philosophy of Mind",
                          "Philosophy of mind studies consciousness, mental states, and the mind-body problem",
                          "yes", 0.99, s, "branch"),
        ])

        # More Philosophy
        self._entries.extend([
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "John Locke",
                          "John Locke (1632-1704) advocated natural rights: life, liberty, and property",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Thomas Hobbes",
                          "Hobbes (1588-1679) wrote Leviathan and argued for strong government (social contract)",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Jean-Jacques Rousseau",
                          "Rousseau (1712-1778) wrote The Social Contract, arguing government derives from consent",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Friedrich Nietzsche",
                          "Nietzsche (1844-1900) wrote Thus Spoke Zarathustra, declared God is dead, coined will to power",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Confucius",
                          "Confucius (551-479 BCE) taught ethics, family values, and social harmony in China",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Immanuel Kant",
                          "Kant (1724-1804) wrote Critique of Pure Reason, establishing modern epistemology",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Utilitarianism Origin",
                          "Utilitarianism was developed by Jeremy Bentham and John Stuart Mill",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Descartes",
                          "Descartes (1596-1650) wrote I think therefore I am, founding modern philosophy",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Free Will",
                          "The free will debate: determinism vs libertarianism vs compatibilism",
                          "yes", 0.99, s, "topic"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Trolley Problem",
                          "The trolley problem explores ethics: should you divert a trolley to save five at the cost of one?",
                          "yes", 0.99, s, "topic"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Social Contract Theory",
                          "Social contract theory: individuals give up some freedoms to a government for protection",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"phil_{len(self._entries)}", d, "Eastern Philosophy",
                          "Eastern philosophy includes Confucianism, Taoism, Buddhism, and Hinduism",
                          "yes", 0.99, s, "classification"),
        ])

    # ══════════════════════════════════════════════════════════════
    # COMPUTER SCIENCE — ACM, MIT OpenCourseWare
    # ══════════════════════════════════════════════════════════════

    def _load_computer_science(self) -> None:
        d = "computer_science"
        s = "ACM / MIT OpenCourseWare"

        self._entries.extend([
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Algorithms",
                          "Algorithms are step-by-step procedures for solving problems",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Big O Notation",
                          "Big O describes algorithm complexity: O(1) constant, O(log n) logarithmic, O(n) linear, O(n^2) quadratic",
                          "yes", 0.99, s, "notation"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Data Structures",
                          "Core data structures: arrays, linked lists, stacks, queues, trees, hash tables, graphs",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Sorting Algorithms",
                          "Common sorts: bubble O(n^2), merge O(n log n), quick O(n log n avg), heap O(n log n)",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Recursion",
                          "Recursion is a technique where a function calls itself to solve smaller subproblems",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "TCP/IP",
                          "TCP/IP is the fundamental protocol suite of the internet",
                          "yes", 0.99, s, "protocol"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "HTTP",
                          "HTTP is the protocol for web communication; methods include GET, POST, PUT, DELETE",
                          "yes", 0.99, s, "protocol"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Databases",
                          "Databases store structured data; SQL databases use relational tables, NoSQL uses documents/key-value",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Operating Systems",
                          "An OS manages hardware resources and provides services for programs",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Compilers",
                          "Compilers translate source code into machine code in stages: lexer, parser, optimizer, code gen",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Cryptography",
                          "Cryptography secures communication through encryption (symmetric and asymmetric)",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Machine Learning Types",
                          "ML types: supervised, unsupervised, reinforcement, semi-supervised",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Neural Network Layers",
                          "Neural networks have input, hidden, and output layers with weighted connections",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Binary Search",
                          "Binary search finds elements in sorted arrays in O(log n) time",
                          "yes", 0.99, s, "algorithm"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Graph Theory",
                          "Graph theory studies networks of nodes and edges; BFS and DFS traverse graphs",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "String Matching",
                          "String matching algorithms: brute force O(nm), KMP O(n+m), Rabin-Karp O(n+m)",
                          "yes", 0.99, s, "classification"),
        ])

        # More CS
        self._entries.extend([
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Object-Oriented Programming",
                          "OOP principles: encapsulation, inheritance, polymorphism, abstraction",
                          "yes", 0.99, s, "principles"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Functional Programming",
                          "Functional programming uses pure functions, immutability, and avoids side effects",
                          "yes", 0.99, s, "paradigm"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Design Patterns",
                          "Common design patterns: Singleton, Observer, Factory, Strategy, MVC",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Version Control",
                          "Version control tracks changes; Git uses branches, commits, merges, and pull requests",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "API Design",
                          "REST APIs use HTTP methods (GET, POST, PUT, DELETE) and status codes (200, 404, 500)",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Distributed Systems",
                          "Distributed systems face challenges: consistency, availability, partition tolerance (CAP theorem)",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Hashing",
                          "Hashing maps data to fixed-size values; used in hash tables and cryptography",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Dynamic Programming",
                          "Dynamic programming solves problems by breaking them into overlapping subproblems",
                          "yes", 0.99, s, "technique"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Greedy Algorithms",
                          "Greedy algorithms make locally optimal choices at each step",
                          "yes", 0.99, s, "technique"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Computer Networks",
                          "Network layers: physical, data link, network, transport, application (OSI model)",
                          "yes", 0.99, s, "structure"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "SQL Basics",
                          "SQL commands: SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, JOIN",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "NoSQL Databases",
                          "NoSQL types: document (MongoDB), key-value (Redis), column (Cassandra), graph (Neo4j)",
                          "yes", 0.99, s, "classification"),
        ])

        # Even more CS
        self._entries.extend([
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Web Development",
                          "Web dev: HTML (structure), CSS (style), JavaScript (behavior), frameworks like React/Vue/Angular",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Mobile Development",
                          "Mobile: native (Swift/Kotlin), cross-platform (Flutter/React Native), hybrid (Ionic)",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "DevOps",
                          "DevOps combines development and operations: CI/CD, containers, monitoring",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Docker",
                          "Docker packages applications into containers for consistent deployment",
                          "yes", 0.99, s, "tool"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Kubernetes",
                          "Kubernetes orchestrates container deployment, scaling, and management",
                          "yes", 0.99, s, "tool"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Machine Learning Models",
                          "Common ML models: linear regression, decision trees, random forests, SVMs, neural networks",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Deep Learning",
                          "Deep learning uses multi-layer neural networks: CNNs for images, RNNs/Transformers for sequences",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Computer Vision",
                          "Computer vision enables machines to interpret images and video",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Speech Recognition",
                          "Speech recognition converts audio to text using acoustic and language models",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "LLMs",
                          "Large Language Models (GPT, Claude, Gemini) are trained on vast text corpora",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Transformer Architecture",
                          "Transformers use self-attention to process sequences in parallel, revolutionizing NLP",
                          "yes", 0.99, s, "architecture"),
            KnowledgeEntry(f"cs_{len(self._entries)}", d, "Reinforcement Learning",
                          "Reinforcement learning trains agents through rewards and penalties in an environment",
                          "yes", 0.99, s, "definition"),
        ])

    # ══════════════════════════════════════════════════════════════
    # MUSIC & ARTS — Various sources
    # ══════════════════════════════════════════════════════════════

    def _load_music_arts(self) -> None:
        d = "music_arts"
        s = "Various sources"

        self._entries.extend([
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Music Theory Basics",
                          "Music is organized in time with rhythm, melody, harmony, and dynamics",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Major Scales",
                          "A major scale follows the pattern: whole, whole, half, whole, whole, whole, half",
                          "yes", 0.99, s, "rule"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Musical Instruments",
                          "Instruments are classified: strings, woodwinds, brass, percussion, keyboards",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Color Theory",
                          "Primary colors (red, blue, yellow) combine to create secondary and tertiary colors",
                          "yes", 0.99, s, "theory"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Art Movements",
                          "Major movements: Renaissance, Baroque, Impressionism, Cubism, Surrealism, Pop Art",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Photography Basics",
                          "Photography uses aperture, shutter speed, and ISO to control exposure",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Da Vinci",
                          "Leonardo da Vinci (1452-1519) painted the Mona Lisa and The Last Supper",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Mozart",
                          "Wolfgang Amadeus Mozart (1756-1791) composed over 600 works including symphonies and operas",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Beethoven",
                          "Ludwig van Beethoven (1770-1827) composed 9 symphonies, despite becoming deaf",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Van Gogh",
                          "Vincent van Gogh (1853-1890) painted Starry Night and Sunflowers",
                          "yes", 0.99, s, "entity"),
        ])

        # More Music and Arts
        self._entries.extend([
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Picasso",
                          "Pablo Picasso (1881-1973) co-founded Cubism, painted Guernica and Les Demoiselles d'Avignon",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Michelangelo",
                          "Michelangelo (1475-1564) painted the Sistine Chapel ceiling and sculpted David",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Rembrandt",
                          "Rembrandt (1606-1669) was a Dutch master known for The Night Watch and self-portraits",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Bach",
                          "J.S. Bach (1685-1750) composed the Brandenburg Concertos and The Well-Tempered Clavier",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Jazz",
                          "Jazz originated in New Orleans in the early 20th century, featuring improvisation",
                          "yes", 0.99, s, "genre"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Classical Music Periods",
                          "Classical music periods: Baroque, Classical, Romantic, Modern",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Renaissance Art",
                          "Renaissance art (14th-17th century) emphasized realism, perspective, and humanism",
                          "yes", 0.99, s, "movement"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Impressionism",
                          "Impressionism (1860s-1880s) emphasized light and color, led by Monet and Renoir",
                          "yes", 0.99, s, "movement"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Film Basics",
                          "Film uses 24 frames per second; aspect ratios include 16:9 and 2.39:1",
                          "24", 0.99, s, "fact"),
            KnowledgeEntry(f"art_{len(self._entries)}", d, "Architecture Styles",
                          "Architecture styles: Gothic, Renaissance, Baroque, Neoclassical, Art Deco, Modernist",
                          "yes", 0.99, s, "classification"),
        ])

    # ══════════════════════════════════════════════════════════════
    # SOCIOLOGY — Various sources
    # ══════════════════════════════════════════════════════════════

    def _load_sociology(self) -> None:
        d = "sociology"
        s = "Various academic sources"

        self._entries.extend([
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Social Stratification",
                          "Societies are stratified by class, race, gender, and other social categories",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Culture",
                          "Culture includes shared beliefs, values, customs, and artifacts of a group",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Socialization",
                          "Socialization is the process of learning social norms and values",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Institutions",
                          "Social institutions: family, education, religion, government, economy, media",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Democracy",
                          "Democracy is a system of government where citizens exercise power by voting",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Globalization",
                          "Globalization is the increasing interconnectedness of economies, cultures, and populations",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Urbanization",
                          "Urbanization is the movement of populations from rural to urban areas",
                          "yes", 0.99, s, "process"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Population Growth",
                          "World population exceeded 8 billion in 2022",
                          "8 billion", 0.99, s, "fact"),
        ])

        # More Sociology
        self._entries.extend([
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Social Movements",
                          "Social movements organize for change: civil rights, feminist, environmental, labor",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Media Influence",
                          "Mass media shapes public opinion, culture, and political discourse",
                          "yes", 0.99, s, "fact"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Gender Roles",
                          "Gender roles are social norms about behaviors considered appropriate for men and women",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Race and Ethnicity",
                          "Race is a social construct; ethnicity refers to cultural heritage and shared identity",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Emile Durkheim",
                          "Durkheim (1858-1917) studied social facts, anomie, and collective consciousness",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Karl Marx",
                          "Marx (1818-1883) analyzed class struggle, capitalism, and historical materialism",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Max Weber",
                          "Weber (1864-1920) studied bureaucracy, authority types, and the Protestant ethic",
                          "yes", 0.99, s, "entity"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Social Capital",
                          "Social capital is the value of social networks and trust within a community",
                          "yes", 0.99, s, "definition"),
        ])

        # More Sociology
        self._entries.extend([
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Education Systems",
                          "Education systems range from compulsory schooling to higher education and vocational training",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Immigration",
                          "Immigration is the movement of people from one country to another for residence",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Poverty",
                          "Poverty is the condition of lacking sufficient resources for basic needs",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Healthcare Systems",
                          "Healthcare systems: single-payer (Canada), private insurance (US), mixed (UK NHS)",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Criminal Justice",
                          "Criminal justice involves law enforcement, courts, and corrections",
                          "yes", 0.99, s, "system"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Social Media",
                          "Social media platforms: Facebook, Twitter, Instagram, TikTok, YouTube",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Mass Media",
                          "Mass media includes newspapers, television, radio, and internet",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Demographic Transition",
                          "Demographic transition: from high birth/death rates to low birth/death rates",
                          "yes", 0.99, s, "model"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Social Mobility",
                          "Social mobility is the ability to move between social classes",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Cultural Relativism",
                          "Cultural relativism: understanding cultures on their own terms without judgment",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Civil Disobedience",
                          "Civil disobedience: nonviolent refusal to obey unjust laws (Gandhi, MLK)",
                          "yes", 0.99, s, "concept"),
            KnowledgeEntry(f"soc_{len(self._entries)}", d, "Human Development",
                          "Human development index combines life expectancy, education, and income",
                          "yes", 0.99, s, "measure"),
        ])

    # ══════════════════════════════════════════════════════════════
    # LAW & POLITICS — Various sources
    # ══════════════════════════════════════════════════════════════

    def _load_supplementary(self) -> None:
        """Load supplementary knowledge from the supplement module."""
        try:
            from .knowledge_supplement import SupplementaryKnowledge
            supp = SupplementaryKnowledge()
            for entry in supp.get_all():
                self._entries.append(entry)
        except Exception:
            pass  # Supplement module not available

    def _load_law_politics(self) -> None:
        d = "law_politics"
        s = "Various legal and political sources"

        self._entries.extend([
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Separation of Powers",
                          "Government is divided into legislative, executive, and judicial branches",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Rule of Law",
                          "The rule of law means all people and institutions are subject to law",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Human Rights",
                          "Universal human rights include life, liberty, education, and freedom of expression",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Constitution",
                          "A constitution is a set of fundamental principles governing a nation",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Due Process",
                          "Due process ensures fair treatment through the judicial system",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "International Law",
                          "International law governs relationships between nations (UN, treaties, customary law)",
                          "yes", 0.99, s, "definition"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Federalism",
                          "Federalism divides power between national and regional governments",
                          "yes", 0.99, s, "system"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Civil Rights",
                          "Civil rights protect individuals from discrimination based on race, gender, religion",
                          "yes", 0.99, s, "definition"),
        ])

        # More Law and Politics
        self._entries.extend([
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Bill of Rights",
                          "The US Bill of Rights (1791) guarantees 10 fundamental rights including free speech",
                          "1791", 0.99, s, "document"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "UN Charter",
                          "The UN Charter (1945) established the United Nations and its principles of peace",
                          "1945", 0.99, s, "document"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Election Systems",
                          "Electoral systems: majority, proportional representation, ranked-choice, electoral college",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Checks and Balances",
                          "Checks and balances prevent any one branch of government from becoming too powerful",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Habeas Corpus",
                          "Habeas corpus protects against unlawful detention by requiring legal justification",
                          "yes", 0.99, s, "principle"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Justice Systems",
                          "Justice systems: adversarial (US/UK), inquisitorial (France/Germany), restorative",
                          "yes", 0.99, s, "classification"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Marshall Plan",
                          "The Marshall Plan (1948) provided $13 billion to rebuild Western Europe after WWII",
                          "13 billion", 0.99, s, "event"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "NATO",
                          "NATO (founded 1949) is a military alliance of North American and European nations",
                          "1949", 0.99, s, "entity"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "European Union",
                          "The EU (est. 1993) is a political and economic union of 27 European countries",
                          "27", 0.99, s, "entity"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Freedom of Speech",
                          "Freedom of speech protects the right to express opinions without government censorship",
                          "yes", 0.99, s, "right"),
            KnowledgeEntry(f"law_{len(self._entries)}", d, "Separation of Church and State",
                          "The separation of church and state prevents government from establishing religion",
                          "yes", 0.99, s, "principle"),
        ])
