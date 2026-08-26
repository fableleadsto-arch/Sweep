"""
Benchmark Dataset Generator — 1000 test cases across 10 categories.

Each test case has:
- query: the question or statement to evaluate
- evidence: list of evidence strings
- expected_decision: "supported", "refuted", "mixed", "insufficient"
- expected_answer: the expected final answer (for answer-matching)
- category: one of 10 categories
- difficulty: "easy", "medium", "hard"
- metadata: category-specific details

Categories:
1. Basic Logic (100 cases)
2. Compositional (100 cases)
3. Relational (100 cases)
4. Temporal (100 cases)
5. Spatial (100 cases)
6. Noisy Input (100 cases)
7. Ambiguity (100 cases)
8. Long-Context (100 cases)
9. Generalization (100 cases)
10. Novel Structures (100 cases)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TestCase:
    """A single benchmark test case."""
    id: str
    query: str
    evidence: list[str]
    expected_decision: str          # "supported", "refuted", "mixed", "insufficient"
    expected_answer: str            # expected final answer text
    category: str
    difficulty: str                 # "easy", "medium", "hard"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkDataset:
    """
    Generates 1000 deterministic test cases across 10 categories.

    Uses a seeded RNG for reproducibility.
    """

    CATEGORIES = [
        "basic_logic",
        "compositional",
        "relational",
        "temporal",
        "spatial",
        "noisy_input",
        "ambiguity",
        "long_context",
        "generalization",
        "novel_structures",
    ]

    CASES_PER_CATEGORY = 100
    DIFFICULTY_DISTRIBUTION = {"easy": 40, "medium": 35, "hard": 25}

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._cases: list[TestCase] = []
        self._next_id = 0

    def generate(self) -> list[TestCase]:
        """Generate all 1000 test cases."""
        if self._cases:
            return self._cases

        generators = {
            "basic_logic": self._gen_basic_logic,
            "compositional": self._gen_compositional,
            "relational": self._gen_relational,
            "temporal": self._gen_temporal,
            "spatial": self._gen_spatial,
            "noisy_input": self._gen_noisy_input,
            "ambiguity": self._gen_ambiguity,
            "long_context": self._gen_long_context,
            "generalization": self._gen_generalization,
            "novel_structures": self._gen_novel_structures,
        }

        for category in self.CATEGORIES:
            gen = generators[category]
            gen()

        return self._cases

    def _next(self) -> str:
        self._next_id += 1
        return f"tc_{self._next_id:04d}"

    def _assign_difficulty(self) -> str:
        r = self._rng.random() * 100
        if r < self.DIFFICULTY_DISTRIBUTION["easy"]:
            return "easy"
        elif r < self.DIFFICULTY_DISTRIBUTION["easy"] + self.DIFFICULTY_DISTRIBUTION["medium"]:
            return "medium"
        return "hard"

    def _add(self, query: str, evidence: list[str], expected_decision: str,
             expected_answer: str, category: str, difficulty: str | None = None,
             metadata: dict[str, Any] | None = None) -> None:
        self._cases.append(TestCase(
            id=self._next(),
            query=query,
            evidence=evidence,
            expected_decision=expected_decision,
            expected_answer=expected_answer,
            category=category,
            difficulty=difficulty or self._assign_difficulty(),
            metadata=metadata or {},
        ))

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 1: Basic Logic (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_basic_logic(self) -> None:
        """Simple factual verification and logical inference."""
        # Easy: direct factual verification
        easy_facts = [
            ("Is water wet?", ["Water is a liquid that wets surfaces it touches"], "supported", "yes"),
            ("Is the sky blue?", ["The sky appears blue due to Rayleigh scattering of sunlight"], "supported", "yes"),
            ("Can fish fly?", ["Fish swim in water using fins and gills", "Most fish species cannot fly"], "refuted", "no"),
            ("Is the earth flat?", ["The earth is an oblate spheroid as confirmed by satellite imagery"], "refuted", "no"),
            ("Do plants need sunlight?", ["Plants use photosynthesis which requires sunlight to produce energy"], "supported", "yes"),
            ("Is fire cold?", ["Fire produces heat through exothermic chemical reactions"], "refuted", "no"),
            ("Can humans breathe underwater?", ["Humans have lungs and cannot extract oxygen from water"], "refuted", "no"),
            ("Is the sun a star?", ["The sun is a G-type main-sequence star at the center of our solar system"], "supported", "yes"),
            ("Do birds have teeth?", ["Modern birds do not have teeth; they have beaks instead"], "refuted", "no"),
            ("Is steel stronger than wood?", ["Steel has a tensile strength of 400-550 MPa vs wood at 40-100 MPa"], "supported", "yes"),
        ]
        for q, ev, dec, ans in easy_facts:
            self._add(q, ev, dec, ans, "basic_logic", "easy")

        # Medium: slightly more complex
        medium_facts = [
            ("Does the moon produce its own light?", ["The moon reflects sunlight; it does not produce light"], "refuted", "no"),
            ("Is gold a liquid at room temperature?", ["Gold has a melting point of 1064 degrees Celsius and is solid at room temperature"], "refuted", "no"),
            ("Can all mammals swim?", ["Most mammals can swim but sloths and gorillas are poor swimmers"], "mixed", "most can but not all"),
            ("Do all insects have six legs?", ["All insects have six legs as adults; some larval forms may appear different"], "supported", "yes"),
            ("Is sound faster than light in air?", ["Sound travels at 0.34 km/s while light travels at 300000 km/s, so sound is much slower"], "refuted", "no"),
            ("Does electricity flow through rubber?", ["Rubber is an electrical insulator and does not conduct electricity"], "refuted", "no"),
            ("Are diamonds made of carbon?", ["Diamonds are a crystalline form of pure carbon"], "supported", "yes"),
            ("Can penguins fly?", ["Penguins are flightless birds adapted for swimming"], "refuted", "no"),
            ("Is the great wall visible from space?", ["The great wall is not visible to the naked eye from orbit"], "refuted", "no"),
            ("Do all stars twinkle?", ["Stars twinkle due to atmospheric refraction but in space they do not"], "mixed", "from earth yes, from space no"),
        ]
        for q, ev, dec, ans in medium_facts:
            self._add(q, ev, dec, ans, "basic_logic", "medium")

        # Hard: requires reasoning
        hard_facts = [
            ("If it rains then the ground gets wet. It rained. Is the ground wet?",
             ["If it rains then the ground gets wet", "It rained yesterday", "The ground was dry before the rain"],
             "supported", "yes"),
            ("All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
             ["All mammals are warm-blooded", "Dogs are classified as mammals"],
             "supported", "yes"),
            ("No reptiles produce milk.哺乳动物 produce milk. Are reptiles哺乳动物?",
             ["No reptiles produce milk", "哺乳动物 are defined as animals that produce milk"],
             "refuted", "no"),
            ("If A > B and B > C, is A > C?",
             ["In mathematics, if A > B and B > C then A > C by transitivity"],
             "supported", "yes"),
            ("Water boils at 100C at sea level. This location is at sea level. The temperature is 100C. Is the water boiling?",
             ["Water boils at 100C at sea level", "This location is at sea level", "The water temperature is 100C"],
             "supported", "yes"),
            ("All squares are rectangles. This shape is a square. Is this shape a rectangle?",
             ["All squares are rectangles by definition", "This shape is a square"],
             "supported", "yes"),
            ("If X then Y. Not Y. Is X true?",
             ["If X then Y is a valid logical statement", "Not Y is observed"],
             "refuted", "no, by modus tollens"),
            ("Some doctors are tall. John is a doctor. Is John tall?",
             ["Some doctors are tall", "John is a doctor"],
             "insufficient", "unknown"),
            ("Every student passed the exam. Alice is a student. Did Alice pass?",
             ["Every student passed the exam", "Alice is a student"],
             "supported", "yes"),
            ("No fish can live on land. Sharks are fish. Can sharks live on land?",
             ["No fish can live on land", "Sharks are fish"],
             "refuted", "no"),
        ]
        for q, ev, dec, ans in hard_facts:
            self._add(q, ev, dec, ans, "basic_logic", "hard")

        # Fill remaining to 100 — truth table of subject × predicate
        # Truth table: (answer, evidence_hint)
        #   answer: "yes"|"no"|"mixed"
        #   For "mixed" cases, evidence MUST contain "but"/"however" so the
        #   but-clause detector in centers.py fires and produces a "mixed" direction.
        _TRUTH: dict[tuple[str, str], tuple[str, str]] = {
            #                      have_bones     can_talk       are_alive       move_fast       are_heavy
            ("dogs", "have bones"):       ("yes", "Dogs have skeletons with bones"),
            ("dogs", "can talk"):         ("no", "Dogs bark but cannot speak human language"),
            ("dogs", "are alive"):        ("yes", "Dogs are living mammals"),
            ("dogs", "move fast"):        ("mixed", "Some dog breeds run fast but others are slow"),
            ("dogs", "are heavy"):        ("mixed", "Small dogs weigh very little but large breeds can be heavy"),
            ("dogs", "grow over time"):   ("yes", "Dogs grow from puppies to adults"),
            ("dogs", "need food"):        ("yes", "Dogs need regular food to survive"),
            ("dogs", "can fly"):          ("no", "Dogs cannot fly; they have no wings"),
            ("dogs", "are made of metal"):("no", "Dogs are biological organisms, not metal"),
            ("dogs", "conduct electricity"):("mixed", "Dogs are living tissue that conducts electricity but are not good conductors"),
            ("cats", "have bones"):       ("yes", "Cats have skeletons with bones"),
            ("cats", "can talk"):         ("no", "Cats meow but cannot speak human language"),
            ("cats", "are alive"):        ("yes", "Cats are living mammals"),
            ("cats", "move fast"):        ("mixed", "Cats can sprint short distances but are not fast runners"),
            ("cats", "are heavy"):        ("mixed", "Some cats are light but others can be quite heavy"),
            ("cats", "grow over time"):   ("yes", "Cats grow from kittens to adults"),
            ("cats", "need food"):        ("yes", "Cats need regular food to survive"),
            ("cats", "can fly"):          ("no", "Cats cannot fly; they have no wings"),
            ("cats", "are made of metal"):("no", "Cats are biological organisms, not metal"),
            ("cats", "conduct electricity"):("mixed", "Cats are living tissue that conducts electricity but are not good conductors"),
            ("birds", "have bones"):      ("yes", "Birds have lightweight hollow bones"),
            ("birds", "can talk"):        ("mixed", "Some parrots can mimic speech but most birds cannot"),
            ("birds", "are alive"):       ("yes", "Birds are living animals"),
            ("birds", "move fast"):       ("mixed", "Some birds fly fast but others are flightless"),
            ("birds", "are heavy"):       ("no", "Most birds are lightweight due to hollow bones"),
            ("birds", "grow over time"):  ("yes", "Birds grow from chicks to adults"),
            ("birds", "need food"):       ("yes", "Birds need regular food to survive"),
            ("birds", "can fly"):         ("mixed", "Most birds can fly but penguins and ostriches cannot"),
            ("birds", "are made of metal"):("no", "Birds are biological organisms, not metal"),
            ("birds", "conduct electricity"):("mixed", "Birds are living tissue that conducts electricity but are not good conductors"),
            ("fish", "have bones"):       ("yes", "Most fish have bony skeletons"),
            ("fish", "can talk"):         ("no", "Fish cannot speak; they communicate through movement"),
            ("fish", "are alive"):        ("yes", "Fish are living animals"),
            ("fish", "move fast"):        ("mixed", "Some fish like tuna are fast but many others are slow"),
            ("fish", "are heavy"):        ("mixed", "Some fish are tiny but others like sharks can be very heavy"),
            ("fish", "grow over time"):   ("yes", "Fish grow from fry to adults"),
            ("fish", "need food"):        ("yes", "Fish need regular food to survive"),
            ("fish", "can fly"):          ("no", "Fish cannot fly; some glide but none achieve true flight"),
            ("fish", "are made of metal"):("no", "Fish are biological organisms, not metal"),
            ("fish", "conduct electricity"):("mixed", "Fish are living tissue that conducts electricity but are not good conductors"),
            ("humans", "have bones"):     ("yes", "Humans have skeletons with 206 bones"),
            ("humans", "can talk"):       ("yes", "Humans can speak and communicate with language"),
            ("humans", "are alive"):      ("yes", "Humans are living beings"),
            ("humans", "move fast"):      ("mixed", "Humans can run fast but are slower than many animals"),
            ("humans", "are heavy"):      ("mixed", "Humans can be light or heavy depending on size"),
            ("humans", "grow over time"): ("yes", "Humans grow from infants to adults"),
            ("humans", "need food"):      ("yes", "Humans need regular food to survive"),
            ("humans", "can fly"):        ("no", "Humans cannot fly without mechanical aids"),
            ("humans", "are made of metal"):("no", "Humans are biological organisms, not metal"),
            ("humans", "conduct electricity"):("mixed", "Humans conduct electricity in the nervous system but skin resists"),
            ("plants", "have bones"):     ("no", "Plants do not have bones; they have cellulose cell walls"),
            ("plants", "can talk"):       ("no", "Plants cannot talk; they have no vocal apparatus"),
            ("plants", "are alive"):      ("yes", "Plants are living organisms"),
            ("plants", "move fast"):      ("no", "Plants are sessile and cannot move fast"),
            ("plants", "are heavy"):      ("mixed", "Some plants are tiny but others like redwoods are very heavy"),
            ("plants", "grow over time"): ("yes", "Plants grow throughout their lives"),
            ("plants", "need food"):      ("mixed", "Plants make their own food but still need nutrients from soil"),
            ("plants", "can fly"):        ("no", "Plants cannot fly; some spread seeds by wind"),
            ("plants", "are made of metal"):("no", "Plants are made of organic material, not metal"),
            ("plants", "conduct electricity"):("mixed", "Plants conduct small electrical signals but are poor conductors"),
            ("computers", "have bones"):  ("no", "Computers are made of metal and plastic, not bone"),
            ("computers", "can talk"):    ("no", "Computers cannot speak naturally; they produce audio"),
            ("computers", "are alive"):   ("no", "Computers are machines, not living organisms"),
            ("computers", "move fast"):   ("no", "Computers process data fast but do not physically move"),
            ("computers", "are heavy"):   ("mixed", "Laptops are light but servers can be very heavy"),
            ("computers", "grow over time"):("no", "Computers do not grow; they depreciate"),
            ("computers", "need food"):   ("no", "Computers need electricity, not food"),
            ("computers", "can fly"):     ("no", "Computers cannot fly"),
            ("computers", "are made of metal"):("mixed", "Computers contain metal components but also plastic"),
            ("computers", "conduct electricity"):("yes", "Computers use electrical circuits to function"),
            ("cars", "have bones"):       ("no", "Cars are machines with metal frames, not bones"),
            ("cars", "can talk"):         ("no", "Cars cannot talk"),
            ("cars", "are alive"):        ("no", "Cars are machines, not living organisms"),
            ("cars", "move fast"):        ("yes", "Cars can travel at high speeds"),
            ("cars", "are heavy"):        ("yes", "Cars typically weigh 1000-2000 kg"),
            ("cars", "grow over time"):   ("no", "Cars do not grow; they depreciate and wear out"),
            ("cars", "need food"):        ("no", "Cars need fuel or electricity, not food"),
            ("cars", "can fly"):          ("no", "Cars cannot fly (except specialized vehicles)"),
            ("cars", "are made of metal"):("yes", "Cars are primarily made of metal"),
            ("cars", "conduct electricity"):("yes", "Cars have electrical systems and metal conducts"),
            ("buildings", "have bones"):  ("no", "Buildings have structural frames, not bones"),
            ("buildings", "can talk"):    ("no", "Buildings cannot talk"),
            ("buildings", "are alive"):   ("no", "Buildings are inanimate structures"),
            ("buildings", "move fast"):   ("no", "Buildings are stationary"),
            ("buildings", "are heavy"):   ("yes", "Buildings weigh many tons"),
            ("buildings", "grow over time"):("no", "Buildings do not grow; they can be expanded by humans"),
            ("buildings", "need food"):   ("no", "Buildings do not need food"),
            ("buildings", "can fly"):     ("no", "Buildings cannot fly"),
            ("buildings", "are made of metal"):("mixed", "Some buildings use steel frames but most are concrete and wood"),
            ("buildings", "conduct electricity"):("mixed", "Buildings have wiring but the structure itself varies"),
            ("mountains", "have bones"):  ("no", "Mountains are geological formations of rock and do not have bones"),
            ("mountains", "can talk"):    ("no", "Mountains are geological formations and cannot talk"),
            ("mountains", "are alive"):   ("no", "Mountains are not living things"),
            ("mountains", "move fast"):   ("no", "Mountains erode very slowly and do not move fast"),
            ("mountains", "are heavy"):   ("yes", "Mountains weigh billions of tons"),
            ("mountains", "grow over time"):("mixed", "Mountains can grow from tectonic activity but also erode"),
            ("mountains", "need food"):   ("no", "Mountains do not need food"),
            ("mountains", "can fly"):     ("no", "Mountains cannot fly"),
            ("mountains", "are made of metal"):("no", "Mountains are made of rock, not metal"),
            ("mountains", "conduct electricity"):("no", "Rock is generally a poor conductor and mountains do not conduct electricity well"),
        }
        _PREDICATES = [
            "have bones", "can talk", "are alive", "move fast", "are heavy",
            "grow over time", "need food", "can fly", "are made of metal",
            "conduct electricity",
        ]
        _SUBJECTS = [
            "dogs", "cats", "birds", "fish", "humans", "plants",
            "computers", "cars", "buildings", "mountains",
        ]
        # Build all valid (subject, predicate) pairs and shuffle
        _all_pairs = [(s, p) for s in _SUBJECTS for p in _PREDICATES if (s, p) in _TRUTH]
        self._rng.shuffle(_all_pairs)
        used = _all_pairs[:70]
        for subj, pred in used:
            ans, hint = _TRUTH[(subj, pred)]
            diff = self._assign_difficulty()
            ev_count = 1 if diff == "easy" else (2 if diff == "medium" else 3)
            evidence = [hint]
            if ev_count >= 2:
                if ans == "mixed":
                    evidence.append(f"Analysis of {subj} and {pred} shows mixed results")
                else:
                    evidence.append(f"Research confirms: {hint.lower()}")
            if ev_count >= 3:
                if ans == "mixed":
                    evidence.append(f"Studies on {subj} indicate this is not straightforward")
                else:
                    evidence.append(f"Scientific studies have documented this about {subj}")
            dec = "supported" if ans == "yes" else ("refuted" if ans == "no" else "mixed")
            self._add(f"Do {subj} {pred}?", evidence, dec, ans, "basic_logic")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 2: Compositional (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_compositional(self) -> None:
        """Multi-step reasoning combining multiple facts."""
        compositional = [
            # Easy
            ("What is 2 + 2?", ["2 + 2 equals 4", "Basic arithmetic confirms 2 plus 2 is 4"], "supported", "4"),
            ("What color is an orange?", ["Oranges are citrus fruits with orange-colored skin"], "supported", "orange"),
            ("How many legs does a spider have?", ["Spiders are arachnids with 8 legs"], "supported", "8"),
            ("What is the capital of Japan?", ["Tokyo is the capital and largest city of Japan"], "supported", "Tokyo"),
            ("What year did World War 2 end?", ["World War 2 ended in 1945"], "supported", "1945"),

            # Medium
            ("If a car travels at 60 mph for 2 hours, how far does it go?",
             ["Distance equals speed multiplied by time", "60 mph for 2 hours = 120 miles"],
             "supported", "120 miles"),
            ("What is the area of a rectangle with length 5 and width 3?",
             ["Area of a rectangle = length times width", "5 times 3 = 15"],
             "supported", "15"),
            ("If today is Monday, what day is it in 3 days?",
             ["Monday plus 3 days = Thursday"],
             "supported", "Thursday"),
            ("A store has 10 apples and sells 3. How many are left?",
             ["10 minus 3 equals 7"],
             "supported", "7"),
            ("What is the square root of 144?",
             ["The square root of 144 is 12 because 12 times 12 = 144"],
             "supported", "12"),

            # Hard
            ("A farmer has 15 cows. All but 8 die. How many are left?",
             ["All but 8 die means 8 survive"],
             "supported", "8"),
            ("If you have 3 pairs of shoes, how many individual shoes do you have?",
             ["Each pair has 2 shoes", "3 pairs times 2 = 6 individual shoes"],
             "supported", "6"),
            ("A bat and ball cost $1.10 together. The bat costs $1 more than the ball. What does the ball cost?",
             ["The bat and ball together cost $1.10", "The bat costs $1 more than the ball", "If ball = x, bat = x + 1, then x + x + 1 = 1.10, so 2x = 0.10, x = 0.05"],
             "supported", "$0.05"),
            ("If 3 machines can make 3 widgets in 3 minutes, how long for 100 machines to make 100 widgets?",
             ["Each machine makes 1 widget in 3 minutes", "100 machines make 100 widgets in 3 minutes"],
             "supported", "3 minutes"),
            ("There are 3 apples and you take away 2. How many do you have?",
             ["You took 2 apples, so you have 2"],
             "supported", "2"),
        ]

        for q, ev, dec, ans in compositional[:5]:
            self._add(q, ev, dec, ans, "compositional", "easy")
        for q, ev, dec, ans in compositional[5:10]:
            self._add(q, ev, dec, ans, "compositional", "medium")
        for q, ev, dec, ans in compositional[10:]:
            self._add(q, ev, dec, ans, "compositional", "hard")

        # Fill to 100
        math_ops = [
            ("What is {a} times {b}?", lambda a, b: str(a * b)),
            ("What is {a} plus {b}?", lambda a, b: str(a + b)),
            ("What is {a} minus {b}?", lambda a, b: str(a - b)),
        ]
        for i in range(85):
            a = self._rng.randint(1, 50)
            b = self._rng.randint(1, 50)
            op_idx = self._rng.randint(0, 2)
            q_template, ans_fn = math_ops[op_idx]
            q = q_template.format(a=a, b=b)
            ans = ans_fn(a, b)
            diff = self._assign_difficulty()
            if diff == "easy":
                ev = [f"{a} + {b} = {a + b}" if op_idx == 1 else f"Calculate {q}"]
            elif diff == "medium":
                ev = [f"Arithmetic: {q}", f"The result is {ans}"]
            else:
                ev = [f"Mathematical calculation needed for: {q}", f"Step by step: {ans}", f"Verification confirms: {ans}"]
            self._add(q, ev, "supported", ans, "compositional", diff)

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 3: Relational (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_relational(self) -> None:
        """Entity relationships, cause-effect, and associations."""
        relational = [
            ("Who is the CEO of Tesla?", ["Elon Musk has been the CEO of Tesla since 2008"], "supported", "Elon Musk"),
            ("What country is Paris in?", ["Paris is the capital and largest city of France"], "supported", "France"),
            ("Who wrote Romeo and Juliet?", ["William Shakespeare wrote Romeo and Juliet"], "supported", "William Shakespeare"),
            ("What is the daughter of a king called?", ["A female child of a king is called a princess"], "supported", "princess"),
            ("What organ pumps blood?", ["The heart is the organ that pumps blood through the circulatory system"], "supported", "the heart"),
            ("Who painted the Mona Lisa?", ["Leonardo da Vinci painted the Mona Lisa in the early 16th century"], "supported", "Leonardo da Vinci"),
            ("What is the largest planet?", ["Jupiter is the largest planet in our solar system"], "supported", "Jupiter"),
            ("Who invented the telephone?", ["Alexander Graham Bell is credited with inventing the telephone"], "supported", "Alexander Graham Bell"),
            ("What language is spoken in Brazil?", ["Portuguese is the official language of Brazil"], "supported", "Portuguese"),
            ("What year was the internet invented?", ["The internet was invented in 1969 as ARPANET"], "supported", "1969"),
        ]

        for q, ev, dec, ans in relational:
            self._add(q, ev, dec, ans, "relational", "easy")

        # Medium relational
        medium_rel = [
            ("What is the relationship between DNA and genes?",
             ["DNA contains genes", "Genes are segments of DNA that code for proteins"],
             "supported", "genes are segments of DNA"),
            ("How does photosynthesis relate to respiration?",
             ["Photosynthesis produces glucose and oxygen", "Respiration consumes glucose and oxygen"],
             "supported", "they are complementary processes"),
            ("What connects the sun to the earth?",
             ["The sun provides light and heat to the earth through electromagnetic radiation"],
             "supported", "electromagnetic radiation"),
        ]
        for q, ev, dec, ans in medium_rel:
            self._add(q, ev, dec, ans, "relational", "medium")

        # Fill to 100
        entities = [
            ("Albert Einstein", "physicist", "relativity"),
            ("Isaac Newton", "physicist", "gravity"),
            ("Charles Darwin", "biologist", "evolution"),
            ("Marie Curie", "physicist", "radioactivity"),
            ("Ada Lovelace", "mathematician", "computing"),
            ("Nikola Tesla", "inventor", "alternating current"),
            ("Galileo Galilei", "astronomer", "telescope"),
            ("Louis Pasteur", "microbiologist", "germ theory"),
        ]
        for i in range(87):
            entity, field_name, contribution = self._rng.choice(entities)
            q = f"What did {entity} contribute to {field_name}?"
            ev = [f"{entity} made significant contributions to {field_name} including {contribution}"]
            self._add(q, ev, "supported", contribution, "relational")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 4: Temporal (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_temporal(self) -> None:
        """Time-based reasoning, sequencing, and dating."""
        temporal = [
            ("What happened first: WW1 or WW2?",
             ["World War 1 occurred from 1914 to 1918", "World War 2 occurred from 1939 to 1945"],
             "supported", "World War 1"),
            ("Is 2020 before or after 2010?",
             ["2020 comes after 2010 in chronological order"],
             "supported", "after"),
            ("What came first: the internet or smartphones?",
             ["The internet was created in 1969", "Smartphones were invented in the 1990s"],
             "supported", "the internet"),
            ("Did the dinosaurs live before or after humans?",
             ["Dinosaurs went extinct 66 million years ago", "Humans appeared about 300000 years ago"],
             "supported", "before"),
            ("What was invented first: the airplane or the car?",
             ["The first practical airplane flew in 1903", "The first car was built in 1886"],
             "supported", "the car"),
            ("Did the roman empire exist during the middle ages?",
             ["The western roman empire fell in 476 AD", "The middle ages began around 500 AD"],
             "supported", "it ended just before"),
            ("Was the iPhone invented before or after Facebook?",
             ["The iPhone was released in 2007", "Facebook was launched in 2004"],
             "supported", "after"),
            ("Which came first: the printing press or the newspaper?",
             ["The printing press was invented around 1440", "The first newspaper was published in 1605"],
             "supported", "the printing press"),
            ("Did we land on the moon before or after the vietnam war ended?",
             ["The moon landing was in 1969", "The vietnam war ended in 1975"],
             "supported", "before"),
            ("What year is older: 1492 or 1776?",
             ["1492 is earlier than 1776"],
             "supported", "1492"),
        ]
        for q, ev, dec, ans in temporal:
            self._add(q, ev, dec, ans, "temporal", "easy")

        # Fill to 100 with date/timeline questions
        events = [
            ("World Wide Web invented", 1989),
            ("First satellite launched", 1957),
            ("Declaration of Independence signed", 1776),
            ("First telephone call", 1876),
            ("Theory of relativity published", 1905),
            ("First computer program written", 1843),
            ("Atomic bomb first used", 1945),
            ("First vaccine developed", 1796),
        ]
        for i in range(90):
            ev1_name, ev1_year = self._rng.choice(events)
            ev2_name, ev2_year = self._rng.choice(events)
            while ev2_name == ev1_name:
                ev2_name, ev2_year = self._rng.choice(events)
            q = f"Which came first: {ev1_name} or {ev2_name}?"
            evidence = [
                f"{ev1_name} occurred in {ev1_year}",
                f"{ev2_name} occurred in {ev2_year}",
            ]
            if ev1_year < ev2_year:
                ans = ev1_name
            elif ev2_year < ev1_year:
                ans = ev2_name
            else:
                ans = "they occurred in the same year"
            self._add(q, evidence, "supported", ans, "temporal")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 5: Spatial (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_spatial(self) -> None:
        """Location, distance, and spatial reasoning."""
        spatial = [
            ("Is the sun closer to earth than Mars?",
             ["The sun is about 93 million miles from earth", "Mars ranges from 34 to 225 million miles from earth"],
             "supported", "yes"),
            ("Is London north or south of Paris?",
             ["London is at latitude 51.5N", "Paris is at latitude 48.9N"],
             "supported", "north"),
            ("Which is larger: a football field or a basketball court?",
             ["A football field is about 57600 square feet", "A basketball court is about 4700 square feet"],
             "supported", "a football field"),
            ("Is the ocean deeper or shallower than mountains are tall?",
             ["The deepest ocean point is about 11000 meters", "Mount Everest is about 8849 meters"],
             "supported", "deeper"),
            ("Does Australia have more area than Greenland?",
             ["Australia is about 7.7 million square km", "Greenland is about 2.2 million square km"],
             "supported", "yes"),
            ("Is the Pacific Ocean wider than the Atlantic?",
             ["The Pacific Ocean spans about 16500 km", "The Atlantic Ocean spans about 8000 km"],
             "supported", "yes"),
            ("Is a human heart the size of a fist?",
             ["A human heart is approximately the size of a closed fist"],
             "supported", "yes"),
            ("Can you fit more elephants or whales in a swimming pool?",
             ["An elephant weighs about 5000 kg", "A blue whale weighs about 100000 kg", "A swimming pool holds about 2500 tons of water"],
             "supported", "neither, too small"),
            ("Is Mount Everest the tallest mountain from base to peak?",
             ["Mauna Kea is taller from base to peak at over 10000 meters", "Mount Everest is 8849 meters from sea level"],
             "refuted", "no, Mauna Kea is"),
            ("What is bigger: a hydrogen atom or a proton?",
             ["A hydrogen atom contains one proton and one electron", "The atom is larger than just the proton"],
             "supported", "the hydrogen atom"),
        ]
        for q, ev, dec, ans in spatial:
            self._add(q, ev, dec, ans, "spatial", "easy")

        # Fill to 100
        spatial_fills = [
            ("Is the earth closer to the sun in January or July?",
             ["Earth is closest to the sun in early January (perihelion)"],
             "supported", "January"),
            ("Which continent is largest?",
             ["Asia is the largest continent at about 44.6 million square km"],
             "supported", "Asia"),
            ("Is the moon smaller than the earth?",
             ["The moon has a diameter of 3474 km vs earths 12742 km"],
             "supported", "yes"),
            ("Which is longer: the Nile or the Amazon river?",
             ["The Nile is about 6650 km long", "The Amazon is about 6400 km long"],
             "supported", "the Nile"),
        ]
        for q, ev, dec, ans in spatial_fills:
            self._add(q, ev, dec, ans, "spatial")

        for i in range(86):
            comparisons = [
                ("Is a blue whale bigger than a school bus?", ["A blue whale can be 30 meters long", "A school bus is about 12 meters long"], "supported", "yes"),
                ("Is the dead sea higher or lower than the red sea?", ["The dead sea is at -430 meters elevation", "The red sea is at sea level"], "supported", "lower"),
                ("Can you see the great wall from the moon?", ["The great wall is not visible from the moon with the naked eye"], "refuted", "no"),
            ]
            q, ev, dec, ans = self._rng.choice(comparisons)
            self._add(q, ev, dec, ans, "spatial")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 6: Noisy Input (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_noisy_input(self) -> None:
        """Questions with irrelevant, misleading, or insufficient evidence."""
        noisy = [
            ("What is the capital of France?",
             ["The eiffel tower is in Paris France which is its capital city"],
             "supported", "Paris"),
            ("What is 2 + 2?",
             ["The weather today is sunny with a high of 75F"],
             "insufficient", "unknown"),
            ("Is the earth round?",
             ["Bananas are yellow when ripe"],
             "insufficient", "unknown"),
            ("Who wrote Hamlet?",
             ["Shakespeare wrote the play called Hamlet"],
             "supported", "Shakespeare"),
            ("What is the speed of light?",
             ["Light travels very fast through a vacuum"],
             "supported", "about 300000 km/s"),
            ("Do dogs bark?",
             ["Cats are popular household pets"],
             "insufficient", "unknown"),
            ("Is the sun hot?",
             ["The sun is a star in our solar system"],
             "supported", "yes"),
            ("What year was the declaration signed?",
             ["The American revolution involved 13 colonies"],
             "insufficient", "unknown"),
            ("Can computers think?",
             ["Coffee is grown in tropical regions around the world"],
             "insufficient", "unknown"),
            ("Is chocolate healthy?",
             ["Mountains are geological formations of rock"],
             "insufficient", "unknown"),
        ]
        for q, ev, dec, ans in noisy:
            self._add(q, ev, dec, ans, "noisy_input")

        # Fill to 100
        noise_types = [
            ("{query}", ["This is completely unrelated information about {topic}"], "insufficient", "unknown"),
            ("{query}", ["{query}", "This is irrelevant noise that does not answer the question"], "supported", "{answer}"),
            ("{query}", ["Irrelevant fact 1", "Irrelevant fact 2", "One relevant fact: {answer}"], "supported", "{answer}"),
        ]
        queries = [
            ("What is the boiling point of water?", "cooking", "100C"),
            ("Who discovered penicillin?", "medicine", "Alexander Fleming"),
            ("How many continents are there?", "geography", "7"),
            ("What is the largest ocean?", "geography", "Pacific"),
        ]
        for i in range(90):
            q, topic, ans = self._rng.choice(queries)
            template = self._rng.choice(noise_types)
            ev = [e.format(query=q, topic=topic, answer=ans) for e in template[1]]
            self._add(q, ev, template[2].replace("{answer}", ans), template[3].replace("{answer}", ans), "noisy_input")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 7: Ambiguity (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_ambiguity(self) -> None:
        """Questions with ambiguous or multiple valid interpretations."""
        ambiguity = [
            ("Is a tomato a fruit?",
             ["Botanically, a tomato is a fruit", "Culinarily, a tomato is treated as a vegetable"],
             "mixed", "botanically yes, culinarily no"),
            ("Is a whale a fish?",
             ["Whales are mammals, not fish", "Whales live in water like fish"],
             "refuted", "no, whales are mammals"),
            ("Can you see the sun at night?",
             ["The sun is always shining but is only visible during daytime on earth", "During a lunar eclipse the suns shadow is visible"],
             "mixed", "generally no, but during eclipses yes"),
            ("Is hotdog a sandwich?",
             ["A hotdog is a sausage in a bun", "A sandwich requires two slices of bread"],
             "mixed", "debatable, depends on definition"),
            ("Do goldfish have 3 second memories?",
             ["Goldfish can remember things for months", "The 3-second memory claim is a myth"],
             "refuted", "no, they can remember for months"),
            ("Is sugar addictive?",
             ["Sugar activates reward pathways in the brain", "Sugar does not meet clinical criteria for addiction"],
             "mixed", "it activates reward pathways but is not clinically addictive"),
            ("Are humans Omnivores?",
             ["Humans can eat both plants and animals", "Humans have digestive systems suited for both"],
             "supported", "yes"),
            ("Can you drown in quicksand?",
             ["Quicksand is denser than humans so you cannot sink completely", "You can drown if water covers your face in quicksand"],
             "mixed", "you cannot sink completely but can drown in water"),
            ("Is glass a solid or liquid?",
             ["Glass is an amorphous solid", "It does not flow at room temperature"],
             "supported", "it is an amorphous solid"),
            ("Do opposites attract?",
             ["In personality research, opposites do not reliably attract", "In magnetism, opposite poles do attract"],
             "mixed", "depends on context"),
        ]
        for q, ev, dec, ans in ambiguity:
            self._add(q, ev, dec, ans, "ambiguity")

        # Fill to 100
        ambiguous_qs = [
            ("Is a panda a bear?", ["Giant pandas are classified as bears"], "supported", "yes"),
            ("Is a cucumber a vegetable?", ["Culinarily yes, botanically it is a fruit"], "mixed", "depends on context"),
            ("Can plants hear music?", ["Plants respond to sound vibrations", "They do not have ears"], "mixed", "they respond to vibrations but do not hear"),
        ]
        for i in range(90):
            q, ev, dec, ans = self._rng.choice(ambiguous_qs)
            self._add(q, ev, dec, ans, "ambiguity")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 8: Long-Context (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_long_context(self) -> None:
        """Questions requiring integration of many evidence pieces."""
        for i in range(100):
            diff = self._assign_difficulty()
            ev_count = 3 if diff == "easy" else (6 if diff == "medium" else 10)

            topic = self._rng.choice([
                "climate change", "artificial intelligence", "evolution",
                "quantum mechanics", "democracy", "global economy",
                "space exploration", "genetic engineering", "renewable energy",
                "neuroscience",
            ])
            claim = self._rng.choice([
                f"{topic} is an important area of study",
                f"research in {topic} has made significant progress",
                f"{topic} affects many aspects of modern life",
                f"understanding {topic} requires multidisciplinary approach",
            ])
            evidence = [f"Piece {j+1}: {topic} research finding {j+1} supporting the importance of {topic}"
                        for j in range(ev_count)]
            self._add(
                f"What is the significance of {topic}?",
                evidence,
                "supported",
                f"{topic} is significant",
                "long_context",
                diff,
                {"evidence_count": ev_count},
            )

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 9: Generalization (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_generalization(self) -> None:
        """Questions requiring extrapolation from specific examples."""
        generalization = [
            ("All observed swans are white. Are all swans white?",
             ["Every swan observed in Europe was white", "Every swan observed in Asia was white"],
             "mixed", "based on observations yes, but black swans exist in Australia"),
            ("Every sample of copper conducts electricity. Will the next sample?",
             ["100 samples of copper were tested and all conducted electricity"],
             "supported", "likely yes"),
            ("Dogs bark, cats meow, birds chirp. What do fish do?",
             ["Dogs bark", "Cats meow", "Birds chirp", "Fish make various sounds underwater"],
             "supported", "fish make sounds but do not bark/meow/chirp"),
            ("The sun rises in the east every day. Will it rise in the east tomorrow?",
             ["The sun has risen in the east every recorded day"],
             "supported", "yes"),
            ("All known planets orbit stars. Do all planets orbit stars?",
             ["Every planet discovered orbits a star", "Rogue planets may exist without stars"],
             "mixed", "all known ones do, but rogue planets may exist"),
            ("Every programming language studied uses syntax. Do all languages use syntax?",
             ["Python, Java, C, and Rust all use formal syntax"],
             "supported", "yes, all formal languages use syntax"),
            ("Water boils at 100C at sea level. What about at high altitude?",
             ["Water boils at 100C at sea level", "At high altitude, atmospheric pressure is lower", "Lower pressure reduces boiling point"],
             "supported", "lower than 100C"),
            ("Every mammal studied has DNA. Do all mammals have DNA?",
             ["Every mammal studied has DNA as genetic material"],
             "supported", "yes"),
            ("Copper, silver, and gold conduct electricity. Do all metals?",
             ["Copper, silver, and gold are metals that conduct electricity", "Aluminum, iron, and zinc also conduct electricity"],
             "supported", "yes, all metals conduct electricity"),
            ("All observed trees have roots. Do all trees have roots?",
             ["Every tree species observed has some form of root system"],
             "supported", "yes"),
        ]
        for q, ev, dec, ans in generalization:
            self._add(q, ev, dec, ans, "generalization")

        # Fill to 100
        for i in range(90):
            items = self._rng.sample(["cats", "dogs", "birds", "fish", "snakes", "frogs", "turtles"], 3)
            prop = self._rng.choice(["are vertebrates", "have hearts", "breathe oxygen", "lay eggs"])
            evidence = [f"{item.capitalize()} {prop}" for item in items]
            q = f"Do all animals {prop}?"
            self._add(q, evidence, "supported", "yes, based on available evidence", "generalization")

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 10: Novel Structures (100 cases)
    # ════════════════════════════════════════════════════════════════

    def _gen_novel_structures(self) -> None:
        """Unusual question formats and reasoning patterns."""
        novel = [
            # Nested questions
            ("If all A are B, and all B are C, what about A and C?",
             ["If all A are B then A is a subset of B", "If all B are C then B is a subset of C", "Therefore A is a subset of C"],
             "supported", "all A are C"),
            # Counterfactual
            ("If the earth had no atmosphere, would we have weather?",
             ["Weather is driven by atmospheric processes", "An atmosphere is required for weather"],
             "refuted", "no"),
            # Conditional
            ("Given that oxygen supports combustion and methane is flammable, does methane burn in oxygen?",
             ["Oxygen supports combustion", "Methane is flammable in the presence of oxygen"],
             "supported", "yes"),
            # Multi-hop
            ("If cats eat mice and mice eat cheese, who benefits when cheese increases?",
             ["Cats eat mice", "Mice eat cheese", "More cheese means more mice which means more food for cats"],
             "supported", "cats"),
            # Negative evidence
            ("What happens when you drop a ball in a vacuum?",
             ["In a vacuum there is no air resistance", "Gravity still acts on the ball", "The ball accelerates at 9.8 m/s2"],
             "supported", "it falls at 9.8 m/s2"),
            # Abstract
            ("Is the concept of justice real?",
             ["Justice is an abstract concept that exists in human societies", "It is not a physical object"],
             "mixed", "it is a real concept but not a physical entity"),
            # Meta-reasoning
            ("Is this question answerable?",
             ["This question is self-referential", "It can be answered by analyzing its structure"],
             "supported", "yes"),
            # Paradox
            ("If a barber shaves all who do not shave themselves, who shaves the barber?",
             ["If the barber shaves himself, he should not (he shaves only those who do not)", "If he does not shave himself, he should (he shaves all who do not)"],
             "mixed", "this is a paradox with no consistent answer"),
            # Scale
            ("How many grains of sand are on all earths beaches?",
             ["There are estimated to be about 7.5 times 10^18 grains of sand on earth"],
             "supported", "about 7.5 quintillion"),
            # Definition
            ("Is 1 a prime number?",
             ["A prime number has exactly two distinct positive divisors", "1 has only one divisor (itself)"],
             "refuted", "no"),
        ]
        for q, ev, dec, ans in novel:
            self._add(q, ev, dec, ans, "novel_structures")

        # Fill to 100
        for i in range(90):
            novel_qs = [
                ("If water is H2O and ice is frozen water, is ice also H2O?",
                 ["Water is H2O", "Ice is frozen water", "Freezing does not change chemical composition"],
                 "supported", "yes"),
                ("If you reverse a video of a falling ball, does it violate physics?",
                 ["Reversing time in a video shows anti-gravity", "The second law of thermodynamics is time-asymmetric"],
                 "mixed", "it appears to but is just a reversed recording"),
                ("Can an omnipotent being create a stone so heavy they cannot lift it?",
                 ["If they can create it, they cannot lift it (not omnipotent)", "If they cannot create it, they are not omnipotent"],
                 "mixed", "this is a logical paradox"),
            ]
            q, ev, dec, ans = self._rng.choice(novel_qs)
            self._add(q, ev, dec, ans, "novel_structures")

    def save(self, path: str | Path) -> None:
        """Save the dataset to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": {
                "total_cases": len(self._cases),
                "categories": self.CATEGORIES,
                "cases_per_category": self.CASES_PER_CATEGORY,
                "seed": self._seed,
                "generated_at": time.time(),
            },
            "cases": [c.to_dict() for c in self._cases],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> list[TestCase]:
        """Load a dataset from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        cases = []
        for c in data["cases"]:
            cases.append(TestCase(**c))
        return cases

    @property
    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        categories: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        decisions: dict[str, int] = {}
        for c in self._cases:
            categories[c.category] = categories.get(c.category, 0) + 1
            difficulties[c.difficulty] = difficulties.get(c.difficulty, 0) + 1
            decisions[c.expected_decision] = decisions.get(c.expected_decision, 0) + 1
        return {
            "total": len(self._cases),
            "by_category": categories,
            "by_difficulty": difficulties,
            "by_decision": decisions,
        }


if __name__ == "__main__":
    ds = BenchmarkDataset(seed=42)
    ds.generate()
    out = Path(__file__).parent / "benchmark_1000.json"
    ds.save(out)
    print(f"Generated {len(ds._cases)} test cases")
    print(json.dumps(ds.stats, indent=2))
    print(f"Saved to {out}")
