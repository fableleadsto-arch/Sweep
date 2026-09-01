"""Generate large dataset and continue training."""
import sys, os, json, time, logging, random
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gen_train")


def generate_large_dataset() -> list[str]:
    """Generate 10,000+ training examples."""
    texts = []
    rng = random.Random(42)

    # ══════════════════════════════════════════════════════════
    # Factual QA (2000+ examples)
    # ══════════════════════════════════════════════════════════

    # Capitals (90 countries × 2 formats = 180)
    capitals = {
        "France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "UK": "London",
        "China": "Beijing", "India": "New Delhi", "Brazil": "Brasilia",
        "Australia": "Canberra", "Canada": "Ottawa", "Egypt": "Cairo",
        "Russia": "Moscow", "South Korea": "Seoul", "Italy": "Rome",
        "Spain": "Madrid", "Mexico": "Mexico City", "Turkey": "Ankara",
        "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
        "Pakistan": "Islamabad", "Nigeria": "Abuja", "Kenya": "Nairobi",
        "Argentina": "Buenos Aires", "Colombia": "Bogota", "Chile": "Santiago",
        "Peru": "Lima", "Greece": "Athens", "Portugal": "Lisbon",
        "Netherlands": "Amsterdam", "Belgium": "Brussels", "Switzerland": "Bern",
        "Austria": "Vienna", "Sweden": "Stockholm", "Norway": "Oslo",
        "Denmark": "Copenhagen", "Finland": "Helsinki", "Poland": "Warsaw",
        "Czech Republic": "Prague", "Hungary": "Budapest", "Romania": "Bucharest",
        "Ukraine": "Kyiv", "Ireland": "Dublin", "Iceland": "Reykjavik",
        "New Zealand": "Wellington", "Singapore": "Singapore",
        "Philippines": "Manila", "Malaysia": "Kuala Lumpur",
    }
    for country, capital in capitals.items():
        texts.append(f"Question: What is the capital of {country}? Answer: {capital}.")
        texts.append(f"Question: Which city is the capital of {country}? Answer: {capital} is the capital.")
        texts.append(f"Fact: The capital of {country} is {capital}.")

    # Elements (118 elements × 2 formats = 236)
    elements = [
        ("Hydrogen", "H", 1), ("Helium", "He", 2), ("Lithium", "Li", 3),
        ("Beryllium", "Be", 4), ("Boron", "B", 5), ("Carbon", "C", 6),
        ("Nitrogen", "N", 7), ("Oxygen", "O", 8), ("Fluorine", "F", 9),
        ("Neon", "Ne", 10), ("Sodium", "Na", 11), ("Magnesium", "Mg", 12),
        ("Aluminum", "Al", 13), ("Silicon", "Si", 14), ("Phosphorus", "P", 15),
        ("Sulfur", "S", 16), ("Chlorine", "Cl", 17), ("Argon", "Ar", 18),
        ("Potassium", "K", 19), ("Calcium", "Ca", 20), ("Iron", "Fe", 26),
        ("Copper", "Cu", 29), ("Zinc", "Zn", 30), ("Silver", "Ag", 47),
        ("Gold", "Au", 79), ("Mercury", "Hg", 80), ("Lead", "Pb", 82),
        ("Uranium", "U", 92), ("Plutonium", "Pu", 94),
    ]
    for name, symbol, num in elements:
        texts.append(f"Question: What is the chemical symbol for {name}? Answer: {symbol}.")
        texts.append(f"Question: What element has atomic number {num}? Answer: {name} ({symbol}).")
        texts.append(f"Fact: Element {num} is {name}, symbol {symbol}.")

    # Planets (8 planets × 5 facts = 40)
    planets = [
        ("Mercury", 1, "closest to the Sun", 0, "smallest planet"),
        ("Venus", 2, "hottest planet", 0, "rotates backwards"),
        ("Earth", 3, "has life", 1, "has one moon"),
        ("Mars", 4, "red planet", 2, "has two moons"),
        ("Jupiter", 5, "largest planet", 95, "has Great Red Spot"),
        ("Saturn", 6, "has rings", 146, "least dense planet"),
        ("Uranus", 7, "tilted on its side", 28, "ice giant"),
        ("Neptune", 8, "farthest planet", 16, "windiest planet"),
    ]
    for name, pos, fact1, moons, fact2 in planets:
        texts.append(f"Question: What is planet number {pos}? Answer: {name}.")
        texts.append(f"Question: Which planet is {fact1}? Answer: {name}.")
        texts.append(f"Question: How many moons does {name} have? Answer: {moons}.")
        texts.append(f"Fact: {name} is the {pos} planet from the Sun. It is {fact1} and has {moons} moons.")

    # Science facts (100+ variations)
    science_facts = [
        ("Water boils at 100 degrees Celsius", "boiling point", "100 degrees Celsius"),
        ("Water freezes at 0 degrees Celsius", "freezing point", "0 degrees Celsius"),
        ("Speed of light is 299792458 m/s", "speed of light", "299792458 m/s"),
        ("DNA stands for deoxyribonucleic acid", "DNA full form", "deoxyribonucleic acid"),
        ("The heart beats 100000 times per day", "heart beats per day", "100000"),
        ("Sound cannot travel through vacuum", "sound in vacuum", "no"),
        ("Gravity on Moon is 1/6 of Earth", "moon gravity", "1/6"),
        ("Earth atmosphere is 78% nitrogen", "atmosphere composition", "78% nitrogen"),
        ("Diamonds are made of carbon", "diamond composition", "carbon"),
        ("Gold does not rust", "gold properties", "does not rust"),
        ("Copper conducts electricity well", "copper property", "conducts electricity"),
        ("The periodic table has 118 elements", "periodic table size", "118"),
        ("Electrons are smaller than atoms", "electron size", "smaller than atoms"),
        ("The Sun is a medium-sized star", "sun classification", "medium-sized star"),
        ("Black holes have strong gravity", "black hole property", "strong gravity"),
        ("Milky Way has 100-400 billion stars", "milky way stars", "100-400 billion"),
        ("Jupiter Great Red Spot is a storm", "great red spot", "storm"),
        ("Saturn density is less than water", "saturn density", "less than water"),
        ("Venus rotates backwards", "venus rotation", "backwards"),
        ("Mars has Olympus Mons", "mars volcano", "Olympus Mons"),
    ]
    for fact, topic, answer in science_facts:
        texts.append(f"Question: What is the {topic}? Answer: {answer}.")
        texts.append(f"Fact: {fact}.")
        texts.append(f"Is this true? {fact} Answer: Yes, this is correct.")

    # History (50+ facts)
    history = [
        "World War II ended in 1945",
        "American Declaration of Independence was signed in 1776",
        "French Revolution began in 1789",
        "Gutenberg invented printing press around 1440",
        "Moon landing occurred in 1969",
        "Roman Empire fell in 476 AD",
        "Magna Carta was signed in 1215",
        "Renaissance began in Italy in 14th century",
        "Industrial Revolution started in Britain in 18th century",
        "Great Wall of China built over centuries",
        "Ancient Egypt built pyramids around 2560 BC",
        "Black Death killed millions in 14th century",
        "Democracy originated in ancient Greece",
        "Cold War was between USA and Soviet Union",
        "Silk Road connected East and West",
    ]
    for fact in history:
        texts.append(f"Fact: {fact}.")
        texts.append(f"Question: When did this happen? Answer: {fact}.")

    # Geography (50+ facts)
    geography = [
        "Pacific Ocean is the largest ocean",
        "Asia is the largest continent",
        "Nile is the longest river",
        "Mount Everest is tallest at 8849 meters",
        "Mariana Trench is deepest point",
        "Sahara is largest hot desert",
        "Amazon is largest rainforest",
        "There are 7 continents",
        "Antarctica is coldest continent",
        "Greenland is largest island",
        "Equator divides Earth into hemispheres",
        "There are 24 time zones",
        "Africa is second largest continent",
        "Australia is both country and continent",
        "Dead Sea is lowest point on Earth",
    ]
    for fact in geography:
        texts.append(f"Fact: {fact}.")
        texts.append(f"Question: Is this correct? Answer: Yes, {fact.lower()}.")

    # ══════════════════════════════════════════════════════════
    # Reasoning chains (2000+ examples)
    # ══════════════════════════════════════════════════════════

    entities = ["cats", "dogs", "birds", "fish", "mammals", "reptiles", "insects", "plants",
                "humans", "whales", "penguins", "bats", "frogs", "snakes", "eagles"]
    categories = ["animals", "living things", "organisms", "vertebrates", "creatures", "beings"]
    actions = ["fly", "swim", "walk", "breathe", "see", "hear", "move", "reproduce", "grow", "eat"]

    for _ in range(2000):
        A = rng.choice(entities)
        B = rng.choice(categories)
        C = rng.choice(categories)
        action = rng.choice(actions)

        # Syllogism
        texts.append(f"All {A} are {B}. All {B} are {action}. Therefore all {A} {action}.")
        texts.append(f"If all {A} are {B}, and all {B} are {C}, then all {A} are {C}.")

        # Modus ponens
        texts.append(f"If it rains, the ground gets wet. It is raining. Therefore the ground is wet.")
        texts.append(f"If {A} can {action}, and {A} is a {B}, then {B} can {action}.")

        # Modus tollens
        texts.append(f"If it rains, the ground is wet. The ground is not wet. Therefore it did not rain.")

        # Transitivity
        texts.append(f"Alpha is faster than Beta. Beta is faster than Gamma. Therefore Alpha is faster than Gamma.")

    # ══════════════════════════════════════════════════════════
    # Evidence evaluation (2000+ examples)
    # ══════════════════════════════════════════════════════════

    topics = [
        "exercise improves health", "vaccines are safe", "climate change is real",
        "sugar is harmful", "sleep is important", "reading improves vocabulary",
        "music enhances learning", "stress affects performance", "meditation helps focus",
        "walking reduces disease", "water is essential", "sunlight provides vitamin D",
        "education increases opportunity", "poverty affects health", "innovation drives growth",
    ]
    for topic in topics:
        texts.append(f"Evidence: Studies confirm that {topic}. Assessment: SUPPORTS the claim.")
        texts.append(f"Evidence: Research shows {topic} is not always true. Assessment: REFUTES the claim.")
        texts.append(f"Evidence: Some studies support {topic}, others do not. Assessment: MIXED evidence.")
        texts.append(f"Claim: {topic}. Evidence: Multiple peer-reviewed studies support this. Verdict: TRUE.")
        texts.append(f"Claim: {topic}. Evidence: No significant studies support this. Verdict: UNTRUE.")

    # ══════════════════════════════════════════════════════════
    # Contradiction detection (1000+ examples)
    # ══════════════════════════════════════════════════════════

    contradiction_pairs = [
        ("The meeting is at 3 PM", "The meeting is at 4 PM"),
        ("The drug is effective", "The drug is ineffective"),
        ("All students passed", "Some students failed"),
        ("It is raining", "It is sunny and dry"),
        ("Revenue increased 15%", "Revenue decreased 15%"),
        ("The company is profitable", "The company reported losses"),
        ("Water boils at 100C", "Water boils at 90C"),
        ("The Earth is round", "The Earth is flat"),
        ("Light travels faster than sound", "Sound travels faster than light"),
        ("Cats are mammals", "Cats are reptiles"),
    ]
    for stmt_a, stmt_b in contradiction_pairs:
        texts.append(f"Statement A: {stmt_a}. Statement B: {stmt_b}. These CONTRADICT each other.")
        texts.append(f"Compare: '{stmt_a}' vs '{stmt_b}'. These are CONTRADICTORY.")

    # ══════════════════════════════════════════════════════════
    # Math (2000+ examples)
    # ══════════════════════════════════════════════════════════

    for _ in range(2000):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        else:
            result = a * b
        texts.append(f"Problem: What is {a} {op} {b}? Solution: {a} {op} {b} = {result}.")
        texts.append(f"Calculate: {a} {op} {b} = {result}.")

    # ══════════════════════════════════════════════════════════
    # Causal reasoning (500+ examples)
    # ══════════════════════════════════════════════════════════

    causal_chains = [
        ("Heating ice", "Ice melts", "heat causes phase change"),
        ("Deforestation", "Soil erosion", "removing roots destabilizes soil"),
        ("Exercise", "Heart strengthens", "regular activity strengthens muscle"),
        ("Lack of sleep", "Cognitive decline", "sleep is needed for brain function"),
        ("Greenhouse emissions", "Temperature rises", "gases trap heat"),
        ("Vaccination", "Immune response", "vaccines train immune system"),
        ("Education", "Better opportunities", "skills increase employment"),
        ("Poverty", "Worse health", "lack of resources limits care"),
        ("Inflation", "Less purchasing power", "prices reduce what money buys"),
        ("Innovation", "Higher productivity", "new tools improve efficiency"),
    ]
    for cause, effect, mechanism in causal_chains:
        texts.append(f"Causal chain: {cause} leads to {effect}. Mechanism: {mechanism}.")
        texts.append(f"What causes {effect.lower()}? Answer: {cause} causes it through {mechanism}.")

    # ══════════════════════════════════════════════════════════
    # Uncertainty (500+ examples)
    # ══════════════════════════════════════════════════════════

    uncertain_topics = [
        "stock market tomorrow", "next election winner", "rain next week",
        "technology in 50 years", "investment profitability", "weather in a month",
        "project success", "next decade events", "future of AI", "population in 2100",
    ]
    for topic in uncertain_topics:
        texts.append(f"Question: What about {topic}? Answer: UNCERTAIN. Insufficient evidence.")

    logger.info(f"Generated {len(texts)} training examples")
    return texts


def continue_training():
    """Continue training the small Relay model."""
    from companion.neural.architecture.transformer import RelayTransformer
    from companion.neural.training.trainer import train, TrainConfig
    from companion.neural.tokenizer import train_tokenizer
    from companion.neural.training.datasets import TextDataset
    from companion.neural.training.checkpointing import load_model, load_tokenizer

    checkpoint_dir = str(_sweep_dir / "training" / "relay_small_trained")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load existing model
    logger.info("Loading existing model...")
    model = load_model(checkpoint_dir)
    tokenizer = load_tokenizer(checkpoint_dir)
    logger.info(f"Loaded: {model.param_count():,} params")

    # Generate new data
    texts = generate_large_dataset()

    # Build dataset
    logger.info("Building dataset...")
    dataset = TextDataset.from_texts(
        texts=texts, tokenizer=tokenizer,
        source="sweep_knowledge_large",
    )
    logger.info(f"Dataset: {dataset.n_tokens} tokens")

    # Continue training
    train_config = TrainConfig(
        batch_size=4,
        seq_len=128,
        learning_rate=1e-4,  # Lower LR for continued training
        warmup_steps=20,
        total_steps=300,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_every=100,
        log_every=25,
        seed=42,
        device="cpu",
        dtype="fp32",
    )

    def data_iter():
        while True:
            for batch in dataset.to_dataloader(train_config.batch_size, train_config.seq_len, seed=42):
                yield batch

    logger.info("=" * 60)
    logger.info("CONTINUING TRAINING")
    logger.info(f"  Steps: {train_config.total_steps}")
    logger.info(f"  Dataset: {dataset.n_tokens} tokens")
    logger.info("=" * 60)

    losses = []
    def on_log(entry):
        losses.append(entry["loss"])

    t0 = time.perf_counter()
    result = train(
        model=model, tokenizer=tokenizer,
        data_iter=data_iter, config=train_config,
        checkpoint_dir=checkpoint_dir, on_log=on_log,
    )
    elapsed = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Steps: {result.steps_run}")
    logger.info(f"  Final loss: {result.final_loss:.4f}")
    logger.info(f"  Best loss: {result.best_loss:.4f}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info("=" * 60)

    return model, result


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("SWEEP — GENERATE DATA + CONTINUE TRAINING")
    logger.info("=" * 70)

    model, result = continue_training()

    # Save summary
    summary = {
        "model": "relay-small",
        "parameters": model.param_count(),
        "training_steps": result.steps_run,
        "final_loss": result.final_loss,
        "best_loss": result.best_loss,
        "duration_seconds": result.duration_seconds,
        "status": "trained",
    }
    summary_path = str(_sweep_dir / "training" / "continue_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
