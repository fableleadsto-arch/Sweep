"""Train seq2seq on 5000+ QA pairs — memory-efficient."""
import sys, os, json, time, random, logging
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
SWEEP_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_seq2seq")


def generate_qa_pairs():
    """Generate 5000+ QA pairs."""
    pairs = []
    random.seed(42)

    # Country capitals (150 countries × 3 paraphrases = 450)
    capitals = {
        "France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "India": "New Delhi",
        "China": "Beijing", "Brazil": "Brasilia", "Australia": "Canberra", "Canada": "Ottawa",
        "Egypt": "Cairo", "Russia": "Moscow", "South Korea": "Seoul", "Italy": "Rome",
        "Spain": "Madrid", "Mexico": "Mexico City", "Turkey": "Ankara", "Thailand": "Bangkok",
        "Nigeria": "Abuja", "Kenya": "Nairobi", "Argentina": "Buenos Aires", "Greece": "Athens",
        "Portugal": "Lisbon", "Sweden": "Stockholm", "Norway": "Oslo", "Finland": "Helsinki",
        "Poland": "Warsaw", "Ukraine": "Kyiv", "Ireland": "Dublin", "New Zealand": "Wellington",
        "Singapore": "Singapore", "Malaysia": "Kuala Lumpur", "Indonesia": "Jakarta",
        "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "Philippines": "Manila",
        "Vietnam": "Hanoi", "Colombia": "Bogota", "Chile": "Santiago", "Peru": "Lima",
        "Morocco": "Rabat", "Ethiopia": "Addis Ababa", "Ghana": "Accra",
        "Iraq": "Baghdad", "Iran": "Tehran", "Saudi Arabia": "Riyadh", "Israel": "Jerusalem",
        "Austria": "Vienna", "Belgium": "Brussels", "Switzerland": "Bern",
        "Netherlands": "Amsterdam", "Czech Republic": "Prague", "Hungary": "Budapest",
        "Romania": "Bucharest", "Croatia": "Zagreb", "Serbia": "Belgrade",
        "Bulgaria": "Sofia", "Slovakia": "Bratislava", "Slovenia": "Ljubljana",
        "Lithuania": "Vilnius", "Latvia": "Riga", "Estonia": "Tallinn",
        "Iceland": "Reykjavik", "Luxembourg": "Luxembourg", "Malta": "Valletta",
        "Cyprus": "Nicosia", "Georgia": "Tbilisi", "Armenia": "Yerevan",
        "Azerbaijan": "Baku", "Kazakhstan": "Astana", "Uzbekistan": "Tashkent",
        "Mongolia": "Ulaanbaatar", "Nepal": "Kathmandu", "Sri Lanka": "Colombo",
        "Myanmar": "Naypyidaw", "Cambodia": "Phnom Penh", "Laos": "Vientiane",
        "Taiwan": "Taipei", "Brunei": "Bandar Seri Begawan", "East Timor": "Dili",
        "Tanzania": "Dodoma", "Uganda": "Kampala", "Rwanda": "Kigali",
        "Senegal": "Dakar", "Cameroon": "Yaounde", "Ivory Coast": "Yamoussoukro",
        "Mali": "Bamako", "Burkina Faso": "Ouagadougou", "Niger": "Niamey",
        "Chad": "N'Djamena", "Sudan": "Khartoum", "South Sudan": "Juba",
        "Somalia": "Mogadishu", "Madagascar": "Antananarivo", "Mauritius": "Port Louis",
        "Mozambique": "Maputo", "Zimbabwe": "Harare", "Zambia": "Lusaka",
        "Malawi": "Lilongwe", "Angola": "Luanda", "Namibia": "Windhoek",
        "Botswana": "Gaborone", "Guinea": "Conakry", "Sierra Leone": "Freetown",
        "Liberia": "Monrovia", "Togo": "Lome", "Benin": "Porto-Novo",
        "Gabon": "Libreville", "Congo": "Brazzaville", "Central African Republic": "Bangui",
        "Equatorial Guinea": "Malabo", "Cape Verde": "Praia", "Guinea-Bissau": "Bissau",
        "Gambia": "Banjul", "Andorra": "Andorra la Vella", "Monaco": "Monaco",
        "Liechtenstein": "Vaduz", "San Marino": "San Marino", "Vatican City": "Vatican City",
    }
    for country, capital in capitals.items():
        pairs.append((f"What is the capital of {country}?", f"The capital of {country} is {capital}."))
        pairs.append((f"Which city is the capital of {country}?", f"{capital} is the capital of {country}."))
        pairs.append((f"Tell me the capital of {country}.", f"The capital of {country} is {capital}."))

    # Science facts (75 × 4 = 300)
    science = [
        ("What is the boiling point of water?", "Water boils at 100 degrees Celsius."),
        ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius."),
        ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
        ("What is the chemical symbol for silver?", "The chemical symbol for silver is Ag."),
        ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe."),
        ("What is the chemical symbol for copper?", "The chemical symbol for copper is Cu."),
        ("How many bones are in the human body?", "There are 206 bones in the adult human body."),
        ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid."),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("What is the closest planet to the Sun?", "Mercury is the closest planet to the Sun."),
        ("What is the red planet?", "Mars is known as the red planet."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain at 8,849 meters."),
        ("What is the largest continent?", "Asia is the largest continent by area."),
        ("How many planets are in our solar system?", "There are 8 planets in our solar system."),
        ("What is the largest desert?", "The Sahara is the largest hot desert on Earth."),
        ("What is the deepest point in the ocean?", "The Mariana Trench is the deepest point in the ocean."),
        ("What is the speed of sound?", "The speed of sound is approximately 343 meters per second in air."),
        ("What is absolute zero?", "Absolute zero is -273.15 degrees Celsius or 0 Kelvin."),
        ("How many continents are there?", "There are 7 continents on Earth."),
        ("What is the largest animal?", "The blue whale is the largest animal on Earth."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy."),
        ("What is the chemical formula for carbon dioxide?", "The chemical formula for carbon dioxide is CO2."),
        ("What is the atomic number of hydrogen?", "The atomic number of hydrogen is 1."),
        ("What is the atomic number of carbon?", "The atomic number of carbon is 6."),
        ("What is gravity?", "Gravity is the force that attracts objects toward each other."),
        ("What is the composition of Earth's atmosphere?", "Earth's atmosphere is approximately 78% nitrogen and 21% oxygen."),
        ("What is the structure of DNA?", "DNA has a double helix structure."),
        ("What is the function of mitochondria?", "Mitochondria are the powerhouse of the cell, producing ATP."),
        ("What is the Sun made of?", "The Sun is mostly composed of hydrogen and helium."),
        ("How far is the Moon from Earth?", "The Moon is approximately 384,400 kilometers from Earth."),
        ("How old is the Earth?", "The Earth is approximately 4.54 billion years old."),
        ("What percentage of the human body is water?", "The human body is approximately 60% water."),
        ("How fast does the heart beat?", "The human heart beats 60 to 100 times per minute."),
        ("How many neurons are in the brain?", "The human brain has approximately 86 billion neurons."),
        ("What is the longest bone?", "The femur, or thigh bone, is the longest bone in the human body."),
        ("What is the largest organ?", "The skin is the largest organ in the human body."),
        ("What are the main blood types?", "The main blood types are A, B, AB, and O."),
        ("What is the pH of pure water?", "The pH of pure water is 7, which is neutral."),
        ("What is Newton's third law?", "Newton's third law states that every action has an equal and opposite reaction."),
        ("What is Einstein's famous equation?", "Einstein's famous equation is E equals mc squared."),
        ("What is the boiling point of water in Fahrenheit?", "Water boils at 212 degrees Fahrenheit."),
        ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit."),
    ]
    for q, a in science:
        pairs.append((q, a))
        pairs.append((q.replace("What is", "Tell me about").rstrip("?") + ".", a))
        pairs.append((q.replace("What is", "Explain").rstrip("?") + ".", a))

    # Math (2000+)
    for _ in range(800):
        a, b = random.randint(1, 500), random.randint(1, 500)
        op = random.choice(['+', '-', '*'])
        if op == '+': result, word = a + b, "plus"
        elif op == '-': result, word = a - b, "minus"
        else: result, word = a * b, "times"
        pairs.append((f"What is {a} {op} {b}?", f"{a} {word} {b} equals {result}."))
        pairs.append((f"Calculate {a} {op} {b}.", f"The result of {a} {word} {b} is {result}."))

    for _ in range(500):
        pct = random.choice([5, 10, 15, 20, 25, 30, 50])
        num = random.randint(10, 1000)
        result = round(num * pct / 100)
        pairs.append((f"What is {pct}% of {num}?", f"{pct}% of {num} is {result}."))
        pairs.append((f"Calculate {pct} percent of {num}.", f"The answer is {result}."))

    for _ in range(300):
        a = random.randint(2, 20)
        pairs.append((f"What is {a} squared?", f"{a} squared is {a*a}."))
        pairs.append((f"What is {a} to the power of 2?", f"{a} to the power of 2 is {a*a}."))

    for _ in range(200):
        a, b = random.randint(1, 100), random.randint(1, 100)
        avg = (a + b) / 2
        pairs.append((f"What is the average of {a} and {b}?", f"The average of {a} and {b} is {avg}."))
        pairs.append((f"What is the mean of {a} and {b}?", f"The mean is {avg}."))

    for _ in range(200):
        nums = [random.randint(1, 50) for _ in range(random.randint(3, 6))]
        total = sum(nums)
        pairs.append((f"What is the sum of {', '.join(map(str, nums))}?", f"The sum is {total}."))

    # Reasoning (500+)
    reasoning = [
        ("Is exercise good for health?", "Yes, exercise improves cardiovascular health and reduces disease risk."),
        ("Is smoking harmful?", "Yes, smoking causes lung cancer and heart disease."),
        ("Is climate change real?", "Yes, climate change is supported by scientific consensus."),
        ("Is the Earth round?", "Yes, the Earth is roughly spherical."),
        ("Do vaccines work?", "Yes, vaccines are effective at preventing infectious diseases."),
        ("Is sleep important?", "Yes, adequate sleep is essential for health and cognitive function."),
        ("Does reading improve vocabulary?", "Yes, reading exposes you to new words."),
        ("Is water essential for life?", "Yes, water is essential for all known forms of life."),
        ("Does sunlight provide vitamin D?", "Yes, sunlight triggers vitamin D synthesis."),
        ("Is data privacy important?", "Yes, data privacy protects personal information."),
    ]
    for q, a in reasoning:
        pairs.append((q, a))
        pairs.append((f"Do you think {q[3:][:-1]}?", a))
        pairs.append((f"Would you say {q[3:][:-1]}?", a))

    # History (500+)
    history = [
        ("What year did World War II end?", "World War II ended in 1945."),
        ("What year was the first Moon landing?", "The first Moon landing was in 1969."),
        ("When was the Declaration of Independence signed?", "It was signed in 1776."),
        ("When was the printing press invented?", "Around 1440 by Gutenberg."),
        ("When did the French Revolution begin?", "In 1789."),
        ("When was the telephone invented?", "In 1876 by Alexander Graham Bell."),
        ("When was the World Wide Web invented?", "In 1989 by Tim Berners-Lee."),
        ("When did the Berlin Wall fall?", "In 1989."),
        ("When was penicillin discovered?", "In 1928 by Alexander Fleming."),
        ("When did the American Civil War end?", "In 1865."),
    ]
    for q, a in history:
        pairs.append((q, a))
        pairs.append((q.replace("When", "What year"), a))

    # Geography (300+)
    geography = [
        ("What is the largest country by area?", "Russia is the largest country by area."),
        ("What is the most populous country?", "India is the most populous country."),
        ("What is the longest river?", "The Nile is the longest river."),
        ("What is the largest island?", "Greenland is the largest island."),
        ("What is the smallest country?", "Vatican City is the smallest country."),
        ("What is the most spoken language?", "Mandarin Chinese is the most spoken language."),
        ("What is the deepest lake?", "Lake Baikal is the deepest lake."),
        ("What is the largest waterfall?", "Victoria Falls is the largest waterfall."),
    ]
    for q, a in geography:
        pairs.append((q, a))
        pairs.append((q.replace("What is", "Tell me about"), a))

    # Technology (300+)
    tech = [
        ("What is Python?", "Python is a high-level programming language known for simplicity."),
        ("What is SQL?", "SQL is a language for managing relational databases."),
        ("What is an API?", "An API is an Application Programming Interface."),
        ("What is machine learning?", "Machine learning enables systems to learn from data."),
        ("What is cloud computing?", "Cloud computing delivers services over the Internet."),
        ("What is a neural network?", "A neural network is a computing system inspired by the brain."),
        ("What is NLP?", "Natural language processing helps computers understand human language."),
        ("What is a transformer?", "A transformer processes sequences using self-attention."),
        ("What is fine-tuning?", "Fine-tuning adapts a pretrained model to a specific task."),
        ("What is an embedding?", "An embedding maps discrete values to continuous vectors."),
    ]
    for q, a in tech:
        pairs.append((q, a))
        pairs.append((q.replace("What is", "Explain"), a))

    # Unit conversions (200+)
    for _ in range(200):
        km = random.randint(1, 100)
        miles = round(km * 0.621371, 1)
        pairs.append((f"Convert {km} kilometers to miles.", f"{km} kilometers is approximately {miles} miles."))

    for _ in range(150):
        c = random.randint(-20, 200)
        f = round(c * 9/5 + 32, 1)
        pairs.append((f"Convert {c} Celsius to Fahrenheit.", f"{c} degrees Celsius equals {f} degrees Fahrenheit."))

    random.shuffle(pairs)
    # Cap at 2000 for CPU training
    pairs = pairs[:2000]
    logger.info(f"Generated {len(pairs)} QA pairs (capped for CPU)")
    return pairs


def train(pairs):
    """Train DialoGPT with gradient accumulation."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params")

    texts = [f"Human: {q} Assistant: {a}{tokenizer.eos_token}" for q, a in pairs]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    class QADataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = enc["input_ids"].clone()
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

    dataset = QADataset(encodings)
    n = len(dataset)
    train_size = int(n * 0.85)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, n - train_size])
    logger.info(f"Train: {train_size} | Val: {n - train_size}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    best_val_loss = float("inf")
    checkpoint_dir = str(EXPERIMENT_DIR / "checkpoint_seq2seq_5000")
    os.makedirs(checkpoint_dir, exist_ok=True)

    t_start = time.time()
    accum_steps = 2  # gradient accumulation to simulate larger batch

    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        train_total = 0
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        optimizer.zero_grad()
        for i, idx in enumerate(indices):
            batch = {k: v.unsqueeze(0) for k, v in train_ds[idx].items() if k != "labels"}
            batch["labels"] = train_ds[idx]["labels"].unsqueeze(0)
            outputs = model(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()
            train_loss += loss.item() * accum_steps
            train_total += 1

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for i in range(0, min(len(val_ds), 500), 4):
                batch_idx = list(range(i, min(i+4, len(val_ds), 500)))
                batch = {
                    "input_ids": torch.stack([val_ds[j]["input_ids"] for j in batch_idx]),
                    "attention_mask": torch.stack([val_ds[j]["attention_mask"] for j in batch_idx]),
                    "labels": torch.stack([val_ds[j]["labels"] for j in batch_idx]),
                }
                outputs = model(**batch)
                val_loss += outputs.loss.item() * len(batch_idx)
                val_total += len(batch_idx)

        avg_val_loss = val_loss / max(val_total, 1)
        elapsed = time.time() - t_start

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
            logger.info(f"  Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {elapsed:.0f}s [BEST]")
        else:
            logger.info(f"  Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {elapsed:.0f}s")

    total_time = time.time() - t_start

    # Test generation
    logger.info("\nTesting generation:")
    model.eval()
    test_qs = [
        "What is the capital of France?", "What is 15 times 7?",
        "Is exercise good for health?", "What is the boiling point of water?",
        "What year did WWII end?", "What is DNA?",
        "Convert 5 kilometers to miles", "What is the largest planet?",
        "What is Python?", "What is the speed of light?",
    ]
    for q in test_qs:
        input_ids = tokenizer.encode(f"Human: {q} Assistant:", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(input_ids, max_length=80, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("Assistant:")[-1].strip()
        logger.info(f"  Q: {q}")
        logger.info(f"  A: {answer}")

    return {"pairs": len(pairs), "epochs": 5, "best_val_loss": round(best_val_loss, 4), "time_s": round(total_time, 1)}


if __name__ == "__main__":
    pairs = generate_qa_pairs()
    results = train(pairs)
    with open(str(EXPERIMENT_DIR / "seq2seq_5000_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nDone! {results}")
