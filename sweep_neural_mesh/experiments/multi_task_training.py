"""
Sweep Multi-Task Training Session — 5 Training Experiments

1. Logic & Reasoning (deduction, induction, syllogisms, modus ponens)
2. Recognition (entity extraction, feature detection)
3. Mathematics (arithmetic, algebra, word problems)
4. Basic Tasks (classification, extraction, formatting)
5. Advanced Tasks (multi-step reasoning, hypothesis, contradiction)

All training uses REAL gradient updates on CPU.
"""
import sys
import os
import json
import time
import hashlib
import random
import logging
from pathlib import Path
from collections import Counter

# Setup paths
EXPERIMENT_DIR = Path(__file__).parent
SWEEP_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(SWEEP_DIR.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_multi")

# ════════════════════════════════════════════════════════════════════
# SHARED MODEL
# ════════════════════════════════════════════════════════════════════

class TaskClassifier(nn.Module):
    """Shared MLP architecture for all tasks.
    
    Sweep-original design: frozen embeddings + trainable classifier.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)


class TextPairDataset(Dataset):
    """Dataset for text pair classification."""
    def __init__(self, data: list[dict], embedder, label_map: dict):
        self.data = data
        self.embedder = embedder
        self.label_map = label_map
        self.cache = {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        key = f"{item['text']}|||{item.get('context', '')}"
        if key not in self.cache:
            t = item['text']
            c = item.get('context', '')
            if c:
                emb1 = self.embedder.embed(t)
                emb2 = self.embedder.embed(c)
                combined = np.array(emb1.vector) + np.array(emb2.vector)
            else:
                combined = np.array(self.embedder.embed(t).vector)
            self.cache[key] = torch.tensor(combined, dtype=torch.float32)
        label = self.label_map[item['label']]
        return self.cache[key], label


# ════════════════════════════════════════════════════════════════════
# DATASET GENERATORS
# ════════════════════════════════════════════════════════════════════

def generate_logic_data(seed=42) -> list[dict]:
    """Logic & Reasoning: deduction, induction, syllogisms, modus ponens/tollens."""
    rng = random.Random(seed)
    data = []
    
    # DEDUCTION
    deduction_pairs = [
        ("If it rains, the ground gets wet. It rained. What happens?", "The ground gets wet", "VALID"),
        ("If it rains, the ground gets wet. It did not rain. What happens?", "Nothing specific from this rule", "VALID"),
        ("All mammals are animals. Dogs are mammals. Are dogs animals?", "Yes, dogs are animals", "VALID"),
        ("If P then Q. P is true. Is Q true?", "Yes, Q must be true", "VALID"),
        ("If temperature drops below zero, water freezes. Temperature is -5. What happens?", "Water freezes", "VALID"),
        ("If the battery is dead, the phone won't turn on. Phone won't turn on. Is the battery dead?", "Not necessarily, could be other causes", "INVALID"),
        ("If X implies Y, and Y implies Z, does X imply Z?", "Yes, by transitivity", "VALID"),
        ("If all birds fly, and penguins are birds, do penguins fly?", "The premise is false, but logically yes", "VALID"),
        ("If A is faster than B, and B is faster than C, is A faster than C?", "Yes, by transitivity", "VALID"),
        ("If the alarm sounds, there is a fire. There is no fire. Will the alarm sound?", "No, not from this rule", "VALID"),
        ("If a number is even, it is divisible by 2. 6 is even. Is 6 divisible by 2?", "Yes", "VALID"),
        ("If all roses are flowers, and all flowers need water, do roses need water?", "Yes, by syllogism", "VALID"),
        ("If prices rise, demand falls. Prices rose. What happens to demand?", "Demand falls", "VALID"),
        ("If X is taller than Y, X can reach higher shelves. X is taller than Y. Can X reach higher?", "Yes", "VALID"),
        ("If studying improves grades, and Alice studies, what happens to Alice's grades?", "Alice's grades improve", "VALID"),
    ]
    for q, a, lbl in deduction_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # INDUCTION
    induction_pairs = [
        ("Sequence: 2, 4, 6, 8, what comes next?", "10 (arithmetic progression)", "VALID"),
        ("Pattern: 1, 1, 2, 3, 5, what comes next?", "8 (Fibonacci sequence)", "VALID"),
        ("Numbers: 3, 6, 12, 24, what comes next?", "48 (doubling)", "VALID"),
        ("Sequence: 1, 4, 9, 16, what comes next?", "25 (perfect squares)", "VALID"),
        ("Pattern: A, B, A, B, what comes next?", "A (alternating pattern)", "VALID"),
        ("Numbers: 10, 8, 6, 4, what comes next?", "2 (decreasing by 2)", "VALID"),
        ("Sequence: 1, 3, 6, 10, what comes next?", "15 (triangular numbers)", "VALID"),
        ("Pattern: Monday, Tuesday, Wednesday, what comes next?", "Thursday (days of week)", "VALID"),
        ("Numbers: 1, 8, 27, 64, what comes next?", "125 (perfect cubes)", "VALID"),
        ("Sequence: 2, 6, 12, 20, what comes next?", "30 (n squared plus n)", "VALID"),
    ]
    for q, a, lbl in induction_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # SYLLOGISMS
    syllogism_pairs = [
        ("All cats are mammals. All mammals are animals. Is a cat an animal?", "Yes, by transitive syllogism", "VALID"),
        ("No reptiles produce milk. Sharks are reptiles. Do sharks produce milk?", "No, sharks do not produce milk", "VALID"),
        ("All teachers are educated. Sarah is a teacher. Is Sarah educated?", "Yes", "VALID"),
        ("Some doctors play golf. Dr. Smith plays golf. Is Dr. Smith a doctor?", "Cannot conclude, insufficient evidence", "INVALID"),
        ("All dogs bark. Rex is a dog. Does Rex bark?", "Yes", "VALID"),
        ("No fish can fly. A salmon is a fish. Can a salmon fly?", "No", "VALID"),
        ("All squares have four sides. This shape is a square. How many sides?", "Four sides", "VALID"),
        ("All planets orbit stars. Earth is a planet. Does Earth orbit a star?", "Yes, the Sun", "VALID"),
    ]
    for q, a, lbl in syllogism_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # MODUS PONENS / TOLLENS
    mp_mt_pairs = [
        ("If it is a cat, it is a mammal. It is a cat. Is it a mammal?", "Yes (modus ponens)", "VALID"),
        ("If it is a cat, it is a mammal. It is not a mammal. Is it a cat?", "No (modus tollens)", "VALID"),
        ("If the switch is on, the light is on. The light is off. Is the switch on?", "No (modus tollens)", "VALID"),
        ("If you study, you pass. You passed. Did you study?", "Not necessarily (affirming consequent)", "INVALID"),
        ("If it snows, schools close. Schools are open. Did it snow?", "No (modus tollens)", "VALID"),
    ]
    for q, a, lbl in mp_mt_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # INVALID REASONING (labeled INVALID)
    invalid_pairs = [
        ("All cats are animals. All dogs are animals. Are all cats dogs?", "No, that's the undistributed middle fallacy", "INVALID"),
        ("If A then B. B is true. Therefore A is true.", "No, that's affirming the consequent", "INVALID"),
        ("Some X are Y. Z is X. Therefore Z is Y.", "No, that's the fallacy of the undistributed middle", "INVALID"),
        ("It rained and the ground is wet. The ground is wet. Therefore it rained.", "No, the ground could be wet for other reasons", "INVALID"),
        ("All A are B. C is not A. Therefore C is not B.", "No, that's denying the antecedent", "INVALID"),
        ("Every time I eat spicy food, I get a headache. I have a headache. Therefore I ate spicy food.", "No, affirming the consequent", "INVALID"),
        ("The sun rose today. Therefore the sun will rise tomorrow.", "Inductive generalization, not deductive certainty", "INVALID"),
        ("I saw a black cat. Therefore all cats are black.", "No, hasty generalization", "INVALID"),
        ("If it rains, the streets get wet. The streets are wet. Therefore it rained.", "No, could be a sprinkler", "INVALID"),
        ("All roses I have seen are red. Therefore all roses are red.", "No, hasty generalization", "INVALID"),
    ]
    for q, a, lbl in invalid_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    rng.shuffle(data)
    return data


def generate_recognition_data(seed=42) -> list[dict]:
    """Recognition: entity extraction, feature detection, pattern matching."""
    rng = random.Random(seed)
    data = []
    
    # PERSON extraction
    person_pairs = [
        ("John Smith visited Paris on March 12", "PERSON: John Smith", "EXTRACT"),
        ("Dr. Sarah Chen published a paper on AI", "PERSON: Dr. Sarah Chen", "EXTRACT"),
        ("The CEO of Apple is Tim Cook", "PERSON: Tim Cook", "EXTRACT"),
        ("Albert Einstein developed the theory of relativity", "PERSON: Albert Einstein", "EXTRACT"),
        ("Marie Curie won the Nobel Prize", "PERSON: Marie Curie", "EXTRACT"),
        ("Barack Obama served as president", "PERSON: Barack Obama", "EXTRACT"),
        ("The team was led by Coach Johnson", "PERSON: Coach Johnson", "EXTRACT"),
        ("Mary and James went to the store", "PERSON: Mary, James", "EXTRACT"),
    ]
    for q, a, lbl in person_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # LOCATION extraction
    location_pairs = [
        ("The meeting is in New York City", "LOCATION: New York City", "EXTRACT"),
        ("Paris is the capital of France", "LOCATION: Paris, France", "EXTRACT"),
        ("The flight lands at Heathrow Airport", "LOCATION: Heathrow Airport", "EXTRACT"),
        ("They traveled from Tokyo to Osaka", "LOCATION: Tokyo, Osaka", "EXTRACT"),
        ("The company is based in Silicon Valley", "LOCATION: Silicon Valley", "EXTRACT"),
        ("The river flows through London", "LOCATION: London", "EXTRACT"),
        ("The conference is held in Berlin, Germany", "LOCATION: Berlin, Germany", "EXTRACT"),
        ("She grew up in a small town in Iowa", "LOCATION: Iowa", "EXTRACT"),
    ]
    for q, a, lbl in location_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # DATE extraction
    date_pairs = [
        ("The event is on January 15, 2025", "DATE: January 15, 2025", "EXTRACT"),
        ("World War II ended in 1945", "DATE: 1945", "EXTRACT"),
        ("The deadline is next Friday", "DATE: next Friday", "EXTRACT"),
        ("Meeting at 3 PM on Tuesday", "DATE: Tuesday 3 PM", "EXTRACT"),
        ("The contract expires December 31", "DATE: December 31", "EXTRACT"),
        ("She was born on July 4, 1990", "DATE: July 4, 1990", "EXTRACT"),
    ]
    for q, a, lbl in date_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # ORGANIZATION extraction
    org_pairs = [
        ("Google announced a new product", "ORG: Google", "EXTRACT"),
        ("The WHO issued a health warning", "ORG: WHO", "EXTRACT"),
        ("NASA launched a new rocket", "ORG: NASA", "EXTRACT"),
        ("Microsoft and Apple reported earnings", "ORG: Microsoft, Apple", "EXTRACT"),
        ("The United Nations held a summit", "ORG: United Nations", "EXTRACT"),
        ("Tesla stock rose 5 percent", "ORG: Tesla", "EXTRACT"),
    ]
    for q, a, lbl in org_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # NO ENTITY (labeled SKIP)
    skip_pairs = [
        ("The weather is nice today", "No entities found", "SKIP"),
        ("It is raining outside", "No entities found", "SKIP"),
        ("The color blue is calming", "No entities found", "SKIP"),
        ("Mathematics is interesting", "No entities found", "SKIP"),
        ("The cat sat on the mat", "No entities found", "SKIP"),
        ("Running is good exercise", "No entities found", "SKIP"),
    ]
    for q, a, lbl in skip_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    rng.shuffle(data)
    return data


def generate_math_data(seed=42) -> list[dict]:
    """Mathematics: arithmetic, algebra, word problems, conversions."""
    rng = random.Random(seed)
    data = []
    
    # ARITHMETIC
    for _ in range(40):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        op = rng.choice(['+', '-', '*'])
        if op == '+': result = a + b
        elif op == '-': result = a - b
        else: result = a * b
        data.append({"text": f"What is {a} {op} {b}?", "context": f"{result}", "label": "COMPUTE"})
    
    # PERCENTAGES
    for _ in range(20):
        pct = rng.choice([10, 15, 20, 25, 30, 50, 75])
        num = rng.choice([100, 200, 300, 400, 500, 1000])
        result = int(num * pct / 100)
        data.append({"text": f"What is {pct}% of {num}?", "context": f"{result}", "label": "COMPUTE"})
    
    # ALGEBRA
    algebra_pairs = [
        ("Solve 2x + 6 = 12", "x = 3", "COMPUTE"),
        ("Solve x - 5 = 10", "x = 15", "COMPUTE"),
        ("If 3x = 15, what is x?", "x = 5", "COMPUTE"),
        ("Solve 2x = 14", "x = 7", "COMPUTE"),
        ("If x + 8 = 20, what is x?", "x = 12", "COMPUTE"),
        ("Solve 4x - 8 = 12", "x = 5", "COMPUTE"),
        ("If 5x = 25, what is x?", "x = 5", "COMPUTE"),
        ("Solve x/3 = 9", "x = 27", "COMPUTE"),
    ]
    for q, a, lbl in algebra_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # WORD PROBLEMS
    word_pairs = [
        ("A store has 50 apples. They sell 20. How many left?", "30 apples", "COMPUTE"),
        ("Train travels 60 mph for 2 hours. How far?", "120 miles", "COMPUTE"),
        ("If 5 workers build a wall in 10 days, how long for 10 workers?", "5 days", "COMPUTE"),
        ("A pizza is cut into 8 slices. 3 are eaten. How many remain?", "5 slices", "COMPUTE"),
        ("You buy 3 books at $12 each. Total cost?", "$36", "COMPUTE"),
        ("A car goes 300 miles on 10 gallons. MPG?", "30 MPG", "COMPUTE"),
        ("If you save $50 per month, how much in a year?", "$600", "COMPUTE"),
        ("A rectangle is 5 by 8. Area?", "40 square units", "COMPUTE"),
    ]
    for q, a, lbl in word_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # UNIT CONVERSION
    conv_pairs = [
        ("Convert 5 kilometers to miles", "3.1069 miles", "COMPUTE"),
        ("Convert 100 Celsius to Fahrenheit", "212 Fahrenheit", "COMPUTE"),
        ("Convert 10 pounds to kilograms", "4.5359 kg", "COMPUTE"),
        ("Convert 12 inches to centimeters", "30.48 cm", "COMPUTE"),
        ("Convert 1 gallon to liters", "3.7854 liters", "COMPUTE"),
        ("Convert 100 Fahrenheit to Celsius", "37.78 Celsius", "COMPUTE"),
    ]
    for q, a, lbl in conv_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # INVALID MATH (labeled INVALID)
    invalid_math = [
        ("What is 5 + 3 * 2?", "22 if left-to-right, 11 if order of operations", "INVALID"),
        ("Divide by zero: 10/0", "Undefined, division by zero", "INVALID"),
        ("What is the square root of -1?", "Not a real number", "INVALID"),
        ("If x + y = 10 and x - y = 2, what is x?", "x = 6, but requires system of equations", "INVALID"),
    ]
    for q, a, lbl in invalid_math:
        data.append({"text": q, "context": a, "label": lbl})
    
    rng.shuffle(data)
    return data


def generate_basic_data(seed=42) -> list[dict]:
    """Basic tasks: classification, extraction, formatting."""
    rng = random.Random(seed)
    data = []
    
    # SENTIMENT
    sentiment_pairs = [
        ("This product is amazing and I love it", "POSITIVE", "CLASSIFY"),
        ("Terrible experience, would not recommend", "NEGATIVE", "CLASSIFY"),
        ("It's okay, nothing special", "NEUTRAL", "CLASSIFY"),
        ("Best purchase I've ever made", "POSITIVE", "CLASSIFY"),
        ("Worst customer service ever", "NEGATIVE", "CLASSIFY"),
        ("The weather is cloudy today", "NEUTRAL", "CLASSIFY"),
        ("I'm so happy with the results", "POSITIVE", "CLASSIFY"),
        ("The food was disgusting and cold", "NEGATIVE", "CLASSIFY"),
        ("The movie was average", "NEUTRAL", "CLASSIFY"),
        ("Outstanding performance by the team", "POSITIVE", "CLASSIFY"),
        ("The service was slow and rude", "NEGATIVE", "CLASSIFY"),
        ("It works as expected", "NEUTRAL", "CLASSIFY"),
    ]
    for q, a, lbl in sentiment_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # TOPIC CLASSIFICATION
    topic_pairs = [
        ("The stock market rose 2 percent today", "FINANCE", "CLASSIFY"),
        ("New research shows exercise benefits health", "HEALTH", "CLASSIFY"),
        ("The team won the championship game", "SPORTS", "CLASSIFY"),
        ("Scientists discover new species in Amazon", "SCIENCE", "CLASSIFY"),
        ("The president signed a new trade agreement", "POLITICS", "CLASSIFY"),
        ("Apple releases new iPhone model", "TECHNOLOGY", "CLASSIFY"),
        ("The concert sold out in minutes", "ENTERTAINMENT", "CLASSIFY"),
        ("New study on climate change impacts", "SCIENCE", "CLASSIFY"),
        ("Housing prices continue to rise", "FINANCE", "CLASSIFY"),
        ("Olympic athletes prepare for competition", "SPORTS", "CLASSIFY"),
    ]
    for q, a, lbl in topic_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # URGENCY
    urgency_pairs = [
        ("System failure detected, servers down", "URGENT", "CLASSIFY"),
        ("Quarterly report is due next week", "MODERATE", "CLASSIFY"),
        ("Please update your profile when convenient", "LOW", "CLASSIFY"),
        ("Fire alarm activated in building", "URGENT", "CLASSIFY"),
        ("Meeting scheduled for Friday", "MODERATE", "CLASSIFY"),
        ("Consider reviewing the documentation", "LOW", "CLASSIFY"),
        ("Security breach detected in network", "URGENT", "CLASSIFY"),
        ("Budget review meeting tomorrow", "MODERATE", "CLASSIFY"),
        ("New guidelines posted on intranet", "LOW", "CLASSIFY"),
    ]
    for q, a, lbl in urgency_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # SUMMARIZATION (extractive)
    summary_pairs = [
        ("The company reported record earnings of $5 billion in Q3, driven by strong sales in the Asian market and new product launches", "Record Q3 earnings of $5B from Asian sales and new products", "EXTRACT"),
        ("Researchers at MIT developed a new AI algorithm that can predict protein structures with 95% accuracy, potentially revolutionizing drug discovery", "MIT AI predicts protein structures at 95% accuracy for drug discovery", "EXTRACT"),
        ("The city council approved a $2 million budget for road repairs, affecting 15 neighborhoods over the next fiscal year", "Council approved $2M for road repairs in 15 neighborhoods", "EXTRACT"),
    ]
    for q, a, lbl in summary_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    rng.shuffle(data)
    return data


def generate_advanced_data(seed=42) -> list[dict]:
    """Advanced tasks: multi-step reasoning, hypothesis, contradiction, evidence."""
    rng = random.Random(seed)
    data = []
    
    # CONTRADICTION DETECTION
    contradiction_pairs = [
        ("Source A: The meeting is at 3 PM. Source B: The meeting is at 4 PM.", "CONTRADICTION", "DETECT"),
        ("Source A: The drug reduces symptoms. Source B: The drug has no effect on symptoms.", "CONTRADICTION", "DETECT"),
        ("Source A: Revenue increased 15%. Source B: Revenue decreased 15%.", "CONTRADICTION", "DETECT"),
        ("Source A: The Earth is round. Source B: The Earth is an oblate spheroid.", "CONSISTENT", "DETECT"),
        ("Source A: Exercise improves health. Source B: Physical activity benefits wellbeing.", "CONSISTENT", "DETECT"),
        ("Source A: The patient has fever. Source B: The patient has no fever.", "CONTRADICTION", "DETECT"),
        ("Source A: Company X is profitable. Source B: Company X reported losses.", "CONTRADICTION", "DETECT"),
        ("Source A: Water boils at 100C. Source B: Water freezes at 0C.", "CONSISTENT", "DETECT"),
        ("Source A: All students passed. Source B: Some students failed.", "CONTRADICTION", "DETECT"),
        ("Source A: Stock price rose. Source B: Stock price fell.", "CONTRADICTION", "DETECT"),
        ("Source A: The bridge is safe. Source B: The bridge has structural issues.", "CONTRADICTION", "DETECT"),
        ("Source A: Coffee contains caffeine. Source B: Tea contains caffeine.", "CONSISTENT", "DETECT"),
    ]
    for q, a, lbl in contradiction_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # EVIDENCE CLASSIFICATION
    evidence_pairs = [
        ("Claim: Vaccines are safe. Evidence: Clinical trials show 95% efficacy with mild side effects.", "SUPPORTS", "EVALUATE"),
        ("Claim: Vaccines are safe. Evidence: An anonymous blog claims vaccines cause harm.", "REFUTES", "EVALUATE"),
        ("Claim: Exercise improves health. Evidence: The stock market is up today.", "NEUTRAL", "EVALUATE"),
        ("Claim: Climate change is real. Evidence: Global temperatures rose 1.1C since 1900.", "SUPPORTS", "EVALUATE"),
        ("Claim: Smoking is harmless. Evidence: Studies link smoking to lung cancer.", "REFUTES", "EVALUATE"),
        ("Claim: Reading improves vocabulary. Evidence: No relevant studies found.", "NEUTRAL", "EVALUATE"),
        ("Claim: Democracy is good. Evidence: Democratic countries have higher GDP.", "SUPPORTS", "EVALUATE"),
        ("Claim: Nuclear energy is dangerous. Evidence: Modern reactors have zero meltdown history.", "REFUTES", "EVALUATE"),
        ("Claim: AI will replace all jobs. Evidence: AI has automated some routine tasks.", "NEUTRAL", "EVALUATE"),
        ("Claim: Sugar is addictive. Evidence: Brain scans show sugar activates reward pathways.", "SUPPORTS", "EVALUATE"),
    ]
    for q, a, lbl in evidence_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # MULTI-STEP REASONING
    multistep_pairs = [
        ("Step 1: All dogs are animals. Step 2: Rex is a dog. Step 3: Rex is an animal?", "Yes, correct chain", "VALID"),
        ("Step 1: If it rains, ground is wet. Step 2: Ground is not wet. Step 3: Did it rain?", "No, by modus tollens", "VALID"),
        ("Step 1: A > B. Step 2: B > C. Step 3: Is A > C?", "Yes, by transitivity", "VALID"),
        ("Step 1: All fish live in water. Step 2: Whales live in water. Step 3: Are whales fish?", "No, the premise doesn't support this conclusion", "INVALID"),
        ("Step 1: If X then Y. Step 2: Y is true. Step 3: Is X true?", "No, affirming the consequent is invalid", "INVALID"),
    ]
    for q, a, lbl in multistep_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    # HYPOTHESIS GENERATION
    hypothesis_pairs = [
        ("Observation: The patient has fever and cough. What are possible causes?", "Flu, cold, pneumonia, COVID-19", "GENERATE"),
        ("Observation: Stock price dropped 20% after earnings. Why?", "Poor earnings, market conditions, sector decline", "GENERATE"),
        ("Observation: The bridge collapsed. What could cause this?", "Structural failure, overload, material fatigue", "GENERATE"),
        ("Observation: Plant leaves are turning yellow. Why?", "Nutrient deficiency, overwatering, disease", "GENERATE"),
    ]
    for q, a, lbl in hypothesis_pairs:
        data.append({"text": q, "context": a, "label": lbl})
    
    rng.shuffle(data)
    return data


# ════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════

def split_data(data: list[dict], train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n = len(data)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return data[:t], data[t:t+v], data[t+v:]


def train_one_task(
    task_name: str,
    data: list[dict],
    label_names: list[str],
    embedder,
    checkpoint_dir: str,
    epochs: int = 20,
    seed: int = 42,
) -> dict:
    """Train one task classifier. Returns metrics."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {task_name}")
    logger.info(f"{'='*60}")
    
    # Split
    train_data, val_data, test_data = split_data(data, seed=seed)
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    logger.info(f"Classes: {label_names}")
    logger.info(f"Distribution: {dict(Counter(d['label'] for d in train_data))}")
    
    # Label map
    label_map = {name: i for i, name in enumerate(label_names)}
    
    # Create datasets
    train_ds = TextPairDataset(train_data, embedder, label_map)
    val_ds = TextPairDataset(val_data, embedder, label_map)
    test_ds = TextPairDataset(test_data, embedder, label_map)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Pre-compute embeddings
    logger.info("Pre-computing embeddings...")
    t0 = time.time()
    for i in range(len(train_ds)):
        _ = train_ds[i]
    for i in range(len(val_ds)):
        _ = val_ds[i]
    for i in range(len(test_ds)):
        _ = test_ds[i]
    logger.info(f"Embeddings computed in {time.time()-t0:.1f}s")
    
    # Detect input dim
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    # Create model
    model = TaskClassifier(input_dim=input_dim, hidden_dim=128, num_classes=len(label_names))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,}")
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    history = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.tolist())
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        elapsed = time.time() - t_start
        
        info = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4),
            "val_acc": round(val_acc, 4), "elapsed": round(elapsed, 1),
        }
        history.append(info)
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:2d} | Loss: {train_loss:.4f} -> {val_loss:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "val_f1": val_f1,
                "task": task_name, "label_names": label_names,
                "input_dim": input_dim,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
    
    total_time = time.time() - t_start
    
    # Load best and evaluate on test set
    ckpt = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.tolist())
    
    from sklearn.metrics import classification_report
    test_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    try:
        report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True)
    except ValueError:
        # Test set may have fewer classes than training set
        present_labels = sorted(set(all_labels + all_preds))
        present_names = [label_names[i] for i in present_labels if i < len(label_names)]
        report = classification_report(all_labels, all_preds, labels=present_labels, target_names=present_names, zero_division=0, output_dict=True)
    
    logger.info(f"\n  FINAL: Test F1={test_f1:.4f} | Test Acc={test_acc:.4f}")
    logger.info(f"  Best val F1: {best_val_f1:.4f} | Time: {total_time:.1f}s")
    
    return {
        "task": task_name,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "label_names": label_names,
        "total_params": total_params,
        "input_dim": input_dim,
        "epochs": len(history),
        "total_time_s": round(total_time, 1),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_f1": round(best_val_f1, 4),
        "test_f1": round(test_f1, 4),
        "test_acc": round(test_acc, 4),
        "train_loss_start": history[0]["train_loss"],
        "train_loss_end": history[-1]["train_loss"],
        "history": history,
        "checkpoint": os.path.join(checkpoint_dir, "best_model.pt"),
        "classification_report": report,
    }


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("SWEEP MULTI-TASK TRAINING SESSION")
    logger.info("=" * 70)
    
    # Load embedder
    from neurons.semantic_embeddings import SemanticEmbedder
    embedder = SemanticEmbedder()
    logger.info(f"Embedder loaded: {embedder.backend}")
    
    # Generate all datasets
    logger.info("\n--- Generating Datasets ---")
    logic_data = generate_logic_data(seed=42)
    recognition_data = generate_recognition_data(seed=42)
    math_data = generate_math_data(seed=42)
    basic_data = generate_basic_data(seed=42)
    advanced_data = generate_advanced_data(seed=42)
    
    logger.info(f"Logic: {len(logic_data)} samples")
    logger.info(f"Recognition: {len(recognition_data)} samples")
    logger.info(f"Math: {len(math_data)} samples")
    logger.info(f"Basic: {len(basic_data)} samples")
    logger.info(f"Advanced: {len(advanced_data)} samples")
    logger.info(f"Total: {len(logic_data)+len(recognition_data)+len(math_data)+len(basic_data)+len(advanced_data)} samples")
    
    # Define tasks
    tasks = [
        ("logic_reasoning", logic_data, ["VALID", "INVALID"]),
        ("recognition", recognition_data, ["EXTRACT", "SKIP"]),
        ("mathematics", math_data, ["COMPUTE", "INVALID"]),
        ("basic_tasks", basic_data, ["CLASSIFY", "EXTRACT"]),
        ("advanced_tasks", advanced_data, ["DETECT", "EVALUATE", "VALID", "INVALID", "GENERATE"]),
    ]
    
    # Train each task
    results = {}
    for task_name, data, labels in tasks:
        checkpoint_dir = str(EXPERIMENT_DIR / f"checkpoints_{task_name}")
        result = train_one_task(
            task_name=task_name,
            data=data,
            label_names=labels,
            embedder=embedder,
            checkpoint_dir=checkpoint_dir,
            epochs=20,
            seed=42,
        )
        results[task_name] = result
    
    # Save combined results
    combined = {
        "experiment": "sweep-multi-task-training",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "tasks": results,
        "summary": {
            name: {"test_f1": r["test_f1"], "test_acc": r["test_acc"], "params": r["total_params"]}
            for name, r in results.items()
        },
    }
    
    results_path = str(EXPERIMENT_DIR / "multi_task_results.json")
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"{'Task':<20s} {'F1':>8s} {'Accuracy':>10s} {'Params':>10s}")
    logger.info("-" * 50)
    for name, r in results.items():
        logger.info(f"{name:<20s} {r['test_f1']:>8.4f} {r['test_acc']:>10.4f} {r['total_params']:>10,}")
    
    total_params = sum(r["total_params"] for r in results.values())
    avg_f1 = sum(r["test_f1"] for r in results.values()) / len(results)
    logger.info("-" * 50)
    logger.info(f"{'TOTAL':<20s} {avg_f1:>8.4f} {'':>10s} {total_params:>10,}")
    logger.info("=" * 70)
    
    return combined


if __name__ == "__main__":
    results = main()
