"""
Sweep Evidence Classifier — Real Training Experiment

Experiment: sweep-exp-20260830-1400
Task: Evidence classification (SUPPORTS / REFUTES / NEUTRAL)
Model: Frozen MiniLM embeddings + 2-layer MLP classifier
Hardware: CPU-only (Intel 12 cores, 15.7GB RAM)

This script performs ACTUAL TRAINING with real gradient updates.
"""
import sys
import os
import json
import time
import hashlib
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# Setup paths
实验_DIR = Path(__file__).parent
SWEEP_DIR =实验_DIR.parent.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(SWEEP_DIR.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_training")

# ════════════════════════════════════════════════════════════════════
# STEP 1: DETECT HARDWARE (honest report)
# ════════════════════════════════════════════════════════════════════

def detect_hardware() -> dict:
    import platform
    import psutil
    info = {
        "cpu": platform.processor() or "Unknown",
        "cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "ram_used_percent": psutil.virtual_memory().percent,
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    except ImportError:
        info["pytorch"] = "not installed"
        info["cuda"] = False
        info["gpu"] = "None"
    return info

# ════════════════════════════════════════════════════════════════════
# STEP 2: CREATE DATASET (Sweep-specific synthetic evidence pairs)
# ════════════════════════════════════════════════════════════════════

def create_dataset(seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Create evidence classification dataset.
    
    Labels:
        SUPPORTS  = evidence supports the claim
        REFUTES   = evidence contradicts the claim
        NEUTRAL   = evidence is irrelevant or insufficient
    """
    rng = random.Random(seed)
    
    # Sweep-specific evidence pairs
    raw_data = [
        # SUPPORTS examples
        {"claim": "Exercise improves cardiovascular health", "evidence": "Studies show regular exercise reduces heart disease risk by 30%", "label": "SUPPORTS"},
        {"claim": "Climate change is real", "evidence": "Global temperatures have risen 1.1 degrees Celsius since pre-industrial times", "label": "SUPPORTS"},
        {"claim": "Vaccines are effective", "evidence": "Clinical trials demonstrate 95% efficacy in preventing infection", "label": "SUPPORTS"},
        {"claim": "Sugar is harmful in excess", "evidence": "High sugar consumption is linked to obesity and diabetes", "label": "SUPPORTS"},
        {"claim": "Reading improves vocabulary", "evidence": "Research shows avid readers have 20% larger vocabulary", "label": "SUPPORTS"},
        {"claim": "Sleep is important for health", "evidence": "Adults who sleep 7-9 hours have better immune function", "label": "SUPPORTS"},
        {"claim": "Music enhances learning", "evidence": "Studies show background music improves retention by 15%", "label": "SUPPORTS"},
        {"claim": "Stress affects performance", "evidence": "Chronic stress impairs memory and decision-making", "label": "SUPPORTS"},
        {"claim": "Meditation helps focus", "evidence": "Regular meditation practitioners show improved attention spans", "label": "SUPPORTS"},
        {"claim": "Walking reduces disease risk", "evidence": "10000 steps daily reduces cardiovascular disease by 25%", "label": "SUPPORTS"},
        {"claim": "Water is essential for life", "evidence": "Human body is 60% water and requires daily hydration", "label": "SUPPORTS"},
        {"claim": "Sunlight provides vitamin D", "evidence": "UV exposure triggers vitamin D synthesis in skin", "label": "SUPPORTS"},
        {"claim": "Education increases opportunity", "evidence": "College graduates earn 67% more over their lifetime", "label": "SUPPORTS"},
        {"claim": "Poverty affects health outcomes", "evidence": "Low-income populations have higher rates of chronic disease", "label": "SUPPORTS"},
        {"claim": "Innovation drives economic growth", "evidence": "Technology sectors account for 35% of GDP growth", "label": "SUPPORTS"},
        {"claim": "Exercise improves mental health", "evidence": "Physical activity reduces symptoms of depression by 30%", "label": "SUPPORTS"},
        {"claim": "Renewable energy is growing", "evidence": "Solar capacity increased 25% in the last year", "label": "SUPPORTS"},
        {"claim": "Air pollution harms health", "evidence": "PM2.5 exposure increases respiratory disease risk", "label": "SUPPORTS"},
        {"claim": "Regular meals help metabolism", "evidence": "Consistent eating patterns support stable blood sugar", "label": "SUPPORTS"},
        {"claim": "Social connection improves wellbeing", "evidence": "Strong social ties correlate with longer lifespan", "label": "SUPPORTS"},
        {"claim": "Forests absorb carbon dioxide", "evidence": "Trees sequester approximately 2.6 billion tons of CO2 annually", "label": "SUPPORTS"},
        {"claim": "Hand washing prevents disease", "evidence": "Proper hand hygiene reduces GI illness by 31%", "label": "SUPPORTS"},
        {"claim": "Crowding increases disease transmission", "evidence": "Densely populated areas show faster spread of airborne diseases", "label": "SUPPORTS"},
        {"claim": "Antibiotics treat bacterial infections", "evidence": "Antibiotics kill or inhibit bacterial growth effectively", "label": "SUPPORTS"},
        {"claim": "Deforestation reduces biodiversity", "evidence": "Habitat loss from logging threatens thousands of species", "label": "SUPPORTS"},
        {"claim": "Exercise strengthens bones", "evidence": "Weight-bearing exercise increases bone density by 1-3%", "label": "SUPPORTS"},
        {"claim": "Omega-3 fatty acids benefit brain health", "evidence": "DHA supports neural membrane structure and function", "label": "SUPPORTS"},
        {"claim": "Regular backups prevent data loss", "evidence": "Automated backups recover 99.9% of deleted files", "label": "SUPPORTS"},
        {"claim": "Version control helps collaboration", "evidence": "Git reduces merge conflicts and tracks code changes", "label": "SUPPORTS"},
        {"claim": "Unit tests catch bugs early", "evidence": "Test-driven development reduces defect rates by 40%", "label": "SUPPORTS"},
        
        # REFUTES examples
        {"claim": "Smoking is harmless", "evidence": "Smoking causes lung cancer and heart disease", "label": "REFUTES"},
        {"claim": "The Earth is flat", "evidence": "Satellite imagery shows Earth is an oblate spheroid", "label": "REFUTES"},
        {"claim": "Vaccines cause autism", "evidence": "Multiple studies find no link between vaccines and autism", "label": "REFUTES"},
        {"claim": "Climate change is a hoax", "evidence": "97% of climate scientists agree human-caused warming is real", "label": "REFUTES"},
        {"claim": "Sugar is healthy", "evidence": "Excess sugar consumption increases obesity and diabetes risk", "label": "REFUTES"},
        {"claim": "Exercise is bad for you", "evidence": "Regular exercise reduces all-cause mortality by 30%", "label": "REFUTES"},
        {"claim": "The Earth is the center of the universe", "evidence": "Heliocentric model is supported by astronomical observations", "label": "REFUTES"},
        {"claim": "Humans do not need sleep", "evidence": "Sleep deprivation impairs cognitive function and immune response", "label": "REFUTES"},
        {"claim": "Germ theory is wrong", "evidence": "Microorganisms cause infectious diseases as proven by Koch", "label": "REFUTES"},
        {"claim": "Evolution is not real", "evidence": "Fossil record and DNA evidence support evolutionary theory", "label": "REFUTES"},
        {"claim": "Water does not boil at 100C", "evidence": "At standard pressure water boils at exactly 100 degrees Celsius", "label": "REFUTES"},
        {"claim": "Gravity does not exist", "evidence": "Objects fall at 9.8 m/s2 confirming gravitational attraction", "label": "REFUTES"},
        {"claim": "Antibiotics treat viruses", "evidence": "Antibiotics only work against bacteria, not viral infections", "label": "REFUTES"},
        {"claim": "The Sun orbits the Earth", "evidence": "Earth orbits the Sun as demonstrated by planetary motion", "label": "REFUTES"},
        {"claim": "Nuclear energy is always dangerous", "evidence": "Modern reactors have safety records better than fossil fuels", "label": "REFUTES"},
        {"claim": "Organic food is always healthier", "evidence": "Studies show minimal nutritional difference from conventional food", "label": "REFUTES"},
        {"claim": "Running damages knees", "evidence": "Runners have lower rates of knee osteoarthritis than non-runners", "label": "REFUTES"},
        {"claim": "Breakfast is the most important meal", "evidence": "Research shows no significant health difference from skipping breakfast", "label": "REFUTES"},
        {"claim": "You need 8 glasses of water daily", "evidence": "Water needs vary by individual; food provides significant hydration", "label": "REFUTES"},
        {"claim": "Brain usage is only 10 percent", "evidence": "Brain imaging shows all areas are active and used regularly", "label": "REFUTES"},
        {"claim": "Lightning never strikes twice", "evidence": "The Empire State Building is struck approximately 25 times per year", "label": "REFUTES"},
        {"claim": "Humans have five senses", "evidence": "Humans have at least nine senses including balance and temperature", "label": "REFUTES"},
        {"claim": "Goldfish have 3-second memory", "evidence": "Goldfish can be trained and remember tasks for months", "label": "REFUTES"},
        {"claim": "Shaving makes hair grow thicker", "evidence": "Shaving does not change hair thickness or growth rate", "label": "REFUTES"},
        {"claim": "Napoleon was unusually short", "evidence": "Napoleon was 5 feet 7 inches, average height for his era", "label": "REFUTES"},
        {"claim": "Bats are blind", "evidence": "All bat species can see; some use echolocation in addition", "label": "REFUTES"},
        {"claim": "Oxygen is the most abundant gas in atmosphere", "evidence": "Nitrogen makes up 78 percent of atmosphere, oxygen only 21 percent", "label": "REFUTES"},
        {"claim": "Pain means tissue damage", "evidence": "Phantom limb pain occurs without any tissue damage present", "label": "REFUTES"},
        {"claim": "We only use 10 percent of our brain", "evidence": "Brain scans show we use virtually all of our brain regularly", "label": "REFUTES"},
        {"claim": "Light travels slower than sound", "evidence": "Light travels at 300000 km/s, sound at 343 m/s", "label": "REFUTES"},
        
        # NEUTRAL examples
        {"claim": "Exercise improves health", "evidence": "The weather forecast predicts rain tomorrow", "label": "NEUTRAL"},
        {"claim": "Climate change is real", "evidence": "The stock market rose 2 percent today", "label": "NEUTRAL"},
        {"claim": "Vaccines are safe", "evidence": "A new restaurant opened downtown", "label": "NEUTRAL"},
        {"claim": "Sugar is harmful", "evidence": "The team won their football game yesterday", "label": "NEUTRAL"},
        {"claim": "Reading improves vocabulary", "evidence": "The museum exhibit features ancient pottery", "label": "NEUTRAL"},
        {"claim": "Sleep is important", "evidence": "Gas prices increased by 15 cents per gallon", "label": "NEUTRAL"},
        {"claim": "Music helps learning", "evidence": "The city council approved a new park budget", "label": "NEUTRAL"},
        {"claim": "Stress affects health", "evidence": "The airline industry reported record profits", "label": "NEUTRAL"},
        {"claim": "Meditation helps focus", "evidence": "A new species of butterfly was discovered in Brazil", "label": "NEUTRAL"},
        {"claim": "Walking is healthy", "evidence": "The movie won three Academy Awards", "label": "NEUTRAL"},
        {"claim": "Water is essential", "evidence": "The population of Tokyo exceeds 13 million", "label": "NEUTRAL"},
        {"claim": "Sunlight provides vitamin D", "evidence": "The smartphone market grew 8 percent last quarter", "label": "NEUTRAL"},
        {"claim": "Education helps careers", "evidence": "The recipe calls for two cups of flour", "label": "NEUTRAL"},
        {"claim": "Poverty is a problem", "evidence": "The bridge construction will take three years", "label": "NEUTRAL"},
        {"claim": "Innovation drives growth", "evidence": "The concert sold out in 10 minutes", "label": "NEUTRAL"},
        {"claim": "Exercise improves mental health", "evidence": "The library has over 500000 books", "label": "NEUTRAL"},
        {"claim": "Renewable energy is growing", "evidence": "The train schedule was updated for holidays", "label": "NEUTRAL"},
        {"claim": "Air pollution is harmful", "evidence": "The recipe book has 200 dessert recipes", "label": "NEUTRAL"},
        {"claim": "Regular meals help metabolism", "evidence": "The opera house was built in 1973", "label": "NEUTRAL"},
        {"claim": "Social connection helps", "evidence": "The volcano last erupted in 2019", "label": "NEUTRAL"},
        {"claim": "Forests absorb CO2", "evidence": "The chess championship is next month", "label": "NEUTRAL"},
        {"claim": "Hand washing prevents disease", "evidence": "The painting sold for 4.5 million dollars", "label": "NEUTRAL"},
        {"claim": "Crowding spreads disease", "evidence": "The submarine reached a depth of 3000 meters", "label": "NEUTRAL"},
        {"claim": "Antibiotics treat infections", "evidence": "The bakery opens at 6 AM every morning", "label": "NEUTRAL"},
        {"claim": "Deforestation reduces biodiversity", "evidence": "The keyboard has 104 keys", "label": "NEUTRAL"},
        {"claim": "Exercise strengthens bones", "evidence": "The river is 1200 kilometers long", "label": "NEUTRAL"},
        {"claim": "Omega-3 benefits brain health", "evidence": "The satellite orbits at 400 kilometers altitude", "label": "NEUTRAL"},
        {"claim": "Backups prevent data loss", "evidence": "The piano has 88 keys", "label": "NEUTRAL"},
        {"claim": "Version control helps", "evidence": "The garden has 15 rose bushes", "label": "NEUTRAL"},
        {"claim": "Unit tests catch bugs", "evidence": "The train travels at 300 km per hour", "label": "NEUTRAL"},
    ]
    
    # Add paraphrased variants
    paraphrased = []
    for item in raw_data:
        # Create 2 variations of each
        words = item["evidence"].split()
        if len(words) > 5:
            # Shuffle middle words slightly
            mid = words[2:-2]
            rng.shuffle(mid)
            new_evidence = " ".join(words[:2] + mid + words[-2:])
            paraphrased.append({"claim": item["claim"], "evidence": new_evidence, "label": item["label"]})
    
    all_data = raw_data + paraphrased
    rng.shuffle(all_data)
    
    # Split: 70% train, 15% val, 15% test
    n = len(all_data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train = all_data[:train_end]
    val = all_data[train_end:val_end]
    test = all_data[val_end:]
    
    return train, val, test

# ════════════════════════════════════════════════════════════════════
# STEP 3: MODEL (frozen MiniLM + trainable classifier)
# ════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class EvidenceDataset(Dataset):
    """Dataset for evidence classification."""
    
    def __init__(self, data: list[dict], embedder, label_map: dict):
        self.data = data
        self.embedder = embedder
        self.label_map = label_map
        self.cache = {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        key = f"{item['claim']}|||{item['evidence']}"
        
        if key not in self.cache:
            # Embed claim and evidence separately, then element-wise add
            import numpy as np
            claim_emb = self.embedder.embed(item['claim'])
            evid_emb = self.embedder.embed(item['evidence'])
            combined = np.array(claim_emb.vector) + np.array(evid_emb.vector)
            self.cache[key] = torch.tensor(combined.tolist(), dtype=torch.float32)
        
        label = self.label_map[item["label"]]
        return self.cache[key], label


class EvidenceClassifier(nn.Module):
    """Small MLP classifier on top of frozen embeddings.
    
    Sweep-original architecture.
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, num_classes: int = 3):
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

# ════════════════════════════════════════════════════════════════════
# STEP 4: TRAINING LOOP (real gradients, real updates)
# ════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    checkpoint_dir: str,
) -> dict:
    """Actual training loop with real gradient updates."""
    
    device = torch.device("cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")
    logger.info(f"Starting training: {config['epochs']} epochs, batch_size={config['batch_size']}")
    
    t_start = time.time()
    
    for epoch in range(1, config["epochs"] + 1):
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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
        
        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Calculate F1
        from sklearn.metrics import f1_score, precision_score, recall_score
        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        elapsed = time.time() - t_start
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_precision": round(val_precision, 4),
            "val_recall": round(val_recall, 4),
            "lr": current_lr,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(epoch_info)
        
        logger.info(
            f"Epoch {epoch:2d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "config": config,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            
            logger.info(f"  -> Saved checkpoint (val_loss={val_loss:.4f}, val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info(f"  -> Early stopping at epoch {epoch} (no improvement for {config['patience']} epochs)")
                break
    
    total_time = time.time() - t_start
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "config": config,
    }, os.path.join(checkpoint_dir, "final_model.pt"))
    
    return {
        "total_epochs": epoch,
        "total_time_s": round(total_time, 1),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_f1": round(best_val_f1, 4),
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "history": history,
        "total_params": total_params,
        "checkpoint": os.path.join(checkpoint_dir, "best_model.pt"),
    }

# ════════════════════════════════════════════════════════════════════
# STEP 5: EVALUATION
# ════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module, data_loader: DataLoader, label_names: list[str]) -> dict:
    """Evaluate model on a dataset."""
    import numpy as np
    from sklearn.metrics import (
        classification_report, f1_score, precision_score, recall_score,
        confusion_matrix,
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.tolist())
            all_probs.extend(probs.tolist())
    
    # Metrics
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    # Per-class metrics
    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Confidence calibration
    probs_arr = np.array(all_probs)
    max_probs = probs_arr.max(axis=1)
    correct = np.array(all_preds) == np.array(all_labels)
    
    # Calibration bins
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
    calibration = []
    for i in range(len(bins) - 1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            calibration.append({
                "range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "count": int(mask.sum()),
                "accuracy": round(float(bin_acc), 4),
                "avg_confidence": round(float(max_probs[mask].mean()), 4),
            })
    
    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "per_class": {label_names[i]: {
            "f1": round(report[str(i)]["f1-score"], 4),
            "precision": round(report[str(i)]["precision"], 4),
            "recall": round(report[str(i)]["recall"], 4),
            "support": int(report[str(i)]["support"]),
        } for i in range(len(label_names)) if str(i) in report},
        "confusion_matrix": cm.tolist(),
        "calibration": calibration,
        "total_samples": len(all_labels),
        "correct": int(correct.sum()),
        "accuracy": round(float(correct.mean()), 4),
    }

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("SWEEP NEURAL ENGINE — REAL TRAINING SESSION")
    logger.info("Experiment: sweep-exp-20260830-1400")
    logger.info("=" * 70)
    
    # 1. Hardware
    hw = detect_hardware()
    logger.info(f"Hardware: {json.dumps(hw, indent=2)}")
    
    # 2. Create dataset
    logger.info("\n--- Creating Dataset ---")
    train_data, val_data, test_data = create_dataset(seed=42)
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Dataset statistics
    train_labels = [d["label"] for d in train_data]
    from collections import Counter
    logger.info(f"Train class distribution: {dict(Counter(train_labels))}")
    logger.info(f"Average claim length: {sum(len(d['claim'].split()) for d in train_data) / len(train_data):.1f} words")
    logger.info(f"Average evidence length: {sum(len(d['evidence'].split()) for d in train_data) / len(train_data):.1f} words")
    
    # Save dataset
    dataset_info = {
        "experiment_id": "sweep-exp-20260830-1400",
        "task": "evidence_classification",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "classes": ["SUPPORTS", "REFUTES", "NEUTRAL"],
        "train_distribution": dict(Counter(train_labels)),
        "dataset_hash": hashlib.md5(json.dumps(train_data, sort_keys=True).encode()).hexdigest()[:12],
        "seed": 42,
    }
    
    # 3. Load embedder
    logger.info("\n--- Loading Embedder ---")
    sys.path.insert(0, str(SWEEP_DIR))
    from neurons.semantic_embeddings import SemanticEmbedder
    embedder = SemanticEmbedder()
    logger.info(f"Embedder loaded: {embedder.backend}, dim=384")
    
    # 4. Create data loaders
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NEUTRAL": 2}
    label_names = ["SUPPORTS", "REFUTES", "NEUTRAL"]
    
    logger.info("\n--- Computing Embeddings (frozen) ---")
    train_ds = EvidenceDataset(train_data, embedder, label_map)
    val_ds = EvidenceDataset(val_data, embedder, label_map)
    test_ds = EvidenceDataset(test_data, embedder, label_map)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Pre-compute embeddings for train set
    logger.info("Pre-computing train embeddings...")
    t0 = time.time()
    for i in range(len(train_ds)):
        _ = train_ds[i]
    logger.info(f"Train embeddings computed in {time.time() - t0:.1f}s")
    
    # 5. Baseline (untrained model)
    logger.info("\n--- Baseline Evaluation (untrained) ---")
    # Detect actual embedding dimension from first sample
    sample_x, _ = next(iter(train_loader))
    actual_dim = sample_x.shape[1]
    logger.info(f"Detected input dimension: {actual_dim}")
    
    model = EvidenceClassifier(input_dim=actual_dim, hidden_dim=128, num_classes=3)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    baseline_metrics = evaluate(model, test_loader, label_names)
    logger.info(f"Baseline F1: {baseline_metrics['f1']:.4f}")
    logger.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Save baseline
    with open(os.path.join(实验_DIR, "baseline.json"), "w") as f:
        json.dump({"baseline": baseline_metrics, "hardware": hw, "dataset": dataset_info}, f, indent=2)
    
    # 6. TRAIN
    logger.info("\n--- Training ---")
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "patience": 5,
        "weight_decay": 0.0001,
        "seed": 42,
    }
    
    checkpoint_dir = str(实验_DIR / "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    random.seed(42)
    torch.manual_seed(42)
    
    training_result = train_model(model, train_loader, val_loader, config, checkpoint_dir)
    
    # 7. Load best checkpoint and evaluate
    logger.info("\n--- Final Evaluation ---")
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    final_metrics = evaluate(model, test_loader, label_names)
    logger.info(f"Final F1: {final_metrics['f1']:.4f}")
    logger.info(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
    
    # 8. Generalization test (paraphrased variants)
    logger.info("\n--- Generalization Test ---")
    gen_data = []
    for item in test_data:
        # Create paraphrased version
        words = item["evidence"].split()
        if len(words) > 4:
            # Reverse word order (extreme paraphrase)
            gen_data.append({"claim": item["claim"], "evidence": " ".join(reversed(words)), "label": item["label"]})
        else:
            gen_data.append(item)
    
    gen_ds = EvidenceDataset(gen_data, embedder, label_map)
    gen_loader = DataLoader(gen_ds, batch_size=32, shuffle=False)
    gen_metrics = evaluate(model, gen_loader, label_names)
    logger.info(f"Generalization F1: {gen_metrics['f1']:.4f}")
    
    # 9. Adversarial test
    logger.info("\n--- Adversarial Test ---")
    adv_data = [
        {"claim": "Exercise is good", "evidence": "", "label": "NEUTRAL"},  # Empty evidence
        {"claim": "", "evidence": "Some evidence here", "label": "NEUTRAL"},  # Empty claim
        {"claim": "X", "evidence": "Y", "label": "NEUTRAL"},  # Minimal
        {"claim": "Exercise improves health", "evidence": "Exercise improves health", "label": "SUPPORTS"},  # Claim = evidence
        {"claim": "Exercise improves health", "evidence": "EXERCISE IMPROVES HEALTH", "label": "SUPPORTS"},  # Uppercase
    ]
    adv_ds = EvidenceDataset(adv_data, embedder, label_map)
    adv_loader = DataLoader(adv_ds, batch_size=32, shuffle=False)
    adv_metrics = evaluate(model, adv_loader, label_names)
    logger.info(f"Adversarial F1: {adv_metrics['f1']:.4f}")
    
    # 10. Save results
    results = {
        "experiment_id": "sweep-exp-20260830-1400",
        "hardware": hw,
        "dataset": dataset_info,
        "model": {
            "architecture": "EvidenceClassifier (MLP)",
            "total_params": total_params,
            "trainable_params": total_params,
            "input_dim": 384,
            "hidden_dim": 128,
            "num_classes": 3,
            "embedding_model": "all-MiniLM-L6-v2 (frozen)",
        },
        "training": training_result,
        "baseline": baseline_metrics,
        "final": final_metrics,
        "generalization": gen_metrics,
        "adversarial": adv_metrics,
        "improvement": {
            "f1_change": round(final_metrics["f1"] - baseline_metrics["f1"], 4),
            "accuracy_change": round(final_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
        },
    }
    
    with open(os.path.join(实验_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save training log
    with open(os.path.join(实验_DIR, "training.log"), "w") as f:
        for entry in training_result["history"]:
            f.write(json.dumps(entry) + "\n")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Baseline F1:    {baseline_metrics['f1']:.4f}")
    logger.info(f"Final F1:       {final_metrics['f1']:.4f}")
    logger.info(f"Improvement:    {results['improvement']['f1_change']:+.4f}")
    logger.info(f"Generalization: {gen_metrics['f1']:.4f}")
    logger.info(f"Adversarial:    {adv_metrics['f1']:.4f}")
    logger.info(f"Training time:  {training_result['total_time_s']:.1f}s")
    logger.info(f"Epochs:         {training_result['total_epochs']}")
    logger.info(f"Best val loss:  {training_result['best_val_loss']:.4f}")
    logger.info(f"Parameters:     {total_params:,}")
    logger.info(f"Checkpoint:     {checkpoint_dir}/best_model.pt")
    logger.info("=" * 70)
    
    return results

if __name__ == "__main__":
    results = main()
