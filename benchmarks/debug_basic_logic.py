"""
Deep analysis of basic_logic failures.
Categorize each failure by sub-type to understand what reasoning is missing.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)
d = json.load(open(r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII\benchmarks\results\dataset.json", encoding="utf-8"))["cases"]

cases = [c for c in d if c["category"] == "basic_logic"]

# Categorize failures
categories = {
    "grammar_wrong": [],      # "Do birds can talk?" - grammar errors expected as refuted
    "syllogism": [],           # "No X are Y. Z is X. Z is not Y?"
    "quantifier": [],          # "All X are Y" / "Some X are Y"
    "transitivity": [],        # "A>B, B>C, A>C?"
    "modus_ponens": [],        # "If X then Y. X is true. Y?"
    "modus_tollens": [],       # "If X then Y. Not Y. Not X?"
    "numerical": [],           # Comparisons with numbers
    "other": [],
}

wrong_total = 0
for c in cases:
    r = cortex.reason(c["query"], c["evidence"])
    if r.decision != c["expected_decision"]:
        wrong_total += 1
        q = c["query"].lower()
        ev = c["evidence"][0].lower() if c["evidence"] else ""

        if "do " in q and (" can " in q or " are " in q):
            categories["grammar_wrong"].append(c)
        elif "no " in ev and ("are " in q or "is " in q):
            categories["syllogism"].append(c)
        elif "all " in q or "some " in q:
            categories["quantifier"].append(c)
        elif ">" in ev or "faster" in ev or "slower" in ev or "bigger" in ev:
            categories["numerical"].append(c)
        elif "if " in q and "then " in q:
            categories["modus_tollens"].append(c)
        else:
            categories["other"].append(c)

print(f"basic_logic: {len(cases)} cases, {wrong_total} wrong, {100-wrong_total/len(cases)*100:.0f}% acc")
print(f"\nFailure breakdown by sub-type:")
for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
    if items:
        print(f"\n  {cat}: {len(items)} failures")
        for c in items[:5]:
            print(f"    Q: {c['query'][:70]}")
            print(f"    Ev: {c['evidence'][0][:80]}")
            print(f"    Expected: {c['expected_decision']}")
