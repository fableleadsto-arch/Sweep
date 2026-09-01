"""
Integration test for all new training modules (Sections 14-29).
"""
import sys, io, json, re, os

# sweep_neural_mesh/ dir (for 'from training import ...' and 'from benchmarks import ...')
_this_dir = os.path.dirname(os.path.abspath(__file__))
_smn_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _smn_dir)         # sweep_neural_mesh/
# parent dir (for 'from sweep_neural_mesh... imports')
sys.path.append(os.path.dirname(_smn_dir))  # parent of sweep_neural_mesh/
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("SWEEP NEURAL MESH — TRAINING SYSTEMS INTEGRATION TEST")
print("=" * 70)
passed = 0
failed = 0

# Test 1: Hard Negatives
print("\n[1/10] Hard Negative Generator...")
try:
    from training.hard_negatives import HardNegativeGenerator
    gen = HardNegativeGenerator(seed=42)
    negatives = gen.generate_all(count=40)
    assert len(negatives) > 0
    summary = gen.summary()
    assert summary["total_generated"] > 0
    print(f"   PASS: {summary['total_generated']} negatives, {len(summary['categories'])} categories")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 2: Adversarial Testing
print("\n[2/10] Adversarial Test Suite...")
try:
    from training.adversarial import AdversarialTestSuite
    suite = AdversarialTestSuite(seed=42)
    tasks = suite.generate_all(count_per_type=3)
    assert len(tasks) > 0
    def dummy_model(text):
        if "contradict" in text.lower() or "conflict" in text.lower():
            return ("There is a contradiction in the sources.", 0.7)
        return ("I can answer this.", 0.8)
    results = suite.evaluate(tasks[:10], dummy_model)
    summary = suite.summary()
    assert summary["total_tasks"] > 0
    print(f"   PASS: {summary['total_tasks']} tasks, {summary['detection_rate']:.1%} detection rate")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 3: Ablation Study
print("\n[3/10] Ablation Study...")
try:
    from training.ablation import AblationStudy
    study = AblationStudy()
    study.set_components([
        ("reasoning", True, "Core reasoning"),
        ("multi_core", True, "Multi-core processing"),
        ("fast_path", True, "Fast path"),
    ])
    def mock_test(config):
        base_acc = 0.85
        if not config.get("reasoning", True):
            base_acc -= 0.3
        if not config.get("multi_core", True):
            base_acc -= 0.1
        return (base_acc, 100.0, 20)
    baseline = study.run_full_system(mock_test)
    assert baseline.accuracy == 0.85
    results = study.run_all_ablations(mock_test)
    impact = study.compute_impact()
    summary = study.summary()
    assert len(impact) > 0
    print(f"   PASS: {len(results)} ablations, baseline={baseline.accuracy:.2f}")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 4: Hardware Detection
print("\n[4/10] Hardware Detection...")
try:
    from training.hardware import HardwareDetector
    detector = HardwareDetector()
    profile = detector.detect()
    config = detector.select_config(profile)
    report = detector.get_system_report()
    assert profile.os_name != ""
    assert config.mode in ("LOW_RESOURCE", "BALANCED", "HIGH_PERFORMANCE")
    print(f"   PASS: {profile.os_name}, {profile.cpu_count} cores, {profile.ram_total_gb:.1f}GB RAM, mode={config.mode}")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 5: Failure Analysis
print("\n[5/10] Failure Analyzer...")
try:
    from training.failure_analysis import FailureAnalyzer
    analyzer = FailureAnalyzer()
    record = analyzer.record_failure(
        input_text="What is 2 + 2?",
        expected_output="4",
        actual_output="5",
        task_domain="mathematics",
        confidence=0.9,
    )
    assert record.failure_id.startswith("FAIL-")
    assert record.category == "ARITHMETIC_ERROR"
    patterns = analyzer.get_patterns()
    summary = analyzer.summary()
    assert summary["total_failures"] == 1
    assert len(analyzer.regression_tests) > 0
    print(f"   PASS: {record.failure_id}, category={record.category}, severity={record.severity}")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 6: Generalization Testing
print("\n[6/10] Generalization Tester...")
try:
    from training.generalization import GeneralizationTester
    tester = GeneralizationTester(seed=42)
    tasks = tester.generate_all(count_per_type=5)
    assert len(tasks) > 0
    def mock_model(text):
        if "paris" in text.lower():
            return ("Paris", 0.9)
        if "+" in text:
            nums = re.findall(r"\d+", text)
            if len(nums) >= 2:
                return (str(int(nums[0]) + int(nums[1])), 0.95)
        return ("I need more information.", 0.5)
    results = tester.evaluate(tasks[:15], mock_model)
    summary = tester.summary()
    assert summary["total_tasks"] > 0
    print(f"   PASS: {summary['total_tasks']} tasks, {summary['overall_accuracy']:.1%} accuracy")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 7: Dataset Pipeline
print("\n[7/10] Dataset Pipeline...")
try:
    from training.dataset_pipeline import DatasetPipeline, DatasetEntry
    pipeline = DatasetPipeline()
    entries = [
        DatasetEntry(
            entry_id=f"TEST-{i}", task_type="logic", difficulty=1,
            modality="text", source="test", license="open", quality=0.8,
            input_text=f"Is {i} > {i-1}?", expected_output="YES",
            evaluation_criteria="exact_match", split="unassigned",
        )
        for i in range(10)
    ]
    added = pipeline.add_entries(entries)
    assert added == 10
    splits = pipeline.split()
    assert "train" in splits
    stats = pipeline.stats()
    assert stats.total_entries == 10
    contamination = pipeline.check_contamination()
    assert contamination["contamination_free"]
    print(f"   PASS: {stats.total_entries} entries, splits: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 8: Safety Manager
print("\n[8/10] Safety Manager...")
try:
    from training.safety import SafetyManager, DataLicense
    safety = SafetyManager()
    assert safety.is_dataset_safe("open_facts", "training")
    safety.register_license(DataLicense(
        dataset_name="private_data", license_type="private",
        allows_training=False,
    ))
    assert not safety.is_dataset_safe("private_data", "training")
    pii_result = safety.check_privacy("My email is test@example.com")
    assert pii_result["contains_pii"]
    safety.log_data_access("open_facts", "trainer", "model training")
    safety.log_retrieval("wikipedia.org", "quantum computing")
    summary = safety.summary()
    assert summary["total_audit_entries"] > 0
    print(f"   PASS: {summary['total_audit_entries']} audit entries, {summary['compliance_rate']:.1%} compliance")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 9: Comprehensive Benchmark
print("\n[9/10] Comprehensive Benchmark...")
try:
    from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
    bench = ComprehensiveBenchmark()
    tasks = bench.generate_benchmark_tasks(per_category=10)
    def simple_model(text):
        if "paris" in text.lower():
            return ("Paris", 0.9)
        if "mercury" in text.lower():
            return ("Mercury", 0.85)
        if "h2o" in text.lower():
            return ("H2O", 0.95)
        if "unknown" in text.lower() or "mars" in text.lower() or "presidential" in text.lower():
            return ("UNKNOWN", 0.3)
        if "+" in text:
            nums = re.findall(r"\d+", text)
            if len(nums) >= 2:
                return (str(int(nums[0]) + int(nums[1])), 0.95)
        if "yes" in text.lower() or "living thing" in text.lower():
            return ("YES", 0.8)
        return ("I need more information.", 0.5)
    metrics = bench.evaluate(simple_model)
    report_path = bench.generate_report()
    with open(report_path) as f:
        summary = json.load(f)
    print(f"   PASS: {summary['summary']['total_tasks']} tasks, accuracy={summary['summary']['overall_accuracy']:.1%}")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

# Test 10: All imports work
print("\n[10/10] Import smoke test...")
try:
    from training import (
        HardNegativeGenerator, AdversarialTestSuite, AblationStudy,
        HardwareDetector, FailureAnalyzer, GeneralizationTester,
        DatasetPipeline, SafetyManager,
    )
    from training import (
        Trainer, TrainingConfig, CurriculumManager,
        ConfidenceCalibrator, VersionManager,
    )
    print("   PASS: All training modules import successfully")
    passed += 1
except Exception as e:
    print(f"   FAIL: {e}")
    failed += 1

print()
print("=" * 70)
print(f"RESULTS: {passed}/{passed + failed} passed")
print("=" * 70)
