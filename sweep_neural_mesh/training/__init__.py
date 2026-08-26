"""
Sweep Neural Mesh — Training and Improvement System.

§1: No general learning rule. Training is explicit.
§2: Verified examples enter primary dataset.
§3: Mechanics cannot be degraded.
§34: Autonomous training mode.
"""

from sweep_neural_mesh.training.domains import DomainScore, ExpertiseTracker
from sweep_neural_mesh.training.task_generator import TaskGenerator, Task
from sweep_neural_mesh.training.solver import Solver, Candidate, SolveResult
from sweep_neural_mesh.training.critique import Critique, CritiqueResult
from sweep_neural_mesh.training.verifier import Verifier
from sweep_neural_mesh.training.experience import Experience, ExperienceMemory
from sweep_neural_mesh.training.curriculum import CurriculumManager, CurriculumState
from sweep_neural_mesh.training.calibration import ConfidenceCalibrator
from sweep_neural_mesh.training.versioning import VersionManager, ModelVersion
from sweep_neural_mesh.training.dashboard import Dashboard
from sweep_neural_mesh.training.trainer import Trainer, TrainingConfig, TrainingResult

__all__ = [
    "DomainScore", "ExpertiseTracker",
    "TaskGenerator", "Task",
    "Solver", "Candidate", "SolveResult",
    "Critique", "CritiqueResult",
    "Verifier",
    "Experience", "ExperienceMemory",
    "CurriculumManager", "CurriculumState",
    "ConfidenceCalibrator",
    "VersionManager", "ModelVersion",
    "Dashboard",
    "Trainer", "TrainingConfig", "TrainingResult",
]
