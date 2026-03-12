"""
Exhaustive Tests for Ensemble Patterns Module

Tests all ensemble types, configurations, execution,
registry, suggestion logic, and helper functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.core.ensemble import (
    EnsembleType,
    AgentRole,
    AgentAssignment,
    EnsembleConfig,
    EnsembleResult,
    EnsemblePattern,
    ArchitectureReviewBoard,
    CodeSprint,
    ResearchCouncil,
    DocumentAssembly,
    RequirementsWorkshop,
    ENSEMBLE_REGISTRY,
    get_ensemble,
    get_all_ensembles,
    suggest_ensemble,
    execute_ensemble,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnsembleType:
    def test_all_types(self):
        assert EnsembleType.ARCHITECTURE_REVIEW_BOARD == "architecture_review_board"
        assert EnsembleType.CODE_SPRINT == "code_sprint"
        assert EnsembleType.RESEARCH_COUNCIL == "research_council"
        assert EnsembleType.DOCUMENT_ASSEMBLY == "document_assembly"
        assert EnsembleType.REQUIREMENTS_WORKSHOP == "requirements_workshop"

    def test_count(self):
        assert len(EnsembleType) == 5


class TestAgentRole:
    def test_all_roles(self):
        assert AgentRole.LEAD == "lead"
        assert AgentRole.REVIEWER == "reviewer"
        assert AgentRole.CONTRIBUTOR == "contributor"
        assert AgentRole.ADVISOR == "advisor"
        assert AgentRole.QUALITY_GATE == "quality_gate"
        assert AgentRole.OBSERVER == "observer"

    def test_count(self):
        assert len(AgentRole) == 6


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestAgentAssignment:
    def test_create(self):
        aa = AgentAssignment(
            agent_name="Analyst", role=AgentRole.LEAD,
            phase="analysis", dependencies=[], parallel_with=[],
        )
        assert aa.max_turns == 30
        assert aa.model == "claude-3-5-sonnet-20241022"

    def test_custom_values(self):
        aa = AgentAssignment(
            agent_name="Executor", role=AgentRole.CONTRIBUTOR,
            phase="execution", dependencies=["Analyst"],
            parallel_with=["Reviewer"], max_turns=50, model="opus",
        )
        assert aa.max_turns == 50
        assert aa.model == "opus"


class TestEnsembleConfig:
    def test_create(self):
        config = EnsembleConfig(
            name="Test", description="Test ensemble",
            ensemble_type=EnsembleType.CODE_SPRINT, tier_level=2,
            agent_assignments=[], required_smes=[], quality_gates=[],
            expected_output="output", success_criteria=["criteria"],
        )
        assert config.name == "Test"
        assert config.tier_level == 2


class TestEnsembleResult:
    def test_create(self):
        result = EnsembleResult(
            ensemble_type=EnsembleType.CODE_SPRINT, success=True,
            outputs={"agent": "output"}, quality_gate_results={"gate": True},
            total_turns=100, execution_time_seconds=60.0,
            recommendations=["Rec 1"],
        )
        assert result.success is True
        assert result.total_turns == 100


# =============================================================================
# Architecture Review Board Tests
# =============================================================================

class TestArchitectureReviewBoard:
    def test_instantiation(self):
        arb = ArchitectureReviewBoard()
        assert arb.name == "Architecture Review Board"

    def test_get_config(self):
        arb = ArchitectureReviewBoard()
        config = arb.get_config()
        assert config.ensemble_type == EnsembleType.ARCHITECTURE_REVIEW_BOARD
        assert config.tier_level == 3
        assert len(config.agent_assignments) > 0
        assert "cloud_architect" in config.required_smes
        assert "security_analyst" in config.required_smes

    def test_execute(self):
        arb = ArchitectureReviewBoard()
        result = arb.execute({"task": "review architecture"})
        assert result.success is True
        assert result.ensemble_type == EnsembleType.ARCHITECTURE_REVIEW_BOARD
        assert len(result.outputs) > 0

    def test_quality_gates(self):
        arb = ArchitectureReviewBoard()
        config = arb.get_config()
        assert "Verifier" in config.quality_gates
        assert "Critic" in config.quality_gates
        assert "Reviewer" in config.quality_gates


# =============================================================================
# Code Sprint Tests
# =============================================================================

class TestCodeSprint:
    def test_instantiation(self):
        cs = CodeSprint()
        assert cs.name == "Code Sprint"

    def test_get_config(self):
        cs = CodeSprint()
        config = cs.get_config()
        assert config.ensemble_type == EnsembleType.CODE_SPRINT
        assert config.tier_level == 2
        assert "test_engineer" in config.required_smes

    def test_execute(self):
        cs = CodeSprint()
        result = cs.execute({"task": "implement feature"})
        assert result.success is True


# =============================================================================
# Research Council Tests
# =============================================================================

class TestResearchCouncil:
    def test_instantiation(self):
        rc = ResearchCouncil()
        assert rc.name == "Research Council"

    def test_get_config(self):
        rc = ResearchCouncil()
        config = rc.get_config()
        assert config.ensemble_type == EnsembleType.RESEARCH_COUNCIL
        assert config.tier_level == 4
        assert "ai_ml_engineer" in config.required_smes

    def test_execute(self):
        rc = ResearchCouncil()
        result = rc.execute({"task": "research topic"})
        assert result.success is True
        assert result.ensemble_type is not None


# =============================================================================
# Document Assembly Tests
# =============================================================================

class TestDocumentAssembly:
    def test_instantiation(self):
        da = DocumentAssembly()
        assert da.name == "Document Assembly"

    def test_get_config(self):
        da = DocumentAssembly()
        config = da.get_config()
        assert config.ensemble_type == EnsembleType.DOCUMENT_ASSEMBLY
        assert config.tier_level == 2
        assert "technical_writer" in config.required_smes

    def test_execute(self):
        da = DocumentAssembly()
        result = da.execute({"task": "create document"})
        assert result.success is True


# =============================================================================
# Requirements Workshop Tests
# =============================================================================

class TestRequirementsWorkshop:
    def test_instantiation(self):
        rw = RequirementsWorkshop()
        assert rw.name == "Requirements Workshop"

    def test_get_config(self):
        rw = RequirementsWorkshop()
        config = rw.get_config()
        assert config.ensemble_type == EnsembleType.REQUIREMENTS_WORKSHOP
        assert config.tier_level == 2
        assert "business_analyst" in config.required_smes

    def test_execute(self):
        rw = RequirementsWorkshop()
        result = rw.execute({"task": "gather requirements"})
        assert result.success is True


# =============================================================================
# Registry Tests
# =============================================================================

class TestEnsembleRegistry:
    def test_all_types_registered(self):
        for et in EnsembleType:
            assert et in ENSEMBLE_REGISTRY

    def test_registry_values_are_classes(self):
        for cls in ENSEMBLE_REGISTRY.values():
            assert issubclass(cls, EnsemblePattern)


# =============================================================================
# get_ensemble Tests
# =============================================================================

class TestGetEnsemble:
    def test_valid_type(self):
        ensemble = get_ensemble(EnsembleType.CODE_SPRINT)
        assert ensemble is not None
        assert isinstance(ensemble, CodeSprint)

    def test_all_types(self):
        for et in EnsembleType:
            ensemble = get_ensemble(et)
            assert ensemble is not None


# =============================================================================
# get_all_ensembles Tests
# =============================================================================

class TestGetAllEnsembles:
    def test_returns_all(self):
        all_ensembles = get_all_ensembles()
        assert len(all_ensembles) == 5

    def test_all_are_patterns(self):
        for et, ensemble in get_all_ensembles().items():
            assert isinstance(ensemble, EnsemblePattern)


# =============================================================================
# suggest_ensemble Tests
# =============================================================================

class TestSuggestEnsemble:
    def test_architecture_review(self):
        result = suggest_ensemble("Review the system architecture design")
        assert isinstance(result, ArchitectureReviewBoard)

    def test_code_sprint(self):
        result = suggest_ensemble("Quick code implementation sprint")
        assert isinstance(result, CodeSprint)

    def test_research(self):
        result = suggest_ensemble("Research the best approach for this problem")
        assert isinstance(result, ResearchCouncil)

    def test_document(self):
        result = suggest_ensemble("Create documentation for the API")
        assert isinstance(result, DocumentAssembly)

    def test_requirements(self):
        result = suggest_ensemble("Gather requirements for the new feature")
        assert isinstance(result, RequirementsWorkshop)

    def test_default_is_code_sprint(self):
        result = suggest_ensemble("Do something random")
        assert isinstance(result, CodeSprint)

    def test_case_insensitive(self):
        result = suggest_ensemble("RESEARCH the ARCHITECTURE")
        assert result is not None


# =============================================================================
# execute_ensemble Tests
# =============================================================================

class TestExecuteEnsemble:
    def test_execute_code_sprint(self):
        result = execute_ensemble(EnsembleType.CODE_SPRINT, {"task": "test"})
        assert result.success is True
        assert result.ensemble_type == EnsembleType.CODE_SPRINT

    def test_execute_all_types(self):
        for et in EnsembleType:
            result = execute_ensemble(et, {"task": "test"})
            assert isinstance(result, EnsembleResult)
            assert result.success is True


