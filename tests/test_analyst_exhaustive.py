"""
Exhaustive tests for AnalystAgent (src/agents/analyst.py).

Covers initialization, every analysis method, modality detection,
intent inference, task decomposition, missing info identification,
assumptions, approach recommendation, tier suggestion, escalation,
confidence calculation, edge cases, and the create_analyst convenience function.
"""

import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from src.agents.analyst import AnalystAgent, create_analyst
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    SeverityLevel,
    ModalityType,
)


# ============================================================================
# Fixtures
# ============================================================================

FAKE_SYSTEM_PROMPT = "You are a test analyst agent."


@pytest.fixture
def agent():
    """Create an AnalystAgent with mocked system prompt file."""
    with patch("builtins.open", mock_open(read_data=FAKE_SYSTEM_PROMPT)):
        return AnalystAgent()


@pytest.fixture
def agent_missing_prompt():
    """Create an AnalystAgent when the system prompt file does not exist."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        return AnalystAgent()


# ============================================================================
# 1. Initialization
# ============================================================================

class TestInitialization:
    """Test __init__ with defaults, custom params, and system prompt loading."""

    def test_default_params(self, agent):
        assert agent.system_prompt_path == "config/agents/analyst/CLAUDE.md"
        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.max_turns == 30

    def test_custom_params(self):
        with patch("builtins.open", mock_open(read_data="custom")):
            a = AnalystAgent(
                system_prompt_path="custom/path.md",
                model="claude-3-opus-20240229",
                max_turns=10,
            )
        assert a.system_prompt_path == "custom/path.md"
        assert a.model == "claude-3-opus-20240229"
        assert a.max_turns == 10

    def test_system_prompt_loaded(self, agent):
        assert agent.system_prompt == FAKE_SYSTEM_PROMPT

    def test_system_prompt_fallback_on_missing_file(self, agent_missing_prompt):
        assert "Task Analyst" in agent_missing_prompt.system_prompt
        assert "Decompose" in agent_missing_prompt.system_prompt

    def test_modality_patterns_initialized(self, agent):
        assert ModalityType.CODE in agent.modality_patterns
        assert ModalityType.IMAGE in agent.modality_patterns
        assert ModalityType.DOCUMENT in agent.modality_patterns
        assert ModalityType.DATA in agent.modality_patterns
        assert ModalityType.TEXT not in agent.modality_patterns


# ============================================================================
# 2. _load_system_prompt
# ============================================================================

class TestLoadSystemPrompt:
    def test_reads_file_with_utf8(self):
        with patch("builtins.open", mock_open(read_data="prompt content")) as m:
            a = AnalystAgent()
            m.assert_called_with("config/agents/analyst/CLAUDE.md", "r", encoding="utf-8")
            assert a.system_prompt == "prompt content"

    def test_returns_default_on_file_not_found(self, agent_missing_prompt):
        expected = "You are the Task Analyst. Decompose user requests into structured requirements."
        assert agent_missing_prompt.system_prompt == expected


# ============================================================================
# 3. _prepare_request
# ============================================================================

class TestPrepareRequest:
    def test_no_attachments(self, agent):
        assert agent._prepare_request("hello", None) == "hello"

    def test_empty_attachments(self, agent):
        assert agent._prepare_request("hello", []) == "hello"

    def test_with_attachments(self, agent):
        result = agent._prepare_request("analyze", ["/path/to/file.py", "/other/data.csv"])
        assert "analyze" in result
        assert "[File: file.py]" in result
        assert "[File: data.csv]" in result

    def test_single_attachment(self, agent):
        result = agent._prepare_request("review", ["/tmp/code.js"])
        assert "[File: code.js]" in result
        assert "\n\n" in result


# ============================================================================
# 4. _detect_modality — every path
# ============================================================================

class TestDetectModality:
    """Test every modality detection path: file extensions, patterns, and default."""

    # --- File extension detection ---
    @pytest.mark.parametrize("ext,expected", [
        (".py", ModalityType.CODE),
        (".js", ModalityType.CODE),
        (".ts", ModalityType.CODE),
        (".java", ModalityType.CODE),
        (".cpp", ModalityType.CODE),
        (".c", ModalityType.CODE),
        (".go", ModalityType.CODE),
        (".rs", ModalityType.CODE),
    ])
    def test_code_file_extensions(self, agent, ext, expected):
        result = agent._detect_modality("check this", [f"/tmp/file{ext}"])
        assert result == expected

    @pytest.mark.parametrize("ext,expected", [
        (".png", ModalityType.IMAGE),
        (".jpg", ModalityType.IMAGE),
        (".jpeg", ModalityType.IMAGE),
        (".gif", ModalityType.IMAGE),
        (".webp", ModalityType.IMAGE),
    ])
    def test_image_file_extensions(self, agent, ext, expected):
        result = agent._detect_modality("look at this", [f"/tmp/img{ext}"])
        assert result == expected

    @pytest.mark.parametrize("ext,expected", [
        (".pdf", ModalityType.DOCUMENT),
        (".docx", ModalityType.DOCUMENT),
        (".doc", ModalityType.DOCUMENT),
        (".xlsx", ModalityType.DOCUMENT),
        (".pptx", ModalityType.DOCUMENT),
    ])
    def test_document_file_extensions(self, agent, ext, expected):
        result = agent._detect_modality("read this", [f"/tmp/doc{ext}"])
        assert result == expected

    @pytest.mark.parametrize("ext,expected", [
        (".json", ModalityType.DATA),
        (".yaml", ModalityType.DATA),
        (".yml", ModalityType.DATA),
        (".xml", ModalityType.DATA),
        (".csv", ModalityType.DATA),
    ])
    def test_data_file_extensions(self, agent, ext, expected):
        result = agent._detect_modality("process this", [f"/tmp/file{ext}"])
        assert result == expected

    def test_unknown_file_extension_falls_through(self, agent):
        # .txt is not in the extension lists, so falls to pattern matching
        result = agent._detect_modality("hello", ["/tmp/file.txt"])
        assert result == ModalityType.TEXT

    def test_file_extension_takes_priority_over_patterns(self, agent):
        # Request says "image" but file is .py -> CODE wins because extensions checked first
        result = agent._detect_modality("analyze this image", ["/tmp/code.py"])
        assert result == ModalityType.CODE

    def test_first_matching_file_wins(self, agent):
        # First file is .py (CODE), second is .csv (DATA) -> CODE wins
        result = agent._detect_modality("check", ["/tmp/a.py", "/tmp/b.csv"])
        assert result == ModalityType.CODE

    # --- Pattern-based detection (no file attachments) ---
    @pytest.mark.parametrize("request_text,expected", [
        ("write a function to sort", ModalityType.CODE),
        ("create a class for users", ModalityType.CODE),
        ("def my_function", ModalityType.CODE),
        ("import os", ModalityType.CODE),
        ("from pathlib import Path", ModalityType.CODE),
        ("implement a cache", ModalityType.CODE),
    ])
    def test_code_keyword_patterns(self, agent, request_text, expected):
        assert agent._detect_modality(request_text, None) == expected

    @pytest.mark.parametrize("request_text,expected", [
        ("analyze the image", ModalityType.IMAGE),
        ("show me a photo", ModalityType.IMAGE),
        ("generate a diagram", ModalityType.IMAGE),
        ("look at this screenshot", ModalityType.IMAGE),
    ])
    def test_image_keyword_patterns(self, agent, request_text, expected):
        assert agent._detect_modality(request_text, None) == expected

    @pytest.mark.parametrize("request_text,expected", [
        ("create a document for me", ModalityType.DOCUMENT),
        ("write a report", ModalityType.DOCUMENT),
        ("update the spreadsheet", ModalityType.DOCUMENT),
        ("make a presentation", ModalityType.DOCUMENT),
    ])
    def test_document_keyword_patterns(self, agent, request_text, expected):
        assert agent._detect_modality(request_text, None) == expected

    @pytest.mark.parametrize("request_text,expected", [
        ("check the data carefully", ModalityType.DATA),
        ("process this dataset", ModalityType.DATA),
        ("query the database", ModalityType.DATA),
        ("run this query please", ModalityType.DATA),
    ])
    def test_data_keyword_patterns(self, agent, request_text, expected):
        assert agent._detect_modality(request_text, None) == expected

    def test_default_text_modality(self, agent):
        assert agent._detect_modality("hello world", None) == ModalityType.TEXT

    def test_default_text_modality_no_attachments(self, agent):
        assert agent._detect_modality("just chat with me", []) == ModalityType.TEXT

    # Pattern with file extension in the text (regex patterns like \.py$)
    def test_code_extension_regex_in_request(self, agent):
        result = agent._detect_modality("check main.py", None)
        assert result == ModalityType.CODE

    def test_image_extension_regex_in_request(self, agent):
        result = agent._detect_modality("look at logo.png", None)
        assert result == ModalityType.IMAGE

    def test_document_extension_regex_in_request(self, agent):
        result = agent._detect_modality("review spec.pdf", None)
        assert result == ModalityType.DOCUMENT

    def test_data_extension_regex_in_request(self, agent):
        result = agent._detect_modality("parse config.yaml", None)
        assert result == ModalityType.DATA


# ============================================================================
# 5. _extract_literal_request
# ============================================================================

class TestExtractLiteralRequest:
    def test_strips_whitespace(self, agent):
        assert agent._extract_literal_request("  hello  ") == "hello"

    def test_preserves_content(self, agent):
        assert agent._extract_literal_request("Write a REST API") == "Write a REST API"

    def test_empty_string(self, agent):
        assert agent._extract_literal_request("") == ""

    def test_only_whitespace(self, agent):
        assert agent._extract_literal_request("   \n\t  ") == ""

    def test_special_characters(self, agent):
        text = "Fix bug #123 <script>alert('xss')</script>"
        assert agent._extract_literal_request(text) == text


# ============================================================================
# 6. _infer_intent — all patterns and default
# ============================================================================

class TestInferIntent:
    @pytest.mark.parametrize("keyword,expected_fragment", [
        ("create", "generate or create something new"),
        ("fix", "resolve a problem or bug"),
        ("explain", "understanding or clarification"),
        ("improve", "enhance existing code/process"),
        ("analyze", "detailed examination or insight"),
        ("convert", "transform one format to another"),
        ("compare", "understand differences"),
        ("implement", "build a specific feature/system"),
    ])
    def test_intent_patterns(self, agent, keyword, expected_fragment):
        result = agent._infer_intent(f"Please {keyword} the thing", None)
        assert expected_fragment in result

    def test_case_insensitive(self, agent):
        result = agent._infer_intent("CREATE a new app", None)
        assert "generate or create" in result

    def test_default_intent(self, agent):
        result = agent._infer_intent("hello world", None)
        assert result.startswith("User wants assistance with:")
        assert "hello world" in result

    def test_default_intent_truncates_long_request(self, agent):
        long_request = "x" * 200
        result = agent._infer_intent(long_request, None)
        # Should truncate to 100 chars plus "..."
        assert result.endswith("...")
        assert len(result) < 200

    def test_context_param_accepted(self, agent):
        # Context is accepted but currently unused in the simple implementation
        result = agent._infer_intent("create something", {"key": "value"})
        assert "generate or create" in result

    def test_none_context(self, agent):
        result = agent._infer_intent("fix this bug", None)
        assert "resolve a problem" in result

    def test_first_matching_keyword_wins(self, agent):
        # "create" comes before "fix" in the dict iteration
        result = agent._infer_intent("create a fix for this", None)
        assert "generate or create" in result


# ============================================================================
# 7. _decompose_tasks — all branches
# ============================================================================

class TestDecomposeTasks:
    def test_api_keyword(self, agent):
        tasks = agent._decompose_tasks("build an api", "intent", ModalityType.CODE)
        descriptions = [t.description for t in tasks]
        assert len(tasks) == 4
        assert "Define data models and schema" in descriptions
        assert "Design API endpoints and routes" in descriptions
        assert "Implement endpoint logic" in descriptions
        assert "Add authentication and validation" in descriptions

    def test_endpoint_keyword(self, agent):
        tasks = agent._decompose_tasks("create an endpoint", "intent", ModalityType.CODE)
        assert len(tasks) == 4
        assert tasks[0].description == "Define data models and schema"

    def test_api_task_dependencies(self, agent):
        tasks = agent._decompose_tasks("build api", "intent", ModalityType.CODE)
        assert tasks[0].dependencies == []
        assert tasks[1].dependencies == ["Define data models and schema"]
        assert tasks[2].dependencies == ["Design API endpoints and routes"]
        assert tasks[3].dependencies == ["Implement endpoint logic"]

    def test_api_task_complexities(self, agent):
        tasks = agent._decompose_tasks("build api", "intent", ModalityType.CODE)
        assert tasks[0].estimated_complexity == "medium"
        assert tasks[1].estimated_complexity == "high"
        assert tasks[2].estimated_complexity == "high"
        assert tasks[3].estimated_complexity == "medium"

    def test_test_keyword(self, agent):
        tasks = agent._decompose_tasks("write test for module", "intent", ModalityType.CODE)
        assert len(tasks) == 3
        descriptions = [t.description for t in tasks]
        assert "Analyze code to identify test scenarios" in descriptions
        assert "Design test cases with coverage" in descriptions
        assert "Implement test code" in descriptions

    def test_document_keyword(self, agent):
        tasks = agent._decompose_tasks("write a document for the module", "intent", ModalityType.TEXT)
        assert len(tasks) == 3
        descriptions = [t.description for t in tasks]
        assert "Analyze code/features to document" in descriptions
        assert "Structure documentation outline" in descriptions
        assert "Generate documentation content" in descriptions

    def test_docs_keyword(self, agent):
        tasks = agent._decompose_tasks("update the docs", "intent", ModalityType.TEXT)
        assert len(tasks) == 3
        assert tasks[0].description == "Analyze code/features to document"

    def test_bug_keyword(self, agent):
        tasks = agent._decompose_tasks("there is a bug in login", "intent", ModalityType.CODE)
        assert len(tasks) == 4
        descriptions = [t.description for t in tasks]
        assert "Analyze the error/bug" in descriptions
        assert "Identify root cause" in descriptions
        assert "Implement fix" in descriptions
        assert "Verify fix resolves issue" in descriptions

    def test_fix_keyword(self, agent):
        tasks = agent._decompose_tasks("fix the broken feature", "intent", ModalityType.CODE)
        assert len(tasks) == 4
        assert tasks[0].description == "Analyze the error/bug"

    def test_error_keyword(self, agent):
        tasks = agent._decompose_tasks("error in production", "intent", ModalityType.CODE)
        assert len(tasks) == 4
        assert tasks[0].description == "Analyze the error/bug"

    def test_generic_decomposition(self, agent):
        tasks = agent._decompose_tasks("hello world", "some intent", ModalityType.TEXT)
        assert len(tasks) == 3
        assert tasks[0].description == "Understand and analyze requirements"
        assert tasks[0].dependencies == []
        assert tasks[0].estimated_complexity == "low"
        assert "some intent" in tasks[1].description
        assert tasks[1].dependencies == ["Understand and analyze requirements"]
        assert tasks[1].estimated_complexity == "high"
        assert tasks[2].description == "Review and validate output"
        assert tasks[2].estimated_complexity == "medium"

    def test_api_takes_priority_over_test(self, agent):
        # "api" check comes before "test" in the if/elif chain
        tasks = agent._decompose_tasks("test the api", "intent", ModalityType.CODE)
        # "api" is in the string, so API branch wins
        assert len(tasks) == 4
        assert tasks[0].description == "Define data models and schema"

    def test_returns_subtask_instances(self, agent):
        tasks = agent._decompose_tasks("hello", "intent", ModalityType.TEXT)
        for t in tasks:
            assert isinstance(t, SubTask)


# ============================================================================
# 8. _identify_missing_info
# ============================================================================

class TestIdentifyMissingInfo:
    def test_api_without_auth(self, agent):
        sub_tasks = []
        result = agent._identify_missing_info("build an api", sub_tasks, ModalityType.CODE)
        auth_items = [m for m in result if m.requirement == "Authentication method"]
        assert len(auth_items) == 1
        assert auth_items[0].severity == SeverityLevel.CRITICAL
        assert auth_items[0].default_assumption == "JWT-based authentication"

    def test_api_with_auth_no_critical(self, agent):
        result = agent._identify_missing_info("build an api with auth", [], ModalityType.CODE)
        auth_items = [m for m in result if m.requirement == "Authentication method"]
        assert len(auth_items) == 0

    def test_database_without_tech(self, agent):
        result = agent._identify_missing_info("setup a database", [], ModalityType.DATA)
        db_items = [m for m in result if m.requirement == "Database technology"]
        assert len(db_items) == 1
        assert db_items[0].severity == SeverityLevel.CRITICAL
        assert db_items[0].default_assumption == "PostgreSQL"

    def test_db_shorthand_without_tech(self, agent):
        result = agent._identify_missing_info("connect to db", [], ModalityType.DATA)
        db_items = [m for m in result if m.requirement == "Database technology"]
        assert len(db_items) == 1

    def test_database_with_postgres(self, agent):
        result = agent._identify_missing_info("setup postgres database", [], ModalityType.DATA)
        db_items = [m for m in result if m.requirement == "Database technology"]
        assert len(db_items) == 0

    def test_database_with_mysql(self, agent):
        result = agent._identify_missing_info("setup mysql database", [], ModalityType.DATA)
        db_items = [m for m in result if m.requirement == "Database technology"]
        assert len(db_items) == 0

    def test_database_with_mongo(self, agent):
        result = agent._identify_missing_info("setup mongo database", [], ModalityType.DATA)
        db_items = [m for m in result if m.requirement == "Database technology"]
        assert len(db_items) == 0

    def test_deploy_without_cloud(self, agent):
        result = agent._identify_missing_info("deploy the app", [], ModalityType.CODE)
        deploy_items = [m for m in result if m.requirement == "Deployment target/platform"]
        assert len(deploy_items) == 1
        assert deploy_items[0].severity == SeverityLevel.IMPORTANT
        assert deploy_items[0].default_assumption == "Docker containers"

    def test_production_without_cloud(self, agent):
        result = agent._identify_missing_info("push to production", [], ModalityType.CODE)
        deploy_items = [m for m in result if m.requirement == "Deployment target/platform"]
        assert len(deploy_items) == 1

    def test_deploy_with_cloud(self, agent):
        result = agent._identify_missing_info("deploy to cloud", [], ModalityType.CODE)
        deploy_items = [m for m in result if m.requirement == "Deployment target/platform"]
        assert len(deploy_items) == 0

    def test_no_test_mentioned_adds_nice_to_have(self, agent):
        result = agent._identify_missing_info("build a feature", [], ModalityType.CODE)
        test_items = [m for m in result if m.requirement == "Testing requirements"]
        assert len(test_items) == 1
        assert test_items[0].severity == SeverityLevel.NICE_TO_HAVE
        assert test_items[0].default_assumption == "Unit tests for core functionality"

    def test_test_mentioned_no_nice_to_have(self, agent):
        result = agent._identify_missing_info("write a test for login", [], ModalityType.CODE)
        test_items = [m for m in result if m.requirement == "Testing requirements"]
        assert len(test_items) == 0

    def test_all_missing_info_are_missinginfo_instances(self, agent):
        result = agent._identify_missing_info("build an api with database and deploy", [], ModalityType.CODE)
        for item in result:
            assert isinstance(item, MissingInfo)

    def test_combined_api_db_deploy(self, agent):
        result = agent._identify_missing_info(
            "build an api with database and deploy", [], ModalityType.CODE
        )
        requirements = [m.requirement for m in result]
        assert "Authentication method" in requirements
        assert "Database technology" in requirements
        assert "Deployment target/platform" in requirements
        assert "Testing requirements" in requirements


# ============================================================================
# 9. _generate_assumptions
# ============================================================================

class TestGenerateAssumptions:
    def test_common_assumptions_always_present(self, agent):
        result = agent._generate_assumptions("hello", [])
        assert "User has necessary permissions/access" in result
        assert "Standard best practices apply unless specified" in result
        assert "Code follows project conventions" in result

    def test_assumptions_from_missing_info(self, agent):
        missing = [
            MissingInfo(
                requirement="Auth method",
                severity=SeverityLevel.CRITICAL,
                impact="big",
                default_assumption="JWT tokens",
            )
        ]
        result = agent._generate_assumptions("build api", missing)
        assert "Assuming Auth method: JWT tokens" in result

    def test_multiple_missing_info_assumptions(self, agent):
        missing = [
            MissingInfo(
                requirement="DB tech",
                severity=SeverityLevel.CRITICAL,
                impact="impact",
                default_assumption="PostgreSQL",
            ),
            MissingInfo(
                requirement="Deploy target",
                severity=SeverityLevel.IMPORTANT,
                impact="impact",
                default_assumption="Docker containers",
            ),
        ]
        result = agent._generate_assumptions("build app", missing)
        assert "Assuming DB tech: PostgreSQL" in result
        assert "Assuming Deploy target: Docker containers" in result

    def test_missing_info_without_default_assumption(self, agent):
        missing = [
            MissingInfo(
                requirement="Something",
                severity=SeverityLevel.IMPORTANT,
                impact="impact",
                default_assumption=None,
            )
        ]
        result = agent._generate_assumptions("do stuff", missing)
        # Should only have the 3 common assumptions, no "Assuming Something:" line
        assuming_lines = [a for a in result if a.startswith("Assuming")]
        assert len(assuming_lines) == 0

    def test_order_missing_before_common(self, agent):
        missing = [
            MissingInfo(
                requirement="X",
                severity=SeverityLevel.CRITICAL,
                impact="impact",
                default_assumption="Y",
            )
        ]
        result = agent._generate_assumptions("request", missing)
        # Missing info assumptions come first
        assert result[0] == "Assuming X: Y"
        assert result[-1] == "Code follows project conventions"

    def test_empty_request(self, agent):
        result = agent._generate_assumptions("", [])
        assert len(result) == 3  # Just the common assumptions


# ============================================================================
# 10. _recommend_approach
# ============================================================================

class TestRecommendApproach:
    def test_critical_missing_info_overrides_all(self, agent):
        critical = [
            MissingInfo(
                requirement="Auth",
                severity=SeverityLevel.CRITICAL,
                impact="big",
                default_assumption="JWT",
            )
        ]
        result = agent._recommend_approach("build api", [], ModalityType.CODE, critical)
        assert "Clarify critical requirements" in result

    def test_code_modality_no_critical(self, agent):
        result = agent._recommend_approach("write code", [], ModalityType.CODE, [])
        assert "code-generation" in result
        assert "Code Reviewer" in result

    def test_document_modality_no_critical(self, agent):
        result = agent._recommend_approach("write a report", [], ModalityType.DOCUMENT, [])
        assert "document-creation" in result
        assert "Formatter" in result

    def test_data_modality_no_critical(self, agent):
        result = agent._recommend_approach("analyze data", [], ModalityType.DATA, [])
        assert "data-analysis" in result
        assert "Verifier" in result

    def test_api_task_type(self, agent):
        result = agent._recommend_approach("build api endpoints", [], ModalityType.TEXT, [])
        assert "schema first" in result
        assert "endpoints" in result

    def test_test_task_type(self, agent):
        result = agent._recommend_approach("write test suite", [], ModalityType.TEXT, [])
        assert "coverage" in result
        assert "test" in result.lower()

    def test_bug_task_type(self, agent):
        result = agent._recommend_approach("fix this bug", [], ModalityType.TEXT, [])
        assert "Reproduce" in result or "root cause" in result

    def test_default_recommendation(self, agent):
        result = agent._recommend_approach("hello world", [], ModalityType.TEXT, [])
        assert "standard pipeline" in result.lower() or "analyze" in result.lower()

    def test_important_missing_does_not_trigger_critical_path(self, agent):
        important = [
            MissingInfo(
                requirement="Deploy",
                severity=SeverityLevel.IMPORTANT,
                impact="impact",
                default_assumption="Docker",
            )
        ]
        result = agent._recommend_approach("write code", [], ModalityType.CODE, important)
        # Should NOT trigger "Clarify critical requirements"
        assert "Clarify critical" not in result

    def test_nice_to_have_missing_does_not_trigger_critical_path(self, agent):
        nice = [
            MissingInfo(
                requirement="Tests",
                severity=SeverityLevel.NICE_TO_HAVE,
                impact="impact",
                default_assumption="Unit tests",
            )
        ]
        result = agent._recommend_approach("write code", [], ModalityType.CODE, nice)
        assert "Clarify critical" not in result

    def test_image_modality_gets_default(self, agent):
        # IMAGE is not handled specially, falls through to task-type or default
        result = agent._recommend_approach("show me something", [], ModalityType.IMAGE, [])
        assert "standard pipeline" in result.lower()


# ============================================================================
# 11. _suggest_tier
# ============================================================================

class TestSuggestTier:
    @pytest.mark.parametrize("keyword", [
        "security", "compliance", "adversarial", "attack", "critical",
    ])
    def test_tier_4_keywords(self, agent, keyword):
        result = agent._suggest_tier(f"review {keyword} concerns", [], [])
        assert result == 4

    @pytest.mark.parametrize("keyword", [
        "architecture", "design pattern", "domain expert", "specialist",
    ])
    def test_tier_3_keywords(self, agent, keyword):
        result = agent._suggest_tier(f"need {keyword} advice", [], [])
        assert result == 3

    def test_tier_2_many_subtasks(self, agent):
        tasks = [
            SubTask(description="a", dependencies=[], estimated_complexity="low"),
            SubTask(description="b", dependencies=[], estimated_complexity="low"),
            SubTask(description="c", dependencies=[], estimated_complexity="low"),
        ]
        result = agent._suggest_tier("simple request", tasks, [])
        assert result == 2

    def test_tier_2_has_missing_info(self, agent):
        missing = [
            MissingInfo(
                requirement="something",
                severity=SeverityLevel.NICE_TO_HAVE,
                impact="impact",
                default_assumption="default",
            )
        ]
        result = agent._suggest_tier("simple request", [], missing)
        assert result == 2

    def test_tier_1_simple_request(self, agent):
        # At most 2 sub_tasks and no missing info
        tasks = [
            SubTask(description="a", dependencies=[], estimated_complexity="low"),
            SubTask(description="b", dependencies=[], estimated_complexity="low"),
        ]
        result = agent._suggest_tier("hello", tasks, [])
        assert result == 1

    def test_tier_1_no_tasks_no_missing(self, agent):
        result = agent._suggest_tier("hello", [], [])
        assert result == 1

    def test_tier_4_takes_priority_over_tier_3(self, agent):
        result = agent._suggest_tier("security architecture review", [], [])
        assert result == 4

    def test_tier_4_takes_priority_over_tier_2(self, agent):
        missing = [
            MissingInfo(
                requirement="x",
                severity=SeverityLevel.CRITICAL,
                impact="y",
                default_assumption="z",
            )
        ]
        result = agent._suggest_tier("security audit", [], missing)
        assert result == 4

    def test_tier_3_takes_priority_over_tier_2(self, agent):
        missing = [
            MissingInfo(
                requirement="x",
                severity=SeverityLevel.CRITICAL,
                impact="y",
                default_assumption="z",
            )
        ]
        result = agent._suggest_tier("architecture review", [], missing)
        assert result == 3

    def test_case_insensitive(self, agent):
        result = agent._suggest_tier("SECURITY review", [], [])
        assert result == 4


# ============================================================================
# 12. _check_escalation
# ============================================================================

class TestCheckEscalation:
    @pytest.mark.parametrize("keyword", [
        "complex", "complicated", "uncertain", "may need",
        "depends on", "multiple factors",
    ])
    def test_escalation_keywords(self, agent, keyword):
        assert agent._check_escalation(f"this is {keyword} work", []) is True

    def test_no_escalation_keywords(self, agent):
        assert agent._check_escalation("simple task", []) is False

    def test_case_insensitive(self, agent):
        # The method uses .lower(), so uppercase should still match
        assert agent._check_escalation("This is COMPLEX", []) is True

    def test_empty_request(self, agent):
        assert agent._check_escalation("", []) is False

    def test_subtasks_param_accepted(self, agent):
        tasks = [SubTask(description="a", dependencies=[], estimated_complexity="low")]
        assert agent._check_escalation("simple", tasks) is False


# ============================================================================
# 13. _calculate_confidence
# ============================================================================

class TestCalculateConfidence:
    def test_base_confidence(self, agent):
        # No missing info, fewer than 3 subtasks
        result = agent._calculate_confidence([], [])
        assert result == pytest.approx(0.8)

    def test_increased_for_well_structured(self, agent):
        tasks = [
            SubTask(description="a", dependencies=[], estimated_complexity="low"),
            SubTask(description="b", dependencies=[], estimated_complexity="low"),
            SubTask(description="c", dependencies=[], estimated_complexity="low"),
        ]
        result = agent._calculate_confidence(tasks, [])
        assert result == pytest.approx(0.9)

    def test_decreased_for_critical_missing(self, agent):
        missing = [
            MissingInfo(
                requirement="x",
                severity=SeverityLevel.CRITICAL,
                impact="y",
                default_assumption="z",
            )
        ]
        result = agent._calculate_confidence([], missing)
        assert result == pytest.approx(0.7)

    def test_two_critical_missing(self, agent):
        missing = [
            MissingInfo(requirement="a", severity=SeverityLevel.CRITICAL, impact="y", default_assumption="z"),
            MissingInfo(requirement="b", severity=SeverityLevel.CRITICAL, impact="y", default_assumption="z"),
        ]
        result = agent._calculate_confidence([], missing)
        assert result == pytest.approx(0.6)

    def test_critical_and_well_structured(self, agent):
        tasks = [
            SubTask(description="a", dependencies=[], estimated_complexity="low"),
            SubTask(description="b", dependencies=[], estimated_complexity="low"),
            SubTask(description="c", dependencies=[], estimated_complexity="low"),
        ]
        missing = [
            MissingInfo(requirement="x", severity=SeverityLevel.CRITICAL, impact="y", default_assumption="z"),
        ]
        result = agent._calculate_confidence(tasks, missing)
        # 0.8 - 0.1 + 0.1 = 0.8
        assert result == pytest.approx(0.8)

    def test_important_missing_does_not_reduce(self, agent):
        missing = [
            MissingInfo(requirement="x", severity=SeverityLevel.IMPORTANT, impact="y", default_assumption="z"),
        ]
        result = agent._calculate_confidence([], missing)
        assert result == pytest.approx(0.8)

    def test_nice_to_have_missing_does_not_reduce(self, agent):
        missing = [
            MissingInfo(requirement="x", severity=SeverityLevel.NICE_TO_HAVE, impact="y", default_assumption="z"),
        ]
        result = agent._calculate_confidence([], missing)
        assert result == pytest.approx(0.8)

    def test_floor_at_zero(self, agent):
        # 10 critical missing items: 0.8 - 1.0 = -0.2 -> clamped to 0.0
        missing = [
            MissingInfo(requirement=f"x{i}", severity=SeverityLevel.CRITICAL, impact="y", default_assumption="z")
            for i in range(10)
        ]
        result = agent._calculate_confidence([], missing)
        assert result == pytest.approx(0.0)

    def test_ceiling_at_one(self, agent):
        # Many well-structured tasks, no critical -> 0.8 + 0.1 = 0.9, capped at 1.0
        tasks = [SubTask(description=f"t{i}", dependencies=[], estimated_complexity="low") for i in range(100)]
        result = agent._calculate_confidence(tasks, [])
        assert result == pytest.approx(0.9)
        assert result <= 1.0

    def test_exactly_two_subtasks_no_bonus(self, agent):
        tasks = [
            SubTask(description="a", dependencies=[], estimated_complexity="low"),
            SubTask(description="b", dependencies=[], estimated_complexity="low"),
        ]
        result = agent._calculate_confidence(tasks, [])
        assert result == pytest.approx(0.8)


# ============================================================================
# 14. Full analyze() method
# ============================================================================

class TestAnalyze:
    def test_returns_task_intelligence_report(self, agent):
        result = agent.analyze("build an api")
        assert isinstance(result, TaskIntelligenceReport)

    def test_analyze_simple_request(self, agent):
        result = agent.analyze("hello world")
        assert result.literal_request == "hello world"
        assert result.modality == ModalityType.TEXT
        assert result.escalation_needed is False
        assert 0.0 <= result.confidence <= 1.0
        assert 1 <= result.suggested_tier <= 4

    def test_analyze_api_request(self, agent):
        result = agent.analyze("implement a REST api for users")
        assert result.modality == ModalityType.CODE  # "implement" matches CODE pattern
        assert len(result.sub_tasks) == 4
        # Should detect api without auth -> critical missing
        auth_missing = [m for m in result.missing_info if m.requirement == "Authentication method"]
        assert len(auth_missing) == 1

    def test_analyze_with_context(self, agent):
        result = agent.analyze("create something", context={"key": "val"})
        assert "generate or create" in result.inferred_intent

    def test_analyze_with_none_context(self, agent):
        result = agent.analyze("fix bug", context=None)
        assert "resolve a problem" in result.inferred_intent

    def test_analyze_with_file_attachments(self, agent):
        result = agent.analyze("review this", file_attachments=["/tmp/code.py"])
        assert result.modality == ModalityType.CODE

    def test_analyze_complex_request(self, agent):
        result = agent.analyze("this is a complex security api with database and deploy")
        assert result.suggested_tier == 4  # "security" keyword
        assert result.escalation_needed is True  # "complex" keyword
        assert len(result.missing_info) > 0

    def test_analyze_assumptions_populated(self, agent):
        result = agent.analyze("build a feature")
        assert len(result.assumptions) >= 3  # At least the 3 common ones

    def test_analyze_recommended_approach_populated(self, agent):
        result = agent.analyze("write code for me")
        assert len(result.recommended_approach) > 0

    def test_analyze_bug_fix_path(self, agent):
        result = agent.analyze("fix the login bug")
        assert len(result.sub_tasks) == 4
        assert result.sub_tasks[0].description == "Analyze the error/bug"

    def test_analyze_test_path(self, agent):
        result = agent.analyze("write test suite")
        assert len(result.sub_tasks) == 3
        # "test" mentioned -> no NICE_TO_HAVE testing requirement
        test_missing = [m for m in result.missing_info if m.requirement == "Testing requirements"]
        assert len(test_missing) == 0

    def test_analyze_document_path(self, agent):
        result = agent.analyze("write the docs for this module")
        assert len(result.sub_tasks) == 3
        assert result.sub_tasks[0].description == "Analyze code/features to document"


# ============================================================================
# 15. Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_request(self, agent):
        result = agent.analyze("")
        assert result.literal_request == ""
        assert isinstance(result, TaskIntelligenceReport)

    def test_very_long_request(self, agent):
        long_request = "build " * 10000
        result = agent.analyze(long_request)
        assert isinstance(result, TaskIntelligenceReport)

    def test_special_characters(self, agent):
        result = agent.analyze("fix bug #123 <script>alert('xss')</script> @user $$$")
        assert isinstance(result, TaskIntelligenceReport)
        assert "fix" in result.literal_request

    def test_unicode_characters(self, agent):
        result = agent.analyze("create a function for handling emoji data")
        assert isinstance(result, TaskIntelligenceReport)

    def test_newlines_in_request(self, agent):
        result = agent.analyze("line one\nline two\nline three")
        assert isinstance(result, TaskIntelligenceReport)

    def test_tabs_in_request(self, agent):
        result = agent.analyze("step1\tstep2\tstep3")
        assert isinstance(result, TaskIntelligenceReport)

    def test_none_context_none_attachments(self, agent):
        result = agent.analyze("hello", context=None, file_attachments=None)
        assert isinstance(result, TaskIntelligenceReport)

    def test_empty_context_dict(self, agent):
        result = agent.analyze("hello", context={})
        assert isinstance(result, TaskIntelligenceReport)

    def test_empty_attachments_list(self, agent):
        result = agent.analyze("hello", file_attachments=[])
        assert result.modality == ModalityType.TEXT

    def test_whitespace_only_request(self, agent):
        result = agent.analyze("   \n\t   ")
        assert result.literal_request == ""

    def test_request_with_only_keyword(self, agent):
        result = agent.analyze("fix")
        assert "resolve a problem" in result.inferred_intent

    def test_modality_detection_case_insensitive(self, agent):
        result = agent._detect_modality("CHECK THIS IMAGE", None)
        assert result == ModalityType.IMAGE

    def test_multiple_file_types_first_recognized_wins(self, agent):
        # .xyz is unknown, .csv is DATA
        result = agent._detect_modality("check", ["/tmp/file.xyz", "/tmp/data.csv"])
        assert result == ModalityType.DATA

    def test_all_unknown_extensions_fall_to_patterns(self, agent):
        result = agent._detect_modality("write a function", ["/tmp/readme.xyz"])
        assert result == ModalityType.CODE  # Pattern match on "function"


# ============================================================================
# 16. create_analyst convenience function
# ============================================================================

class TestCreateAnalyst:
    def test_returns_analyst_agent(self):
        with patch("builtins.open", mock_open(read_data="prompt")):
            a = create_analyst()
        assert isinstance(a, AnalystAgent)

    def test_default_params(self):
        with patch("builtins.open", mock_open(read_data="prompt")):
            a = create_analyst()
        assert a.system_prompt_path == "config/agents/analyst/CLAUDE.md"
        assert a.model == "claude-3-5-sonnet-20241022"

    def test_custom_params(self):
        with patch("builtins.open", mock_open(read_data="prompt")):
            a = create_analyst(
                system_prompt_path="custom/path.md",
                model="claude-3-opus-20240229",
            )
        assert a.system_prompt_path == "custom/path.md"
        assert a.model == "claude-3-opus-20240229"

    def test_max_turns_default(self):
        with patch("builtins.open", mock_open(read_data="prompt")):
            a = create_analyst()
        assert a.max_turns == 30


# ============================================================================
# 17. Schema validation (integration with pydantic)
# ============================================================================

class TestSchemaIntegration:
    def test_report_fields_complete(self, agent):
        result = agent.analyze("build an api")
        # Verify all fields are set
        assert result.literal_request is not None
        assert result.inferred_intent is not None
        assert result.sub_tasks is not None
        assert result.missing_info is not None
        assert result.assumptions is not None
        assert result.modality is not None
        assert result.recommended_approach is not None
        assert result.escalation_needed is not None
        assert result.suggested_tier is not None
        assert result.confidence is not None

    def test_report_serializes_to_dict(self, agent):
        result = agent.analyze("hello")
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "literal_request" in d
        assert "sub_tasks" in d

    def test_report_serializes_to_json(self, agent):
        result = agent.analyze("hello")
        j = result.model_json_schema()
        assert isinstance(j, dict)

    def test_subtask_schema(self):
        st = SubTask(description="test", dependencies=["a"], estimated_complexity="high")
        assert st.description == "test"
        assert st.dependencies == ["a"]
        assert st.estimated_complexity == "high"

    def test_missing_info_schema(self):
        mi = MissingInfo(
            requirement="auth",
            severity=SeverityLevel.CRITICAL,
            impact="big",
            default_assumption="JWT",
        )
        assert mi.requirement == "auth"
        assert mi.severity == SeverityLevel.CRITICAL

    def test_modality_enum_values(self):
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.CODE.value == "code"
        assert ModalityType.DOCUMENT.value == "document"
        assert ModalityType.DATA.value == "data"

    def test_severity_enum_values(self):
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.IMPORTANT.value == "important"
        assert SeverityLevel.NICE_TO_HAVE.value == "nice_to_have"
