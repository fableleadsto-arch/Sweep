from __future__ import annotations

from companion.brain import CompanionBrain, Intent


def _brain(settings) -> CompanionBrain:
    return CompanionBrain(settings)


def test_intent_scheduling(settings) -> None:
    assert _brain(settings)._detect_intent("can you book a meeting for tomorrow") == Intent.SCHEDULING


def test_intent_automation(settings) -> None:
    assert _brain(settings)._detect_intent("set up an n8n workflow for leads") == Intent.AUTOMATION


def test_intent_research(settings) -> None:
    assert _brain(settings)._detect_intent("find me new leads in SaaS") == Intent.RESEARCH


def test_intent_draft(settings) -> None:
    assert _brain(settings)._detect_intent("write a follow-up email") == Intent.DRAFT


def test_intent_crisis(settings) -> None:
    assert _brain(settings)._detect_intent("I feel so overwhelmed right now") == Intent.CRISIS


def test_intent_default_follow_up(settings) -> None:
    assert _brain(settings)._detect_intent("what do you think?") == Intent.FOLLOW_UP


def test_mood_detection(settings) -> None:
    brain = _brain(settings)
    assert "stressed" in brain._detect_mood("I'm swamped with deadlines")
    assert "rushed" in brain._detect_mood("need this ASAP")


def test_provider_health_without_keys(settings) -> None:
    health = _brain(settings).providers.health()
    assert health == {"gemini": False, "openai": False, "ollama": True, "anthropic": False}


def test_turn_returns_graceful_fallback_without_providers(settings) -> None:
    import asyncio

    settings.gemini_api_key = ""
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.ollama_base_url = ""
    brain = _brain(settings)

    async def run():
        return await brain.process_turn("user1", "hello there")

    result = asyncio.run(run())
    assert result.requires_approval is False
    assert result.provider == "none"
    assert "sorry" in result.text.lower()
