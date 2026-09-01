"""Agent-framework capabilities — LangChain, crewAI, Microsoft autogen and
Langflow.

Each capability lazy-imports the real framework and either builds a small
working chain/crew/chat or, for server products (Langflow), talks to the
configured instance over HTTP. LLM keys are read from Relay settings (never
from the model) so agent runs use the same configured providers.
"""

from __future__ import annotations

from typing import Any

from .common import CapabilityUnavailable, is_safe_http_url, load


# ── LangChain ────────────────────────────────────────────────────────────


def run_langchain(payload: dict[str, Any]) -> dict[str, Any]:
    """Build and run a small LangChain chain over the configured LLM."""
    params = payload.get("params") or {}
    settings = payload.get("_settings")
    mode = str(params.get("mode") or "chain").lower()

    load("langchain")
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"langchain-core import failed: {exc}") from exc

    if mode == "providers":
        return {
            "result": {
                "engine": "langchain",
                "configured": {
                    "openai": bool(settings and getattr(settings, "openai_api_key", "")),
                    "anthropic": bool(settings and getattr(settings, "anthropic_api_key", "")),
                    "ollama_base_url": bool(settings and getattr(settings, "ollama_base_url", "")),
                },
            },
            "summary": "LangChain installed; provider availability reported from server config.",
            "libraries_used": ["langchain"],
        }

    prompt_text = str(params.get("prompt") or "Answer the user's question.\nQuestion: {question}")
    question = str(params.get("question") or payload.get("data") or "")
    if not question and mode == "chain":
        raise ValueError("LangChain chain needs `params.question` (or `data`).")

    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception:  # noqa: BLE001
        ChatOllama = None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:  # noqa: BLE001
        ChatOpenAI = None

    llm = None
    if ChatOllama is not None and settings and getattr(settings, "ollama_base_url", ""):
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=str(params.get("model") or settings.ollama_model or "llama3.1:8b"),
            temperature=float(params.get("temperature") or 0.5),
        )
    elif ChatOpenAI is not None and settings and getattr(settings, "openai_api_key", ""):
        llm = ChatOpenAI(model=str(params.get("model") or settings.openai_model or "gpt-4o-mini"), temperature=float(params.get("temperature") or 0.5))
    if llm is None:
        return {
            "result": {
                "engine": "langchain",
                "ready": False,
                "note": "No configured LLM provider. Set OPENAI_API_KEY or OLLAMA_BASE_URL server-side, or use Ollama locally.",
            },
            "summary": "LangChain installed but no provider is configured for execution.",
            "libraries_used": ["langchain"],
        }

    prompt = ChatPromptTemplate.from_template(prompt_text[:2000])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question})
    return {
        "result": {"engine": "langchain", "mode": "chain", "answer": answer},
        "summary": f"LangChain chain answered ({len(answer)} chars).",
        "libraries_used": ["langchain"],
    }


# ── crewAI ───────────────────────────────────────────────────────────────


def run_crewai(payload: dict[str, Any]) -> dict[str, Any]:
    """Assemble and run a small crewAI crew (agents + task)."""
    params = payload.get("params") or {}
    settings = payload.get("_settings")

    crewai = load("crewai")
    from crewai import Agent, Crew, Task

    role = str(params.get("role") or "researcher")
    goal = str(params.get("goal") or "Answer the user's question with a short, accurate reply.")
    backstory = str(params.get("backstory") or "You are a helpful assistant.")
    task_desc = str(params.get("task") or params.get("prompt") or payload.get("data") or "")
    if not task_desc:
        raise ValueError("crewAI needs `params.task` describing what the agent should do.")

    llm = None
    if settings and getattr(settings, "openai_api_key", ""):
        llm = str(params.get("llm") or settings.openai_model or "gpt-4o-mini")
    elif settings and getattr(settings, "ollama_base_url", ""):
        llm = f"ollama/{params.get('model') or settings.ollama_model or 'llama3.1:8b'}"

    try:
        agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm, verbose=False)
        task = Task(description=task_desc[:2000], expected_output="a concise answer", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        output = crew.kickoff()
    except Exception as exc:  # noqa: BLE001 - LLM config issues surface cleanly
        return {
            "result": {"engine": "crewai", "ready": False, "error": str(exc)[:300]},
            "summary": f"crewAI kickoff failed: {str(exc)[:160]}",
            "libraries_used": ["crewai"],
        }

    return {
        "result": {"engine": "crewai", "role": role, "output": str(output)[:2000]},
        "summary": f"crewAI {role} agent completed the task.",
        "libraries_used": ["crewai"],
    }


# ── Microsoft autogen ────────────────────────────────────────────────────


def run_autogen(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a two-agent autogen conversation (user + assistant)."""
    params = payload.get("params") or {}
    settings = payload.get("_settings")

    load("autogen")
    try:
        # autogen 0.4+ moved agents into autogen_agentchat; older 0.2 kept
        # them at the top level. Support both so the capability works across
        # versions instead of failing on whichever one is installed.
        try:
            from autogen_agentchat.agents import AssistantAgent  # type: ignore
            from autogen_agentchat.ui import Console  # type: ignore

            AUTOGEN_04 = True
        except Exception:  # noqa: BLE001
            from autogen import AssistantAgent, UserProxyAgent  # type: ignore

            AUTOGEN_04 = False
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"autogen import failed: {exc}") from exc

    message = str(params.get("message") or params.get("task") or payload.get("data") or "")
    if not message:
        raise ValueError("autogen needs `params.message` (or `data`) to start the conversation.")

    api_key = settings and getattr(settings, "openai_api_key", "")
    model = str(params.get("model") or settings and getattr(settings, "openai_model", "") or "gpt-4o-mini")
    if not api_key:
        return {
            "result": {"engine": "autogen", "ready": False, "note": "autogen needs OPENAI_API_KEY server-side."},
            "summary": "autogen installed but no OpenAI key is configured.",
            "libraries_used": ["autogen"],
        }

    try:
        if AUTOGEN_04:
            import asyncio

            from autogen_agentchat.agents import AssistantAgent as _Assistant
            from autogen_agentchat.messages import TextMessage
            from autogen_agentchat.teams import RoundRobinGroupChat  # type: ignore
            from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore

            client = OpenAIChatCompletionClient(model=model, api_key=api_key)
            agent = _Assistant(name="assistant", model_client=client)

            async def _run() -> str:
                team = RoundRobinGroupChat([agent], max_turns=2)
                stream = team.run_stream(task=message[:2000])
                chunks: list[str] = []
                async for _event in stream:
                    if isinstance(_event, TextMessage) and _event.source != "assistant":
                        chunks.append(str(_event.content))
                return "\n".join(chunks)[:2000]

            summary = asyncio.run(_run())
        else:
            config_list = [{"model": model, "api_key": api_key}]
            assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
            user_proxy = UserProxyAgent("user_proxy", code_execution_config={"use_docker": False}, human_input_mode="NEVER")
            result = user_proxy.initiate_chat(assistant, message=message[:2000], max_consecutive_auto_reply=2)
            summary = str(result) if result else ""
    except Exception as exc:  # noqa: BLE001
        return {
            "result": {"engine": "autogen", "ready": False, "error": str(exc)[:300]},
            "summary": f"autogen chat failed: {str(exc)[:160]}",
            "libraries_used": ["autogen"],
        }

    return {
        "result": {"engine": "autogen", "model": model, "summary": summary[:2000]},
        "summary": "autogen two-agent conversation completed.",
        "libraries_used": ["autogen"],
    }


# ── Langflow (low-code platform, HTTP) ───────────────────────────────────


def run_langflow(payload: dict[str, Any]) -> dict[str, Any]:
    """List / run flows on a Langflow instance via its HTTP API."""
    import httpx

    settings = payload.get("_settings")
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "flows").lower()
    base_url = str(
        params.get("base_url")
        or (settings and getattr(settings, "langflow_base_url", ""))
        or ""
    ).rstrip("/")
    if not base_url:
        raise ValueError("Langflow needs `params.base_url` (or a LANGFLOW_BASE_URL setting) pointing at a public instance.")
    if not is_safe_http_url(base_url):
        raise ValueError("Langflow base_url must be a public http(s) endpoint (SSRF guard).")
    # Token comes from server settings only — never from model params.
    api_key = str(settings and getattr(settings, "langflow_api_key", "") or "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=30.0) as client:
            if mode == "flows":
                resp = client.get(f"{base_url}/api/v1/flows", headers=headers)
            elif mode == "run":
                flow_id = str(params.get("flow_id") or params.get("flow") or "")
                input_value = str(params.get("input") or payload.get("data") or "")
                if not flow_id or not input_value:
                    raise ValueError("Langflow `run` needs `params.flow_id` and `params.input`.")
                resp = client.post(
                    f"{base_url}/api/v1/run/{flow_id}",
                    headers=headers,
                    json={"input_value": input_value[:4000], "output_type": "chat", "input_type": "chat"},
                )
            else:
                raise ValueError("langflow mode must be 'flows' or 'run'.")
            resp.raise_for_status()
            body = resp.json()
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"Langflow at {base_url} unreachable: {exc}") from exc

    if mode == "flows":
        flows = body if isinstance(body, list) else body.get("flows", [])
        return {
            "result": {
                "engine": "langflow",
                "base_url": base_url,
                "flows": [{"id": f.get("id"), "name": f.get("name")} for f in flows[:25]],
                "count": len(flows) if isinstance(flows, list) else "unknown",
            },
            "summary": f"Langflow reported {len(flows) if isinstance(flows, list) else '?'} flow(s).",
            "libraries_used": ["langflow"],
        }

    outputs = body.get("outputs") or []
    text = ""
    if outputs and isinstance(outputs[0], dict):
        results = outputs[0].get("outputs") or []
        if results and isinstance(results[0], dict):
            text = str(results[0].get("message") or results[0].get("results") or results[0].get("text") or "")
    return {
        "result": {"engine": "langflow", "flow_id": params.get("flow_id"), "output": text[:2000]},
        "summary": f"Langflow flow produced {len(text)} chars.",
        "libraries_used": ["langflow"],
    }
