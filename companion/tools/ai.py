"""Advanced AI capabilities — PyTorch, Transformers, TensorFlow/Keras, JAX,
LiteLLM and LlamaIndex.

Every framework here is a heavy/optional dependency: each tool lazy-imports
its framework only when invoked, so a plain chat never pays the import cost.
"""

from __future__ import annotations

from typing import Any, Optional

from .common import CapabilityUnavailable, as_rows, load


def run_deep_learning(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a small neural network (PyTorch) on tabular data."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "fit").lower()

    torch = load("torch")
    from .common import module_available

    if not module_available("torch"):
        raise CapabilityUnavailable("PyTorch is required for deep learning.")

    if mode == "autograd":
        x = float(params.get("x") or 2.0)
        var = torch.tensor(x, requires_grad=True)
        loss = var ** 3 + 2 * var
        loss.backward()
        return {
            "result": {
                "mode": "autograd",
                "x": x,
                "f(x)=x^3+2x": float(loss.item()),
                "f'(x)": float(var.grad.item()),
            },
            "summary": f"Autograd: at x={x:g}, f={float(loss.item()):.4g}, f'={float(var.grad.item()):.4g}.",
            "libraries_used": ["torch"],
        }

    if mode == "device":
        return {
            "result": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            "summary": f"PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}.",
            "libraries_used": ["torch"],
        }

    import numpy as np

    data = payload.get("data")
    if data is None:
        raise ValueError("deep-learning 'fit' needs tabular `data` with a `params.target` column.")
    X, y, target, cols, _ = _tabular_for_ai(data, params, np)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) < 8:
        raise ValueError("Need at least 8 rows to train a small network.")

    torch.manual_seed(42)
    from torch import nn

    hidden = int(params.get("hidden") or 32)
    epochs = int(params.get("epochs") or 60)
    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    split = int(len(X) * 0.8)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    X_tr, y_tr = Xt[:split], yt[:split]
    X_te, y_te = Xt[split:], yt[split:]

    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        test_pred = model(X_te)
        test_loss = float(loss_fn(test_pred, y_te).item())
        pred_values = [round(float(v), 4) for v in test_pred.flatten()[:10]]

    return {
        "result": {
            "mode": "fit",
            "engine": "pytorch",
            "target": target,
            "rows": len(X),
            "epochs": epochs,
            "final_train_loss": round(losses[-1], 6),
            "test_mse": round(test_loss, 6),
            "sample_predictions": pred_values,
        },
        "summary": (
            f"Trained a {X.shape[1]}→{hidden}→1 MLP on {len(X)} rows "
            f"({epochs} epochs) — final train MSE {losses[-1]:.5g}, test MSE {test_loss:.5g}."
        ),
        "libraries_used": ["torch", "numpy"],
    }


def run_local_llm(payload: dict[str, Any]) -> dict[str, Any]:
    """Run an open-source model locally via Hugging Face Transformers."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "generate").lower()

    transformers = load("transformers")
    from .common import module_available

    if not (module_available("torch") or module_available("tensorflow")):
        raise CapabilityUnavailable(
            "Local Transformers models need a backend (torch or tensorflow)."
        )

    if mode == "embed":
        from transformers import AutoTokenizer, AutoModel

        model_name = str(params.get("model") or "sentence-transformers/all-MiniLM-L6-v2")
        texts = payload.get("data")
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = [str(texts or "")]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vectors = []
        for text in texts[:8]:
            encoded = tokenizer(text[:4000], return_tensors="pt", truncation=True)
            with _no_grad():
                output = model(**encoded)
                vector = output.last_hidden_state.mean(dim=1).squeeze(0)
                vectors.append([round(float(v), 6) for v in vector[:8]])
        return {
            "result": {"mode": "embed", "model": model_name, "vectors_preview": vectors},
            "summary": f"Embedded {len(texts)} text(s) with {model_name} (mean-pooled, preview shown).",
            "libraries_used": ["transformers"],
        }

    model_name = str(params.get("model") or "distilgpt2")
    prompt = str(params.get("prompt") or payload.get("data") or params.get("text") or "")
    if not prompt:
        raise ValueError("Provide `params.prompt` (or `data`) for text generation.")
    from transformers import pipeline

    generator = pipeline("text-generation", model=model_name)
    max_new = int(params.get("max_new_tokens") or 60)
    outputs = generator(prompt[:2000], max_new_tokens=max_new, do_sample=True, temperature=0.8)
    generated = outputs[0]["generated_text"] if outputs else prompt
    return {
        "result": {
            "mode": "generate",
            "model": model_name,
            "generated_text": generated,
            "max_new_tokens": max_new,
        },
        "summary": f"Local model {model_name} generated {len(generated)} chars.",
        "libraries_used": ["transformers"],
    }


def run_llm_gateway(payload: dict[str, Any]) -> dict[str, Any]:
    """Unified LLM completion via LiteLLM (any provider, routing + fallbacks)."""
    params = payload.get("params") or {}
    settings = payload.get("_settings")

    litellm = load("litellm")

    model = str(params.get("model") or "")
    if not model:
        if settings and getattr(settings, "relai_model", ""):
            model = settings.relai_model
        else:
            model = "gpt-4o-mini"
    if model.startswith("gemini") and "/" not in model:
        model = f"gemini/{model}"
    elif model.startswith("claude") and "/" not in model:
        model = f"anthropic/{model}"
    elif model.startswith("llama") and "/" not in model:
        model = f"ollama/{model}"

    prompt = str(params.get("prompt") or payload.get("data") or "")
    system = str(params.get("system") or "")
    if not prompt:
        raise ValueError("Provide `params.prompt` (or `data`) to call the LLM.")

    kwargs: dict[str, Any] = {}
    if settings:
        if getattr(settings, "openai_api_key", ""):
            kwargs.setdefault("api_key", settings.openai_api_key)
        if model.startswith("gemini/") and getattr(settings, "gemini_api_key", ""):
            kwargs["api_key"] = settings.gemini_api_key

    try:
        response = litellm.completion(
            model=model,
            messages=[
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            temperature=float(params.get("temperature") or 0.5),
            max_tokens=int(params.get("max_tokens") or 500),
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - surface provider errors cleanly
        return {
            "result": {"mode": "llm", "model": model, "error": str(exc)[:300]},
            "summary": f"LiteLLM call to {model} failed: {str(exc)[:160]}",
            "libraries_used": ["litellm"],
        }

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = response.get("usage") or {}
    return {
        "result": {
            "mode": "llm",
            "model": response.get("model", model),
            "provider": response.get("model", model).split("/")[0],
            "content": content,
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
        },
        "summary": f"LiteLLM → {model} replied ({len(content)} chars).",
        "libraries_used": ["litellm"],
    }


def run_rag(payload: dict[str, Any]) -> dict[str, Any]:
    """Index provided documents and answer from them (LlamaIndex)."""
    params = payload.get("params") or {}
    documents = _as_documents(payload.get("data"), params)
    if not documents:
        raise ValueError(
            "No documents found. Send `data` as a list of text strings (or "
            "params.documents)."
        )
    query = str(params.get("query") or "")
    if not query:
        task = str(payload.get("task") or "")
        query = task

    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.node_parser import SimpleNodeParser
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(
            f"LlamaIndex import failed: {exc}. Install with "
            "`pip install -r requirements.companion-ai.txt`."
        ) from exc

    embed_model = str(params.get("embed_model") or "default")
    if embed_model == "huggingface":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: F401

            Settings.embed_model = HuggingFaceEmbedding(
                model_name=str(params.get("embed_model_name") or "sentence-transformers/all-MiniLM-L6-v2")
            )
        except Exception:  # noqa: BLE001 - optional embedding package
            raise CapabilityUnavailable(
                "Hugging Face embeddings need `llama-index-embeddings-huggingface` "
                "(pip install llama-index-embeddings-huggingface)."
            ) from None

    try:
        parser = SimpleNodeParser.from_defaults(chunk_size=800, chunk_overlap=60)
        nodes = parser.get_nodes_from_documents([Document(text=d) for d in documents])
        index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
        retriever = index.as_retriever(similarity_top_k=int(params.get("top_k") or 3))
        hits = retriever.retrieve(query)
        context = "\n\n".join(f"[score {h.score:.3f}] {h.node.text[:400]}" for h in hits)
        return {
            "result": {
                "query": query,
                "documents": len(documents),
                "chunks_retrieved": len(hits),
                "retrieved": [{"score": round(float(h.score), 4), "text": h.node.text[:400]} for h in hits],
            },
            "summary": (
                f"LlamaIndex indexed {len(documents)} document(s) and retrieved "
                f"{len(hits)} chunk(s) for the query."
            ),
            "libraries_used": ["llama_index"],
        }
    except Exception as exc:  # noqa: BLE001 - LLM/embedding dependency issues
        return {
            "result": {
                "query": query,
                "error": str(exc)[:300],
                "hint": "LlamaIndex needs an embedding model. Set params.embed_model='huggingface' "
                "(installs a local embedder) or configure an OpenAI key for the default.",
            },
            "summary": f"LlamaIndex indexing/retrieval failed: {str(exc)[:160]}",
            "libraries_used": ["llama_index"],
        }


def run_tensorflow(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a small Keras model (TensorFlow backend) on tabular data."""
    params = payload.get("params") or {}

    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"TensorFlow import failed: {exc}") from exc

    import numpy as np

    data = payload.get("data")
    if data is None:
        raise ValueError("tensorflow 'fit' needs tabular `data` with a `params.target` column.")
    X, y, target, cols, _ = _tabular_for_ai(data, params, np)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) < 8:
        raise ValueError("Need at least 8 rows to train a small network.")

    split = int(len(X) * 0.8)
    model = keras.Sequential(
        [
            keras.Input(shape=(X.shape[1],)),
            keras.layers.Dense(int(params.get("hidden") or 32), activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics=["mae"])
    history = model.fit(
        X[:split], y[:split], epochs=int(params.get("epochs") or 40), verbose=0, validation_data=(X[split:], y[split:])
    )
    final_loss = float(history.history["loss"][-1])
    val_loss = float(history.history["val_loss"][-1])
    return {
        "result": {
            "mode": "fit",
            "engine": "tensorflow",
            "target": target,
            "rows": len(X),
            "final_train_mse": round(final_loss, 6),
            "val_mse": round(val_loss, 6),
        },
        "summary": f"Keras/TensorFlow MLP trained on {len(X)} rows — final train MSE {final_loss:.5g}, val MSE {val_loss:.5g}.",
        "libraries_used": ["tensorflow", "keras"],
    }


def run_jax(payload: dict[str, Any]) -> dict[str, Any]:
    """JAX automatic differentiation / JIT computation."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "autodiff").lower()

    import jax
    import jax.numpy as jnp

    if mode == "autodiff":
        a = float(params.get("a") or 1.0)
        b = float(params.get("b") or 0.0)
        c = float(params.get("c") or 0.0)
        x = float(params.get("x") or 3.0)

        def f(v):
            return a * v * v + b * v + c

        value = float(f(jnp.array(x)))
        gradient = float(jax.grad(f)(jnp.array(x)))
        second = float(jax.grad(jax.grad(f))(jnp.array(x)))
        return {
            "result": {"mode": "autodiff", "f(x)=ax²+bx+c": value, "f'(x)": gradient, "f''(x)": second},
            "summary": f"JAX autodiff: at x={x:g}, f={value:.4g}, f'={gradient:.4g}, f''={second:.4g}.",
            "libraries_used": ["jax"],
        }

    if mode == "jit":
        import numpy as np

        n = int(params.get("n") or 1000)
        key = jax.random.PRNGKey(int(params.get("seed") or 0))

        @jax.jit
        def square_sum(vec):
            return jnp.sum(vec * vec)

        vec = jax.random.normal(key, (n,))
        result = float(square_sum(vec))
        return {
            "result": {"mode": "jit", "n": n, "sum_of_squares": result},
            "summary": f"JAX JIT-compiled sum of squares over {n} values = {result:.4g}.",
            "libraries_used": ["jax"],
        }

    raise ValueError("jax mode must be 'autodiff' or 'jit'.")


# ── shared helpers ───────────────────────────────────────────────────────

def _tabular_for_ai(data, params: dict[str, Any], np) -> tuple[list, list, str, list, list]:
    from .common import as_rows

    columns, rows = as_rows(data)
    if not rows:
        raise ValueError("No tabular data found.")
    if columns is None:
        width = max(len(r) for r in rows)
        columns = [f"col{i}" for i in range(width)]
    target = str(params.get("target") or columns[-1])
    if target not in columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {', '.join(columns)}")
    idx = columns.index(target)
    X = [[r[i] for i in range(len(columns)) if i != idx] for r in rows]
    y = [r[idx] for r in rows]
    # Numeric coercion with NaN-safe conversion; non-numeric stays as float placeholder.
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    X = [[_num(v) for v in row] for row in X]
    y = [_num(v) for v in y]
    return X, y, target, [c for i, c in enumerate(columns) if i != idx], columns


def _as_documents(data, params: dict[str, Any]) -> list[str]:
    docs = params.get("documents")
    if isinstance(docs, list) and all(isinstance(d, str) for d in docs):
        return docs
    if isinstance(data, list) and all(isinstance(d, str) for d in data):
        return data
    if isinstance(data, str) and len(data) > 40:
        return [data]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [" ".join(str(v) for v in d.values() if v is not None) for d in data]
    return []


def _no_grad():
    """torch.no_grad() — works whether or not TF is imported."""
    try:
        import torch

        return torch.no_grad()
    except Exception:  # noqa: BLE001
        import contextlib

        return contextlib.nullcontext()
