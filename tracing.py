"""
LangFuse Tracing Helper — shared by multi_agent.py and agent_sdk_demo.py.

Usage:
    from tracing import get_langfuse, flush, NOOP

    langfuse = get_langfuse()
    trace = langfuse.trace(name="...", input=...) if langfuse else NOOP

    span = trace.span(name="...", input=...)      # always safe
    gen = span.generation(name="...", model=...)   # always safe
    gen.end(output=...)
    span.end(output=...)
    trace.end(output=...)

    flush()  # before process exit

NOOP is a sentinel whose .span(), .generation(), .end(), .update() all return
itself (another NOOP), so tracing code reads naturally without if-checks.

Env vars required:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST  (optional, defaults to https://cloud.langfuse.com)
"""

import os


# ── NOOP sentinel ─────────────────────────────────────────────
class _NoOp:
    """No-op trace/span/generation — all method calls return self."""
    def __getattr__(self, name):
        return lambda *a, **kw: self
    def __call__(self, *a, **kw):
        return self
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __bool__(self): return False  # truthy check


NOOP = _NoOp()


# ── LangFuse lifecycle ────────────────────────────────────────
_langfuse = None
_checked = False


def is_available() -> bool:
    """True when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set."""
    return all(
        os.environ.get(k)
        for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    )


def get_langfuse():
    """Return the global Langfuse client, or None if not configured."""
    global _langfuse, _checked
    if not _checked:
        if is_available():
            from langfuse import Langfuse
            host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            _langfuse = Langfuse(host=host)
        _checked = True
    return _langfuse


def flush():
    """Block until all pending events are sent to LangFuse."""
    if _langfuse:
        _langfuse.flush()
