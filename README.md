# Lumina

Lumina adapts difficult text in response to dwell time, then uses Claude-Mem's
real observation worker to remember recurring reading-friction patterns during a
browser session.

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_key"
uvicorn public.app:app --reload
```

Open http://127.0.0.1:8000 rather than the `file://` HTML file.

## Claude-Mem observer setup

Lumina uses Claude-Mem's local worker API — it does not maintain its own memory
database. Install and start the official worker once:

```bash
npx claude-mem install --ide codex-cli --provider claude
npx claude-mem start
```

Sign in to Claude when prompted so the worker can compress queued observations.
The app sends a structured `lumina_adapt` event after every adaptation and reads
the worker's stored observations before each new adaptation. After three matching
patterns in the same browser session, Lumina adds proactive simplification for
that pattern. Set `CLAUDE_MEM_URL` only if the worker runs somewhere other than
`http://127.0.0.1:37701`.
