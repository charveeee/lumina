# Lumina

Lumina adapts difficult text in real time based on how long a reader dwells on it. The longer you stay on a paragraph, the more aggressively it simplifies — shorter sentences, simpler vocabulary, and shorter chunks. It also uses Claude-Mem's observation worker to track recurring reading-friction patterns across a browser session, so a pattern that keeps showing up gets simplified more aggressively right away.

Adaptation is fully rule-based (sentence splitting + vocabulary simplification) — no LLM API call and no API key required, so the demo has no external dependency that can fail mid-presentation.

## Run locally

```
pip install -r requirements.txt
python3 app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000/) — don't open the HTML file directly, since it needs the API endpoints served by `app.py`.

## How adaptation works

- **Dwell-based intensity:** hovering 2-4s gives light simplification, 4-6s is moderate, 6s+ is aggressive (short, fragmented chunks). Staying on a paragraph keeps re-adapting it roughly every 3 seconds.
- **Struggle-pattern tagging:** each adaptation analyzes the original text and tags why it was likely difficult (e.g. `long_sentence`, `complex_vocabulary`, `dense_clause`, `passive_voice`).
- **Personalization:** if Claude-Mem reports the same pattern has recurred 3+ times in a session, Lumina jumps straight to aggressive simplification for that paragraph, regardless of dwell time.
- **Reset session:** the "Reset session" button clears all adapted paragraphs so the same demo can be re-run without restarting the server.

## Claude-Mem observer setup

Lumina uses Claude-Mem's local worker API — it does not maintain its own memory database. Install and start the official worker once:

```
npx claude-mem install --ide codex-cli --provider claude
npx claude-mem start
```

Sign in to Claude when prompted so the worker can compress queued observations. The app sends a structured `lumina_adapt` event after every adaptation and reads the worker's stored observations before each new adaptation. Set `CLAUDE_MEM_URL` only if the worker runs somewhere other than `http://127.0.0.1:37701`.

Note: the memory-badge/personalization path degrades gracefully if the worker isn't reachable — Lumina keeps adapting text normally, it just won't show a live pattern count or trigger personalization.
