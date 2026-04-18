# BgPrunR — Claude Guidelines

## Honesty over cop-outs

When declining a task, skipping a refactor, or marking something "not worth it," **state the real reason**. Never invent risk or complexity narratives to cover "I didn't do the homework" or "I was pacing myself."

Acceptable:
- "I haven't verified X, would need to read Y first."
- "This requires plumbing through 3 call sites for a net-negative LOC change — not a win."
- "Out of scope for the current task — separate commit."

Not acceptable:
- "Not worth the risk" (generic hand-wave).
- "Medium refactor" (without naming the actual cost).
- Inventing trade-offs that don't exist to justify skipping work.

If the honest answer is "I was lazy," say that. The user would rather hear it and redirect than unpack a fake justification.
