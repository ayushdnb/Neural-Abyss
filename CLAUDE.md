# Repository working rules

## Mission
Treat this repository as a serious public technical codebase. Maintain formal, strict, compact, and technically rigorous standards across source and documentation.

## Global editing policy
- Preserve behavior unless a change is required for correctness, reliability, consistency, or maintainability.
- Do not speculate about code that has not been opened.
- Investigate relevant files before making claims or edits.
- Do not introduce decorative abstractions or speculative architecture changes.
- Prefer conservative, verifiable improvements.

## Source-code standards
- Maintain consistent naming across modules, classes, functions, constants, parameters, and state variables.
- Prefer clear names over explanatory comments.
- Minimize comments.
- Remove informal, redundant, historical, patch-style, and obvious comments.
- Keep comments only for non-obvious invariants, subtle reasoning, mathematical intent, or safety-critical behavior.
- Use formal technical language only.
- Reduce code noise and duplication where safe.
- Keep control flow compact and readable.
- Maintain consistent typing, import style, constant placement, and exception handling.

## Documentation standards
- Verify Markdown claims against the implementation before preserving them.
- Remove hype, bragging, filler, and unverifiable claims.
- Keep tone formal, precise, and technically grounded.
- Keep terminology aligned with the code.

## Working method
- For large tasks, work in small audited batches.
- For each batch: inspect, identify issues, edit conservatively, validate, then summarize.
- Ignore generated or runtime folders unless they affect source quality directly.

## Ignore by default
- results
- __pycache__
- .git
- .venv
- venv
- build
- dist