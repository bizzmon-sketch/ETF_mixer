# Repository Guidelines

## Project Overview
This repo currently hosts a **static, single-file MVP UI** for portfolio-analysis visualization prototypes. The `index2.html` page is a **DEMO** (sample data) implementation aligned to the PRD and is **not investment advice**.

## Project Structure & Module Organization
- `index.html`: ETF Mixer prototype (single-file UI).
- `index2.html`: Pension ETF optimization demo UI (single-file, Plotly + Bootstrap via CDN).
- `PRD.md`: Product requirements.
- `연금 ETF 포트폴리오 최적화 웹서비스 요구사항 명세서.pdf`: Source PRD document.

When backend work starts, prefer introducing:
- `backend/` (Flask, data ingestion, optimization)
- `frontend/` (if/when separated from single-file)
- `data/` (seed CSV, local SQLite)
- `docs/` (architecture, ops notes)

## Build, Test, and Development Commands
No build tools are required today.
- Run locally (recommended): `python -m http.server 8000` then open `http://localhost:8000/index2.html`.
- Quick open (Windows): `start .\index2.html` (may be blocked by some browser security features).

## Coding Style & Naming Conventions
- Indentation: **2 spaces** for HTML/CSS/JS.
- JS structure: keep logic in small functions; centralize constants in a single `CONFIG` object.
- Naming: `camelCase` for JS functions/variables, `SCREAMING_SNAKE_CASE` only for constants if needed.
- Security: always escape untrusted strings before injecting into HTML (`escapeHtml`).

## Testing Guidelines
No automated tests exist yet. When adding backend code, place tests in `backend/tests/` and name them `test_*.py`.

## Commit & Pull Request Guidelines
Git is not initialized yet, so conventions are not established. Use:
- Commit messages: `feat: ...`, `fix: ...`, `docs: ...` (Conventional Commits).
- PRs: include a short description, screenshots/GIF for UI changes, and note any UX/legal copy changes.

## Safety & Product Constraints
- Do not add language implying guarantees, “best”, or personalized recommendations.
- Clearly label placeholder calculations as **DEMO** until real data + backend are wired.
