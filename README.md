# Sentiment Analyzer

Collects feedback through a form and uses
[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
to automatically classify each submission as **POSITIVE** or **NEGATIVE**.
All feedback is stored in a local SQLite database and visualised on an interactive dashboard.

```
sentiment-analyzer/
├── backend/
│   ├── app.py              # FastAPI — /health, /analyze, /feedback, /feedback/stats
│   ├── database.py         # SQLite schema and query helpers
│   ├── requirements.txt
│   ├── .env                # Local config (gitignored)
│   └── .env.example        # Config template (committed)
├── frontend/
│   ├── app.py              # Streamlit entry point + home page
│   ├── pages/
│   │   ├── 1_Feedback.py   # Feedback form with live sentiment result
│   │   └── 2_Dashboard.py  # KPI strip, charts, filterable feedback table
│   ├── requirements.txt
│   ├── .env                # Local config (gitignored)
│   └── .env.example        # Config template (committed)
├── .gitignore
└── README.md
```

---

## Architecture

```
Streamlit frontend (:8501)  →  HTTP  →  FastAPI backend (:8000)  →  SQLite
```

The frontend makes no direct database or model calls — all logic lives in the backend.

---

## Backend (FastAPI)

### Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
source venv/bin/activate         # Mac/Linux

pip install -r backend/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **First-run download** — On first startup the model weights (~250 MB) are
> downloaded to `~/.cache/huggingface/hub/`. Subsequent runs load from cache.

### Configuration

```bash
cp backend/.env.example backend/.env
```

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed CORS origins |
| `DB_PATH` | `feedback.db` | Path to the SQLite database file |

### Running

```bash
uvicorn backend.app:app --reload
```

API at `http://127.0.0.1:8000` · Docs at `http://127.0.0.1:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | API and model status |
| `POST` | `/analyze` | Classify text sentiment (no persistence) |
| `POST` | `/feedback` | Analyze + save a full feedback submission |
| `GET` | `/feedback` | Return all feedback rows (newest first) |
| `GET` | `/feedback/stats` | Return aggregate stats for the dashboard |

---

## Frontend (Streamlit)

### Setup

```bash
pip install -r frontend/requirements.txt
```

### Configuration

```bash
cp frontend/.env.example frontend/.env
```

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend |

### Running

Make sure the backend is running first, then:

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`. Three pages in the sidebar:
- **Home** — model info and status
- **Feedback** — submit feedback (name optional, email + category + rating + text required)
- **Dashboard** — KPI strip, sentiment donut, category bar chart, rating distribution, confidence histogram, submissions-over-time line chart, filterable full table

---

## Stack

- [FastAPI](https://fastapi.tiangolo.com/) — REST API backend
- [Streamlit](https://streamlit.io/) — multi-page web frontend
- [Transformers](https://huggingface.co/docs/transformers) — DistilBERT sentiment pipeline
- [PyTorch](https://pytorch.org/) (CPU) — model inference
- [SQLite](https://www.sqlite.org/) — local feedback storage
- [Plotly](https://plotly.com/python/) — interactive charts
- [Pandas](https://pandas.pydata.org/) — data manipulation
