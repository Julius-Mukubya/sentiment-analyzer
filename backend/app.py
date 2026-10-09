"""
backend/app.py — FastAPI sentiment analyzer API.

Endpoints
---------
GET  /health              API + model status
POST /analyze             Classify text sentiment (no persistence)
POST /feedback            Analyze + persist a full feedback submission
GET  /feedback            Return all feedback rows
GET  /feedback/stats      Return aggregate stats for the dashboard
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator
from transformers import pipeline

import sys
from pathlib import Path

# Ensure the backend/ directory is on the path regardless of where
# uvicorn is launched from.
sys.path.insert(0, str(Path(__file__).parent))

from database import get_all_feedback, get_summary_stats, init_db, insert_feedback

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
CATEGORIES = ["Product", "Service", "General"]

# ── Model loading ──────────────────────────────────────────────────────────────

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initialising database…")
    init_db()
    print("Loading sentiment model…")
    ml_models["classifier"] = pipeline(
        "text-classification",
        model=MODEL_NAME,
    )
    print("Model ready.")
    yield
    ml_models.clear()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Analyzer API",
    description="Classify feedback sentiment using DistilBERT (SST-2).",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8501")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, description="Text to classify")


class SentimentResult(BaseModel):
    sentiment: str   # 'POSITIVE' | 'NEGATIVE'
    confidence: float


class FeedbackRequest(BaseModel):
    name:     Optional[str] = Field(None, description="Submitter name (optional)")
    email:    EmailStr      = Field(..., description="Submitter email")
    category: str           = Field(..., description="'Product' | 'Service' | 'General'")
    rating:   int           = Field(..., ge=1, le=5, description="Star rating 1-5")
    feedback: str           = Field(..., min_length=10, max_length=5000, description="Feedback text")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        return v


class FeedbackResponse(BaseModel):
    id:         int
    name:       Optional[str]
    email:      str
    category:   str
    rating:     int
    feedback:   str
    sentiment:  str
    confidence: float
    created_at: str


class StatsResponse(BaseModel):
    total:       int
    positive:    int
    negative:    int
    avg_rating:  float
    by_category: dict
    by_sentiment: dict


# ── Helper ─────────────────────────────────────────────────────────────────────

def _classify(text: str) -> tuple[str, float]:
    result = ml_models["classifier"](text, truncation=True, max_length=512)[0]
    return result["label"], round(result["score"], 4)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health():
    """Check that the API and model are ready."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze", response_model=SentimentResult, tags=["Sentiment"])
def analyze(request: AnalyzeRequest):
    """
    Classify the sentiment of a piece of text.
    Returns the label and confidence score — does NOT persist anything.
    """
    sentiment, confidence = _classify(request.text)
    return SentimentResult(sentiment=sentiment, confidence=confidence)


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(request: FeedbackRequest):
    """
    Analyze sentiment and persist the full feedback submission.
    Returns the saved row including the generated id and created_at timestamp.
    """
    sentiment, confidence = _classify(request.feedback)
    row_id = insert_feedback(
        email=request.email,
        category=request.category,
        rating=request.rating,
        feedback=request.feedback,
        sentiment=sentiment,
        confidence=confidence,
        name=request.name,
    )
    all_rows = get_all_feedback()
    saved = next(r for r in all_rows if r["id"] == row_id)
    return FeedbackResponse(**saved)


@app.get("/feedback", response_model=List[FeedbackResponse], tags=["Feedback"])
def list_feedback():
    """Return all feedback rows, newest first."""
    return [FeedbackResponse(**row) for row in get_all_feedback()]


@app.get("/feedback/stats", response_model=StatsResponse, tags=["Feedback"])
def feedback_stats():
    """Return aggregate stats for the dashboard."""
    return StatsResponse(**get_summary_stats())
