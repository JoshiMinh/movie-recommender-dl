from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.inference import RecommenderService


class RecommendRequest(BaseModel):
    user_sequence: List[int] = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=50)


class RecommendResponse(BaseModel):
    recommendations: List[int]


@lru_cache(maxsize=1)
def get_service() -> RecommenderService:
    return RecommenderService.from_default()


app = FastAPI(title="Movie Recommender DL", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        service = get_service()
        recs = service.recommend(payload.user_sequence, top_k=payload.top_k)
        return RecommendResponse(recommendations=recs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
