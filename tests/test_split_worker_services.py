from fastapi.testclient import TestClient

from src.recommendation import sbert_service
from src.recommendation.video import blip_service, whisper_service


class _FakeHashtagRecommender:
    corpus_captions = ["a", "b"]

    def recommend(self, caption, k=25, top_n=10, exclude_tags=None, diversity_weight=0.5):
        return [
            {
                "hashtag": "#ramen",
                "frequency": 2,
                "avg_engagement": 1.0,
                "avg_similarity": 0.8,
                "score": 0.9,
            }
        ][:top_n]


def test_worker_health_endpoints_do_not_require_model_load():
    assert TestClient(whisper_service.app).get("/v1/health").json()["service"] == "whisper"
    assert TestClient(blip_service.app).get("/v1/health").json()["service"] == "blip"


def test_sbert_worker_suggest_endpoint_uses_loaded_recommender(monkeypatch):
    monkeypatch.setattr(sbert_service, "_recommender", _FakeHashtagRecommender())

    response = TestClient(sbert_service.app).post(
        "/v1/hashtags/suggest",
        json={"caption": "ramen night", "top_n": 1, "diversity_weight": 0.25},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["hashtags"][0]["hashtag"] == "#ramen"
    assert payload["corpus_size"] == 2
