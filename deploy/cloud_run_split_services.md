# Cloud Run Split Services

The original `Dockerfile` still supports the unified local/demo deployment. For
production cost control, deploy the heavy model workers independently so Cloud
Run can scale each model family on its own:

| Service | Dockerfile | Endpoint | Scaling intent |
|---|---|---|---|
| `tiktok-whisper` | `Dockerfile.whisper` | `/v1/whisper/transcribe` | Scale only when audio transcription jobs arrive. |
| `tiktok-blip` | `Dockerfile.blip` | `/v1/blip/caption` | Scale CPU/GPU captioning separately from recommender traffic. |
| `tiktok-sbert` | `Dockerfile.sbert` | `/v1/hashtags/suggest` | Keep SBERT/FAISS memory isolated from video analysis workers. |
| `tiktok-recommender` | `Dockerfile.recommender` | `/v1/recommendations` | Core retrieval/ranking API without video model warmup. |

## Build And Deploy

Replace `PROJECT_ID` and region as needed.

```bash
gcloud auth configure-docker

docker build -f Dockerfile.whisper -t gcr.io/PROJECT_ID/tiktok-whisper .
docker build -f Dockerfile.blip -t gcr.io/PROJECT_ID/tiktok-blip .
docker build -f Dockerfile.sbert -t gcr.io/PROJECT_ID/tiktok-sbert .
docker build -f Dockerfile.recommender -t gcr.io/PROJECT_ID/tiktok-recommender .

docker push gcr.io/PROJECT_ID/tiktok-whisper
docker push gcr.io/PROJECT_ID/tiktok-blip
docker push gcr.io/PROJECT_ID/tiktok-sbert
docker push gcr.io/PROJECT_ID/tiktok-recommender

gcloud run deploy tiktok-whisper \
  --image gcr.io/PROJECT_ID/tiktok-whisper \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 5

gcloud run deploy tiktok-blip \
  --image gcr.io/PROJECT_ID/tiktok-blip \
  --region us-central1 \
  --memory 6Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3

gcloud run deploy tiktok-sbert \
  --image gcr.io/PROJECT_ID/tiktok-sbert \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10

gcloud run deploy tiktok-recommender \
  --image gcr.io/PROJECT_ID/tiktok-recommender \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 20
```

## Integration Notes

- Keep the recommender API free of Whisper/BLIP startup preloading in production.
- Route video uploads to `tiktok-whisper` and `tiktok-blip` asynchronously from the gateway or job queue.
- Route hashtag suggestion traffic to `tiktok-sbert` instead of the unified `/v1/hashtags/suggest` endpoint when deployed.
- Apply `deploy/cloud_run_alerts.yaml` with the split service names so latency and error alerts point at the responsible worker.
