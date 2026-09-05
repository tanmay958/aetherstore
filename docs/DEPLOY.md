# Deploying

The query service runs on Cloud Run.
Everything else either stays on a laptop or is not a service at all, and the reasons are worth stating because they are the reasons the hosted piece is shaped the way it is.

```
GitHub Actions          Cloudflare R2                Cloud Run
(training, batch)  -->  models/model.npz       -->   /api/predict
                        models/model.pkl             /api/search
                        idx100k/segments/*.seg  -->
                        idx100k/manifest.json

your machine
docker compose: Redpanda + MinIO + 3 predictor replicas
```

**Live:** https://aether-441173057461.us-central1.run.app

## What is not hosted, and why

**The Redpanda broker and the streaming predictor replicas.**
Both are Kafka consumers, and a consumer that scales to zero does not idle cheaply, it falls behind.
Nothing free will keep a process running continuously, so these stay local.
That is the demo to record rather than to link, and it is the one that matches the distributed-inference argument.

**The raw CSV.**
14.7 GB of REES46 is input to a build step.
It is never deployed and never read at query time, which is what lets the query service hold nothing and therefore scale to zero.

## Cost

Everything below sits inside a permanent free tier.

| | Free allowance | This project |
|---|---|---|
| Cloud Run requests | 2M / month | a handful |
| Cloud Run compute | 180k vCPU-seconds / month | zero while idle |
| Artifact Registry | 0.5 GB | 147 MB per image |
| Secret Manager | 6 active versions | 3 |
| Cloudflare R2 storage | 10 GB | ~9 MB |
| Cloudflare R2 reads | 10M class B / month | ~60 per cold query |
| Cloudflare R2 egress | unmetered | - |

**The one setting that would break this is `--min-instances`.**
Anything above zero keeps a container alive and bills continuously.
It is tempting, because it removes the cold start.
Leave it at zero.

Keep the number of image versions small too: three revisions of a 147 MB image is 441 MB, which is most of the Artifact Registry free tier.

## The deployment, start to finish

Assumes `gcloud` is authenticated and the project has billing enabled, which Cloud Run requires even inside the free tier.

```bash
PROJECT=aetherstore-507714
REGION=us-central1

gcloud services enable \
  run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com --project $PROJECT
```

### Credentials

R2 keys go in Secret Manager rather than in `--set-env-vars`, where they would sit in the service config and in shell history.

```bash
for name in r2-account-id r2-access-key-id r2-secret-access-key; do
  gcloud secrets create $name --replication-policy=automatic --project $PROJECT
done
# then add a version for each, reading from .env, and check the byte counts:
gcloud secrets versions access latest --secret=r2-account-id --project $PROJECT | wc -c   # 32
```

That last check is not ceremony.
An earlier attempt at this used shell indirect expansion that silently expanded to nothing, and `gcloud` accepted three empty secrets and reported success.

### A service account with only what it needs

```bash
gcloud iam service-accounts create aether-service --project $PROJECT

for name in r2-account-id r2-access-key-id r2-secret-access-key; do
  gcloud secrets add-iam-policy-binding $name \
    --member="serviceAccount:aether-service@$PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" --project $PROJECT
done
```

IAM is eventually consistent, and a binding made immediately after creating the service account can fail.
Verify it landed before deploying, rather than discovering it in a failed revision:

```bash
gcloud secrets get-iam-policy r2-account-id --project $PROJECT
```

### Build and deploy

Built on Cloud Build rather than locally, because an Apple Silicon laptop produces an `arm64` image and Cloud Run needs `amd64`.

```bash
gcloud artifacts repositories create aether \
  --repository-format=docker --location=$REGION --project $PROJECT

gcloud builds submit \
  --tag $REGION-docker.pkg.dev/$PROJECT/aether/service:v1 \
  --project $PROJECT --region $REGION

gcloud run deploy aether \
  --image=$REGION-docker.pkg.dev/$PROJECT/aether/service:v1 \
  --region=$REGION --project=$PROJECT \
  --service-account=aether-service@$PROJECT.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --min-instances=0 --max-instances=3 \
  --memory=512Mi --cpu=1 --concurrency=40 --timeout=60 \
  --set-env-vars=AETHER_INDEX=r2://aether/idx100k,AETHER_MODEL=r2://aether/models/model.npz \
  --set-secrets=R2_ACCOUNT_ID=r2-account-id:latest,R2_ACCESS_KEY_ID=r2-access-key-id:latest,R2_SECRET_ACCESS_KEY=r2-secret-access-key:latest
```

`512Mi` is the smallest tier and is ample: the container measured 147 MB under load, because it holds no data.

## Measured, on the deployed service

```
cold start, first request after idle     514 ms
warm query, terms already cached         297 ms   10 requests
warm query, new terms                    618 ms   21 requests
warm query, many matching segments     1,266 ms   40 requests
```

A warm instance still spends requests, and it should.
Only a segment's footer and hotcache are cached, and those are what make it *openable*; a query for a term nobody has asked for still has to read that term's dictionary block and postings.
What caching removes is the cost of opening, which is paid once, not the cost of reading, which is paid per distinct term.

The same query measured 3,453 ms on the very first request against a cold instance, which is the compaction argument restated: that time is ten segments being opened, and it is why the index is compacted rather than left as one segment per flush.

## Checking it

```bash
U=https://aether-441173057461.us-central1.run.app
curl -s $U/health
curl -s "$U/api/search?q=samsung+smartphone&k=3&explain=true"
curl -s -X POST $U/api/predict -H 'content-type: application/json' \
  -d '{"events":[{"ts":1570000000,"event_type":"add_to_cart","product_id":"p","price":1206.45}]}'
```

`$U/docs` is the generated API browser, which is a usable demo surface until the dashboard exists.

## Updating it

```bash
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/aether/service:v2 --project $PROJECT --region $REGION
gcloud run deploy aether --image=$REGION-docker.pkg.dev/$PROJECT/aether/service:v2 --region=$REGION --project=$PROJECT
```

Publishing new data needs no deployment at all.
The service reads the manifest at start-up, so a new segment becomes visible to the next cold instance, and `Coordinator.refresh` picks it up in a warm one.
