# Deploying the app to Google Cloud Run

These are the recommended step-by-step commands to build the container image and deploy to Cloud Run from your local machine (Windows PowerShell). They assume you have the `gcloud` CLI installed and authenticated, and you have selected the right GCP project.

1) Set variables (adjust PROJECT, REGION, SERVICE_NAME as desired):

```powershell
$env:PROJECT = 'YOUR_GCP_PROJECT_ID'
$env:REGION = 'us-central1'
$env:SERVICE_NAME = 'brain-tumor-detector'
```

2) (Optional) Enable required APIs:

```powershell
gcloud services enable run.googleapis.com containerregistry.googleapis.com
```

3) Build the container and push to Artifact Registry / Container Registry

Using Container Registry (docker.io style):

```powershell
gcloud builds submit --tag gcr.io/$env:PROJECT/$env:SERVICE_NAME
```

4) Deploy to Cloud Run

Pick a threshold for the MRI filter (example: 0.45). You can later update this with `gcloud run services update`.

```powershell
gcloud run deploy $env:SERVICE_NAME \
  --image gcr.io/$env:PROJECT/$env:SERVICE_NAME \
  --region $env:REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars MRI_SCORE_THRESHOLD=0.45
```

Notes and tips
- If you prefer Artifact Registry, push there and use the corresponding image location (e.g., `us-docker.pkg.dev/<PROJECT>/<REPO>/<IMAGE>`).
- For production, consider increasing `--concurrency` and `--memory` and set `--min-instances` to warm the model.
- If you want to avoid shipping the large full TensorFlow Keras model into the container, remove/omit `models/final_model.h5` and rely on the TFLite runtime + the MRI filter. The app already lazy-loads TensorFlow only when needed.
- To update environment variables (e.g., adjust `MRI_SCORE_THRESHOLD`), run:

```powershell
gcloud run services update $env:SERVICE_NAME --region $env:REGION --set-env-vars MRI_SCORE_THRESHOLD=0.3
```

Troubleshooting
- If the image fails to start due to missing system libs for `tflite-runtime` or `opencv-python-headless`, add the required apt packages to the Dockerfile.
- If conversion to TFLite is needed on build, make sure the build environment includes TensorFlow (heavy) or pre-convert locally and add the `.tflite` into `models/` before deploying.
