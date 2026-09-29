# BLRDetector – Render deployment

This folder is the deployment-ready BLRDetector build.

## Runtime
- Python 3.11
- Dash + Plotly
- OpenCV headless
- Gunicorn
- No database or persistent disk required

## Render settings
If deploying manually instead of using `render.yaml`:

- Runtime: `Python 3`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:server --workers 1 --threads 2 --timeout 180 --max-requests 100 --max-requests-jitter 20 --bind 0.0.0.0:$PORT`
- Health check path: `/`
- Plan for first test: Free

The project pins Python 3.11 using `.python-version`.

## Why one Gunicorn worker?
Image analysis creates NumPy/OpenCV arrays. Multiple worker processes duplicate memory. One worker with two threads is a safer starting point for Render's 512 MB Free instance.

## Local test
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:8050`.

## Deploy
1. Put this folder in a GitHub repository.
2. In Render choose **New > Web Service** and connect the repository, or create a Blueprint from `render.yaml`.
3. Deploy.
4. Test one small batch first, then a ~100–120 object batch.
5. Watch Render logs and memory while opening Review Objects and generating the ZIP.

## Current processing build
- V0.54 optimized Analyze/Review architecture
- V0.56 manual-object and scanner-strip cleanup
