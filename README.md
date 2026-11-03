# Challenge Flask App

A minimal Flask application with Docker Compose support.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Run with Docker Compose

```bash
docker compose up --build
```

The app will be available at http://localhost:5000
