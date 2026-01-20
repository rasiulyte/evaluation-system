# Easy Deployment Options

This project is designed to deploy quickly with minimal setup. Choose one of the options below.

## Option A: Streamlit Community Cloud (Easiest)

1. Make this repo public on GitHub.
2. Go to https://streamlit.io/cloud and create an app.
3. Select your repo and set the app file to `src/dashboard.py`.
4. In "Secrets", add `OPENAI_API_KEY`.
5. Deploy. Streamlit handles build and hosting automatically.

Notes:
- Requirements are installed from `requirements.txt`.
- No extra files or Docker needed.

## Option B: Render (One-click via render.yaml)

1. Push the included `render.yaml` to your repo (already added).
2. In Render, choose "New +" → "Blueprint" and point to your repo.
3. Set `OPENAI_API_KEY` in Environment.
4. Deploy. The service uses the `startCommand` and port env automatically.

## Option C: Railway

1. Create a new Railway project and connect your repo.
2. Railway detects Python; set the Start Command to:
   `python -m streamlit run src/dashboard.py --server.port $PORT --server.address 0.0.0.0`
3. Add the `OPENAI_API_KEY` variable.
4. Deploy.

## Option D: Hugging Face Spaces (Streamlit)

1. Create a new Space → Framework: Streamlit → Link your repo.
2. Set the app file to `src/dashboard.py` (or default).
3. Add `OPENAI_API_KEY` in the Space Secrets.
4. Spaces builds and hosts automatically.

## Tips
- Secrets: Always use platform secret managers, never commit keys.
- Cost Control: Use saved results in `data/results/` to demo without frequent API calls.
- Health Checks: Platforms expect the app to bind to `$PORT` and `0.0.0.0` (already configured in commands).
