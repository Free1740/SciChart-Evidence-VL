FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command: build one demo EP/EA counterfactual pair.
CMD ["python", "scripts/build_counterfactual.py", "--demo", "--output", "/tmp/counterfactual_pairs.jsonl"]
