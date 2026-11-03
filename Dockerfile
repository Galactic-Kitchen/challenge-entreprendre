# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY . .
RUN pip install --no-cache-dir .

EXPOSE 5000

CMD ["python", "src/app.py"]
