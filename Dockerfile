FROM python:3.10-slim

WORKDIR /medcare

COPY requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend /medcare/backend

CMD ["python", "backend/main.py"]
