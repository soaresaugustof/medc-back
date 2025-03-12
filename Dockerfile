FROM python:3.10-slim

WORKDIR /medc-back

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY classificador.py .

COPY cardiom3.PNG .

CMD ["python", "main.py"]
