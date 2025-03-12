FROM python:3.10-slim

WORKDIR /medc-back

COPY .. /medc-back

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
