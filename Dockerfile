FROM python:3.10-slim

WORKDIR /medc-back

COPY requirements.txt /medc-back

COPY main.py /medc-back

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
