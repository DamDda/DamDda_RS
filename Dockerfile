FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# RUN pip install --upgrade pip

RUN apt-get update -y && \
    apt-get install -y build-essential libmariadb-dev && \
    apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]