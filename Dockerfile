FROM python:3.10.15

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]