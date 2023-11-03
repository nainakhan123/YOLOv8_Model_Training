FROM python:3.10.2

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ /app/src

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
