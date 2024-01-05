FROM python:3.10.2

RUN apt-get update && apt-get install -y libgl1-mesa-glx awscli

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ /app/src

# CMD ["uvicorn", "src.handler:app", "--host", "0.0.0.0", "--port", "9000"]
CMD ["sh", "-c", "MPLCONFIGDIR=/tmp/matplotlib && uvicorn src.handler:app --host 0.0.0.0 --port 9000"]


EXPOSE 9000
