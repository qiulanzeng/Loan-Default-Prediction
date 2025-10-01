FROM python:3.10

RUN apt update -y && apt install awscli -y
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]