FROM python:3.10

RUN apt update -y && apt install awscli -y
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]