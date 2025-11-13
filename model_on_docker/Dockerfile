FROM python:3.9

COPY requirements.txt /app/
COPY . /app/

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]
