FROM python:3.9

WORKDIR /user/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY main.py .
COPY ml_object_detection.py .
COPY templates/index.html .

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0"]