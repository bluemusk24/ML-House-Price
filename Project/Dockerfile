FROM python:3.10.13-slim

RUN pip install pipenv

WORKDIR /app 
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "LR.bin", "./"]

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "localhost:8080", "predict:app"]