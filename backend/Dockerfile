FROM python:3.10

COPY . /backend

WORKDIR /backend

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

