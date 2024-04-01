from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import logging
import psycopg2
import os

app = FastAPI()

# Load the model
with open("logistic_regression_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Koneksi ke database PostgreSQL
conn = psycopg2.connect(
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"],
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"]
)

# Fungsi untuk membuat tabel user_predicts
def create_user_predicts_table():
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_predicts (
            id SERIAL PRIMARY KEY,
            transaction numeric,
            age varchar(10),
            tenure numeric,
            num_pages_visited numeric,
            has_credit_card boolean,
            items_in_cart numeric,
            purchase_prediction boolean
        );
    """)
    conn.commit()
    cursor.close()

# Fungsi untuk menambahkan data user_predicts baru
def add_user_prediction(transaction, age, tenure, num_pages_visited, 
                        has_credit_card, items_in_cart, purchase_prediction):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_predicts (transaction, age, tenure, num_pages_visited, 
                                  has_credit_card, items_in_cart, purchase_prediction)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (transaction, age, tenure, num_pages_visited, 
          has_credit_card, items_in_cart, purchase_prediction))
    conn.commit()
    cursor.close()

templates = Jinja2Templates(directory="templates")

@app.get('/app/', response_class=HTMLResponse)
def index(request: Request):
    context = {'request' : request}
    return templates.TemplateResponse('app.html', context)

# Pydantic model for user input
class UserInput(BaseModel):
    transaction: int
    age: int
    tenure: int
    num_pages_visited: int
    has_credit_card: bool
    items_in_cart: int

# Configure logger
logging.basicConfig(filename='prediction_logs.log', level=logging.INFO)

@app.post("/predict/")
def predict(user_input: UserInput):
    data = user_input.dict()
    transaction = data["transaction"]
    age = data["age"]
    tenure = data["tenure"]
    num_pages_visited = data["num_pages_visited"]
    has_credit_card = data["has_credit_card"]
    items_in_cart = data["items_in_cart"]
    
    # Perform prediction using the loaded model
    prediction = loaded_model.predict([[transaction, age, tenure, 
                                        num_pages_visited, has_credit_card, items_in_cart]])
    
    # Log prediction
    logging.info(f"Prediction for user input {data}: {int(prediction[0])}")
    
    # Add prediction to database
    add_user_prediction(transaction, age, tenure, num_pages_visited, 
                        has_credit_card, items_in_cart, bool(prediction[0]))
    
    return {"prediction": int(prediction[0])}

# Panggil fungsi untuk membuat tabel saat aplikasi dimulai
create_user_predicts_table()

