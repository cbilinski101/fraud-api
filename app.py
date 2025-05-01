from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("optimized_model.joblib")

class Transaction(BaseModel):
    amt: float
    city_pop: int
    age: int
    day_of_week: int
    hour_of_day: int
    category_entertainment: int
    category_food_dining: int
    category_gas_transport: int
    category_grocery_net: int
    category_grocery_pos: int
    category_health_fitness: int
    category_home: int
    category_misc_net: int
    category_misc_pos: int
    category_personal_care: int
    category_shopping_net: int
    category_shopping_pos: int
    category_travel: int

@app.get("/")
def read_root():
    return {"message": "Fraud detection API is live!"}

@app.post("/predict")
def predict(tx: Transaction):
    input_data = np.array([[getattr(tx, field) for field in tx.__annotations__.keys()]])
    prediction = model.predict(input_data)[0]
    return {"is_fraud": int(prediction)}
