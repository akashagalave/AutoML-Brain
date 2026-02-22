from locust import HttpUser, task, between
import random


class ChurnUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def predict_churn(self):

        payload = {
            "customer_id": f"{random.randint(1000,9999)}-TEST",
            "features": {
                "gender": random.choice(["Male", "Female"]),
                "SeniorCitizen": random.choice([0, 1]),
                "Partner": random.choice(["Yes", "No"]),
                "Dependents": random.choice(["Yes", "No"]),
                "tenure": random.randint(1, 72),
                "PhoneService": "Yes",
                "MultipleLines": random.choice(["Yes", "No"]),
                "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
                "OnlineSecurity": random.choice(["Yes", "No"]),
                "OnlineBackup": random.choice(["Yes", "No"]),
                "DeviceProtection": random.choice(["Yes", "No"]),
                "TechSupport": random.choice(["Yes", "No"]),
                "StreamingTV": random.choice(["Yes", "No"]),
                "StreamingMovies": random.choice(["Yes", "No"]),
                "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
                "PaperlessBilling": random.choice(["Yes", "No"]),
                "PaymentMethod": random.choice([
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]),
                "MonthlyCharges": round(random.uniform(20, 120), 2),
                "TotalCharges": round(random.uniform(20, 8000), 2)
            }
        }

        self.client.post("/predict", json=payload)