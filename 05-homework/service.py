import requests

url = "http://localhost:9696/predict"

record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

response = requests.post(url, json=record)

conversion = response.json()

print ("response:", conversion)