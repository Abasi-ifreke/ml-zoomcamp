import requests

url = "http://localhost:9696/predict"

customer = {
    "customerid": "8879-zkjof",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 1,
    "phoneservice": "no",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url, json=customer)

churn = response.json()

print ("response:", churn)

if churn["churn"] >= 0.5:
    print(churn["churn"])
    print("send email promo")

else:
    print("do nothing")