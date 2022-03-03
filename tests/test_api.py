import json

from fastapi.testclient import TestClient
import os
import sys

sys.path.append(os.getcwd())

# Import our app from main.py.
from main import app, InputData

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    test_message = "Hello stranger, welcome to the Udacity Machine Learning DevOps Engineer Model Deploy exercise!"
    r = client.get("/")
    assert r.status_code == 200
    assert json.loads(r.content)["message"]==test_message

def test_api_locally_predict_0():
    data = InputData.Config.schema_extra["example"]
    r = client.post("/predict/", json=data)
    result = r.json()
    assert r.status_code == 200
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert result["predictions"][0]=="<=50K"

def test_api_locally_predict_1():
    data = InputData.Config.schema_extra["example_1"]
    r = client.post("/predict/", json=data)
    result = r.json()
    assert r.status_code == 200
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert result["predictions"][0]==">50K"

if __name__=="__main__":
    pass