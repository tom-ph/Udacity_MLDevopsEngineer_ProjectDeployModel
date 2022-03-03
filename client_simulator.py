import os
import requests
import sys

sys.path.append(os.getcwd())

# Import our app from main.py.
from main import InputData

data = InputData.Config.schema_extra["example"]
r = requests.post("http://localhost:8000/predict/", json=data)