import os
import requests
import sys

sys.path.append(os.getcwd())

# Import our app from main.py.
from main import InputData

BASE_URL = "https://udacitymlops-project-deploy.herokuapp.com/"

data = InputData.Config.schema_extra["example"]
r = requests.post(BASE_URL + "predict/", json=data)
print(r.status_code)
print(r.json())
1==1