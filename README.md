# FastApi
this api is called by the backend, it is responsible for generating responses for the caller. 

python -m venv myenv
myenv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

to run:
uvicorn app:app --reload

this will allow users to access the api for that run the testAPI.py in a seperate VS folder by "python testAPI.py".