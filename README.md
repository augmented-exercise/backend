# Backend
This is the backend for the sYZTMic app.
## Getting started
```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
export FLASK_APP=server
flask run
```
## Deployment
We can deploy to localhost:8000 with
```
bash run_server.sh
````
And setup a reverse proxy such as nginx
