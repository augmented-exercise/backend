#! /usr/bin/env bash
cd ~/backend
source .env/bin/activate
gunicorn -b 127.0.0.1:8000 wsgi:app
