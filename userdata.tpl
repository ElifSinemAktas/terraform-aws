#!/bin/bash
sudo yum -y update
sudo yum -y install git
python3 -m pip install virtualenv
python3 -m virtualenv fastapi
source /fastapi/bin/activate
git clone https://github.com/ElifSinemAktas/terraform-docker-fastapi.git
cd /terraform-docker-fastapi
pip install -r requirements.txt
gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8002 --daemon