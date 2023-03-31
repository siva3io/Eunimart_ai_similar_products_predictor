#!/bin/bash

#cd /home/ubuntu/vdezi_ai_similar_products_predictor    #replace service_repowith  your repo name 
cd /home/azureuser-datascience-prod/vdezi_ai_similar_products_predictor

git stash

git pull origin master

source ./venv/bin/activate

pip3 install -r requirements.txt

python3 serve.py



