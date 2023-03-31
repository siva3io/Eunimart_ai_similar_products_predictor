import logging
from flask import Blueprint, jsonify, request
from app.services.similar_products import SimilarProducts

similar_products = Blueprint('similar_products', __name__)

logger = logging.getLogger(__name__)    

@similar_products.route('/get_similar_products', methods=['POST'])
def get_similar_products():
    request_data = request.get_json()
    
    data = SimilarProducts.get_similar_products(request_data)
    if not data:
        data = {}
    return jsonify(data)

@similar_products.route('/dummy', methods=['GET'])
def dummy():
    request_data = request.get_json()
    print("request_data==>",request_data)
    return jsonify({"test":"Working"})