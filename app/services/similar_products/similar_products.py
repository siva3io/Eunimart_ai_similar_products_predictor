import os
import pickle
import base64
import logging
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from annoy import AnnoyIndex
from keras import backend as K
from urllib.parse import urlparse
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from app.utils import catch_exceptions,download_from_s3
from constants import channel_id_name_mapping
from app.models.products import Products,NewProducts
import joblib
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from .annoy_ensemble_algorithm import Similar_Image_Search_Engine
from constants import image_annoy_model, title_annoy_model, vgg_model, bert_model

logger = logging.getLogger(name=__name__)
class GetSimilarProducts(object):

    def __init__(self):
        pass
    
    @catch_exceptions
    def predict_similar_products(self,request_data):
        """     
        #Code Updated Date - 17-11-2021           
        if request_data["marketplace"]=="Amazon_India" or request_data['marketplace']=='Amazon_USA':
            path = request_data['marketplace']+'/'
            path_file_name = ['image_annoy.ann','title_annoy.ann','path_list.pkl']

            category_file_path = path+"{0}/{0}_Data.csv".format(request_data['category_name'])
            category_data = self.check_file_exists(request_data, category_file_path)
            category_data = pd.read_csv(category_file_path)
            files = []
            for each_path in path_file_name:
                file_path = path+"{0}/{1}/{1}_{2}".format(request_data['category_name'],request_data['sub_category_name'], each_path)
                self.check_file_exists(request_data, file_path)
                try:
                    files.append(joblib.load(file_path))
                except:
                    if each_path=='title_annoy.ann': 
                        title_annoy_model.load(file_path)
                        files.append(title_annoy_model)
                    if each_path=='image_annoy.ann': 
                        image_annoy_model.load(file_path)
                        files.append(image_annoy_model)

            sise = Similar_Image_Search_Engine(category_data, files[0], files[1], files[2], vgg_model, bert_model)
            image = self.get_image(request_data["image"])
            similar_pids = sise.top_10_similar_products_extraction(image, request_data['product_title']) 
            pids = [i.split('/')[-1].split('.')[0] for i in similar_pids]
            return pids

        """

        tokenized_query = request_data["product_title"].lower().split(" ")
        model_name = request_data["marketplace"]+ "/" + request_data["category_name"] + "/" + request_data["sub_category_name"] + '.model'
        csv_name = request_data["marketplace"] + "/" + request_data["category_name"] + "/" + request_data["sub_category_name"] + '.csv'
        
        model_path = self.check_file_exists(request_data,model_name)
        csv_path = self.check_file_exists(request_data,csv_name)
        
        if csv_path:
            with open(csv_path,'rb') as csv_file:
                df = pd.read_csv(csv_file)
        if model_path:
            with open(model_path,'rb') as file:
                bm25 = pickle.load(file)
                similar_pids = bm25.get_top_n(tokenized_query, df.fid.values, n=25)
                return similar_pids
        return ""
    

    @catch_exceptions
    def check_file_exists(self,request_data,abs_file_path):
        """
        This function is used to check if required files exists, if not we call 'download_from_s3_file' function.
        """
        try:
            if request_data["marketplace"] in ["Amazon_India","Amazon_USA"]:    #List of marketplaces trained with annoy ensemble approach to be updated here. 
                s3_path = 'Similar_Products/' + abs_file_path
            else:
                s3_path = 'bm25_models/'+abs_file_path
            if not os.path.exists(abs_file_path):
                print('===> abs_path :',abs_file_path)
                os.makedirs("/".join(abs_file_path.split('/')[:-1]), exist_ok=True)
                download_from_s3(s3_path,abs_file_path)
            return abs_file_path
        except Exception as e:
            logger.error(e,exc_info=True)


    @catch_exceptions
    def get_image(self,image):
        """
        This function is used to download image given in request_data.
        """
        try:
            parsed_image = urlparse(image)
            if all([parsed_image.scheme, parsed_image.netloc, parsed_image.path]):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                bytes_string = base64.b64decode(image)
                img = Image.open(BytesIO(bytes_string))
                if img.mode in ("RGBA", "P"):    
                    img = img.convert("RGB")
            return img
        except Exception as e:
            logger.error(e,exc_info=True)
            return ''
    

    @catch_exceptions
    def get_image_features(self,image):
        """
        This function is used to extract the image features.
        """
        try:
            image_width, image_height = 224, 224
            image = image.resize((image_width,image_height)) 
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            model = VGG16(include_top=False, weights='imagenet')   
            image_features = model.predict(image)
            K.clear_session()
            image_features = image_features.reshape((25088,))
            return image_features
        except Exception as e:
            logger.error(e,exc_info=True)
    

    @catch_exceptions
    def fetch_competitive_products(self, similar_fids,marketplace):
        final_products_dict = dict()
        similar_fids = [x for x in similar_fids if str(x) != 'nan']
        # products_query = {
            # "fid__in":similar_fids
        # }
        try:
            engine = create_engine("postgresql://eunimart_user:o2R7chzDlxEYKDvp2g2W@crawl-data.c9xsogbkltnk.us-east-2.rds.amazonaws.com:5432/crawled_data")
            cursor = engine.connect()
            product_list = []
            """
            #Code Updated Date - 17-11-2021            

            if marketplace=='Amazon_India' or marketplace=='Amazon_USA':
                products = text('''select id,pid,fid,image_2 ,description,product_title from products where pid in {} '''.format(str(tuple(similar_fids))))
                product_list = cursor.execute(products).fetchall()

                '''If the products are not available in products table, it will be available in new_products table.'''

                if len(product_list) < len(similar_fids):
                    products = text('''select id,pid,fid,image_2 ,description,product_title from new_products where pid in {} '''.format(str(tuple(similar_fids))))
                    new_product_list = cursor.execute(products).fetchall()
                    product_list = product_list+new_product_list

            else:
                products = text('''select id, pid,fid, image_2, description, product_title from products where fid in {}'''.format(str(tuple(similar_fids))))
                product_list = cursor.execute(products).fetchall()

            df = pd.DataFrame(product_list,columns = ['id','pid','fid','image_2','description','product_title'])
            for ind in df.index:
                final_products_dict[df["image_2"][ind]] = df["fid"][ind]

            """
            products = text('''select id, pid,fid, image_2, description, product_title from products where fid in {}'''.format(str(tuple(similar_fids))))
            product_list = cursor.execute(products).fetchall()

            df = pd.DataFrame(product_list,columns = ['id','pid','fid','image_2','description','product_title'])
            for ind in df.index:
                final_products_dict[df["image_2"][ind]] = df["fid"][ind]
           
        except Exception as e:
            logger.error(e,exc_info=True)
            #return self.fetch_competitive_products(similar_fids, marketplace)
            return {}   #Code updated on 31-01-2022. commented above line to stop recursion.
        return final_products_dict


    @catch_exceptions
    def similar_image_retrival(self,request_data,input_image):
        
        if request_data["marketplace"] in ["Amazon_India","Amazon_USA"]:    #List of marketplaces trained with annoy ensemble approach to be updated here. 
            path = request_data['marketplace']+'/'
            path_file_name = ['image_annoy.ann','title_annoy.ann','path_list.pkl']
            """
            #Code Updated Date - 17-11-2021 
            #Changed csv filename '*_Data.csv' to '*_Updated_Data.csv' 
            category_file_path = path+"{0}/{0}_Data.csv".format(request_data['category_name']) #CSV with no fid.
            """
            category_file_path = path+"{0}/{0}_Updated_Data.csv".format(request_data['category_name']) #Updated CSV containing fid of products.
            category_data = self.check_file_exists(request_data, category_file_path)
            category_data = pd.read_csv(category_file_path)
            files = []
            for each_path in path_file_name:
                file_path = path+"{0}/{1}/{1}_{2}".format(request_data['category_name'],request_data['sub_category_name'], each_path)
                self.check_file_exists(request_data, file_path)
                try:
                    files.append(joblib.load(file_path))
                except:
                    if each_path=='title_annoy.ann': 
                        title_annoy_model.load(file_path)
                        files.append(title_annoy_model)
                    if each_path=='image_annoy.ann': 
                        image_annoy_model.load(file_path)
                        files.append(image_annoy_model)

            sise = Similar_Image_Search_Engine(category_data, files[0], files[1], files[2], vgg_model, bert_model)
            image = self.get_image(request_data["image"])
            similar_fids = sise.top_25_similar_products_extraction(image, request_data['product_title']) # fids of top 10 similar products
            #fids = [i.split('/')[-1].split('.')[0] for i in similar_fids] #code updated on 25-11-2021 --> fids are already in required format. 
            return similar_fids
        #Updated code to return 25 similar products
        '''
        BM25 and text tokenization approach to extract similar products. 
        This approach is for marketplaces that is not yet trained to use annoy_ensemble_algorithm.py 
        '''
        annoy_model={}
        mapping=0
        filtered_indexes = annoy_model.get_nns_by_vector(self.get_image_features(input_image), 25)[:25]
        similar_products = []
        for indx in filtered_indexes:
            similar_products.append(mapping[indx])

        return similar_products

    @catch_exceptions
    def get_similar_products(self,request_data):
        try:

            response_data = {}
            missing_fields = []
            mandatory_fields = ["channel_id","category_name","sub_category_name","image","product_title"]
            for field in mandatory_fields:
                if not field in request_data["data"]:
                    missing_fields.append(field)
            response_data = {
                "status":False,
                "message":"Required field is missing",
                "error_obj":{
                    "description":"{} is/are missing".format(','.join(missing_fields)),
                    "error_code":"REQUIRED_FIELD_IS_MISSING"
                }
            }
            if len(missing_fields) == 0:
                request_data["data"]["category_name"] = request_data["data"]["category_name"].replace(" ","_")
                request_data["data"]["sub_category_name"] = request_data["data"]["sub_category_name"].replace(" ","_")
                request_data["data"]["marketplace"] = channel_id_name_mapping.get(request_data["data"]["channel_id"])
                image = self.get_image(request_data["data"]["image"])
                if image:
                    similar_products = self.similar_image_retrival(request_data["data"],image)
                    response_data = {
                        "status":True,
                        "data":{
                        "similar_products":similar_products
                        }
                    }

            return response_data
        except Exception as e:
            logger.error(e,exc_info=True)

SimilarProducts = GetSimilarProducts()