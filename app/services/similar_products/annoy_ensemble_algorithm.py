from annoy import AnnoyIndex
import re
import base64
import logging
from tensorflow.python.ops.gen_array_ops import where
from app.utils import get_session
from config import Config
from app.models.products import Products, NewProducts
from io import BytesIO
import requests
import numpy as np
from PIL import Image
from annoy import AnnoyIndex
from urllib.parse import urlparse


class Similar_Image_Search_Engine:
    def __init__(self, category_data, image_annoy_model, title_annoy_model, image_path_list, vgg_model, bert_model):
        self.session = get_session(Config.SQLALCHEMY_DATABASE_URI)
        self.category_data = category_data
        self.logger = logging.getLogger(name=__name__)
        self.image_annoy_model = image_annoy_model
        self.title_annoy_model = title_annoy_model
        self.image_path_list = image_path_list
        self.vgg_model = vgg_model
        self.bert_model = bert_model

    def image_2_array_conversion(self, image_path):
        image = image_path
        image_vector = image.resize((224,224))
        return image_vector
    
    def image_feature_extraction(self, image_path):
        image_vector = self.image_2_array_conversion(image_path)
        if(image_vector==''):
            return ''
        extracted_image_features = self.vgg_model.predict(image_vector)
        extracted_image_features = extracted_image_features.reshape(extracted_image_features.shape[0], extracted_image_features.shape[1]*extracted_image_features.shape[2]*extracted_image_features.shape[3])
        return extracted_image_features
    
    def title_preprocessing(self, title):
        title = re.sub(r'^[a-zA-Z0-9]',' ',title)
        processed_title = ""
        for i in title.split(' '):
            if len(i)>=3 or i.isdigit(): processed_title+=i+" "
        return processed_title

    def feature_extraction(self, image_path, product_title):
        image_features = self.image_feature_extraction(image_path)
        title_features = self.bert_model.encode(self.title_preprocessing(product_title))
        return image_features, title_features
    
    def annoy_model_products_retrieval(self, annoy_model, features):
        similar_products = annoy_model.get_nns_by_vector(features, 20)
        return similar_products
    
    def top_40_similar_products_extraction(self, image_features, title_features):
        image_similar_products = self.annoy_model_products_retrieval(self.image_annoy_model, image_features)
        title_similar_products = self.annoy_model_products_retrieval(self.title_annoy_model, title_features)
        return image_similar_products+title_similar_products
    
    def extracted_features_concat(self, image_feature, title_feature):
        if image_feature.shape==(1,25088):concatenated_features = image_feature[0,:]
        else: concatenated_features = image_feature
        concatenated_features = np.append(concatenated_features, title_feature, axis=0)
        return concatenated_features

    def path_list_mapping_database(self, path_list_num):
        """
        #Code Updated Date - 17-11-2021

        If pid is in category_data, then image_url,product_title,fid is fetched from the category_data(CSV File)
        else, the required data is fetched from the database.

        #Added code to fetch fid also. 
        """
        pid = self.image_path_list[path_list_num].split('/')[-1].split('.')[0]
        if 'image_url' in self.category_data.columns:
            product = self.category_data[self.category_data['pid']==pid]
            if product.empty==False:        
                image_2 = list(product['image_url'])[0]; product_title = list(product['product_title'])[0]
                fid = list(product['fid'])[0] 
                return image_2, product_title, fid     

        products = self.session.query(Products.image_2, Products.product_title, Products.fid).filter(Products.pid==pid).all()
        if len(products)==0:
            products = self.session.query(NewProducts.image_2, NewProducts.product_title, NewProducts.fid).filter(NewProducts.pid==pid).all()
        return products[0][0], products[0][1], products[0][2]

    def image_request_extraction(self,image):
        try:
            parsed_image = urlparse(image)
            if all([parsed_image.scheme, parsed_image.netloc, parsed_image.path]):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
                '''Converts Grayscale to RGB'''
                if(img.mode =='L'):
                    img = img.convert("RGB")
            else:
                bytes_string = base64.b64decode(image)
                try:
                    img = Image.open(BytesIO(bytes_string))
                    if img.mode in ("RGBA", "P", "L"):    
                        img = img.convert("RGB")
                except:
                    return ''
            return img
        except Exception as e:
            self.logger.error(e,exc_info=True)
            return ''

    def top_25_similar_products_extraction(self, image_path:np.array, product_title:str)->list:
        annoy = AnnoyIndex(25856, 'angular')
        iterator = 0
        image_features, title_features = self.feature_extraction(image_path, product_title)
        image_features = image_features[0,:]
        
        top_40_similar_products = self.top_40_similar_products_extraction(image_features, title_features)
        #print(top_40_similar_products)
        top_40_similar_products = list(set(top_40_similar_products))
        top_40_similar_products_fids = []
        #print("top 40 similar products: ",top_40_similar_products)
        for each_product in top_40_similar_products:
            image_url, product_title, fid = self.path_list_mapping_database(each_product) #Added Fetching fid of the products
            #print(image_url) 
            image = self.image_request_extraction(image_url)
            if image!='':
                sub_category_image_features, sub_category_title_features = self.feature_extraction(image, product_title)
                if sub_category_image_features=='':
                    continue
                top_40_similar_products_fids.append(fid)
                annoy.add_item(iterator, self.extracted_features_concat(sub_category_image_features, sub_category_title_features))
                #annoy.add_item(iterator, self.extracted_features_concat(self.sub_category_image_features[each_product], self.sub_category_title_features[each_product]))
                iterator+=1
        annoy.build(10)
        top_25_similar_products = annoy.get_nns_by_vector(self.extracted_features_concat(image_features, title_features), 25)
        top_25_fids = [top_40_similar_products_fids[i] for i in top_25_similar_products] #mapping fids with index of top 10 similar products 
        
        """
        #Code Updated Date - 17-11-2021
        
        top_10_similar_products = [top_40_similar_products[i] for i in top_10_similar_products]
        top_10_pids = [self.image_path_list[i].split('/')[-1].split('.')[0] for i in top_10_similar_products]
        # print(top_10_pids)
        #self.plot_similar_images(image_path, top_10_similar_products)
        #return top_10_similar_products        
        #return top_10_pids
        """
        return top_25_fids