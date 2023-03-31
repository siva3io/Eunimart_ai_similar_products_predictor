from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer 

channel_id_name_mapping = {
    "1"    : "Amazon_USA",
    "12"   : "Amazon_India",
    "24"   : "Etsy"
}
vgg_model = VGG16(weights='imagenet', include_top = False, input_shape=(224,224,3))
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
image_annoy_model = AnnoyIndex(25088, 'angular')
title_annoy_model = AnnoyIndex(768, 'angular')