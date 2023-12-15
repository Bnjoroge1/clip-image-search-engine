import sys
sys.path.append('C:\\Users\\hp\\Downloads\\image-search-engine')
import pysolr
from elasticsearch import Elasticsearch
import numpy as np
import faiss 
from scripts.alamy_images import fetch_alamy_image_urls
from clip_image_search.clip_feature_extractor import CLIPFeatureExtractor
from clip_image_search.process_data import clip_process_data

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'ec2-16-170-159-232.eu-north-1.compute.amazonaws.com', 'port': 9200}])

# Establish a connection to your Solr instance
solr = pysolr.Solr('http://ec2-16-170-159-232.eu-north-1.compute.amazonaws.com:8984/solr/features/', timeout=10)

# Indexing data into Elasticsearch
def index_data(keyword, results):
    es.index(index='image_search_index', body={'keyword': keyword, 'results': results})
    
def search_and_index_new_keyword(keyword):
    existing_doc = es.search(index='image_search_index', body={'query': {'match': {'keyword': keyword}}})
    if existing_doc['hits']['total']['value'] == 0:
        # Keyword doesn't exist in the index, perform a new search and index the results
        img_urls, img_data = fetch_alamy_image_urls(keyword)
        new_results, list_of_urls_and_features = clip_process_data(keyword, img_urls, img_data)  # Fetch search results using process_data.py
        for t in list_of_urls_and_features:
            features_str = ';'.join(','.join(str(f) for f in feature) for feature in t[1])
            doc = {
                'id': t[0],
                'image_features': features_str
            }
            solr.add([doc])
        index_data(keyword, new_results)  # Index keyword and results
        return new_results
    else:
        # Keyword already exists in the index, fetch and display the existing results
        existing_results = existing_doc['hits']['hits'][0]['_source']['results']
        return existing_results

def convert_str_to_list(features_str):
    return [list(map(float, feature.split(','))) for feature in features_str]

# Function to normalize vectors
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Function to perform image search based on uploaded image
def perform_image_search(uploaded_image):
    # Initialize CLIP feature extractor
    clip_extractor = CLIPFeatureExtractor()

    total = solr.search('*:*', rows=0)
    # Get the total count of documents
    total_docs = total.hits

    # Extract features of the uploaded image
    uploaded_image_features = clip_extractor.get_image_features([uploaded_image])
    uploaded_image_np = np.array(uploaded_image_features).astype('float32')

    results = solr.search('*:*', fl='id,image_features', rows = total_docs)

    image_urls = []
    image_features = []
    for result in results:
        image_urls.append(result['id'])
        image_features.append(convert_str_to_list(result['image_features']))
        
    # Convert features to a suitable numpy array for FAISS and normalize
    image_features_array = np.array(image_features).astype('float32')
    image_features_normalized = np.vstack([normalize(vec) for vec in image_features_array])

    # Build an index for FAISS
    index1 = faiss.IndexFlatIP(image_features_normalized.shape[1])  # Using Inner Product (IP) index
    index1.add(image_features_normalized)

    # Normalize the uploaded image features
    uploaded_image_normalized = normalize(uploaded_image_np[0])

    # Perform a similarity search using FAISS
    k = 40  # Number of similar images to retrieve
    uploaded_image_normalized = np.expand_dims(uploaded_image_normalized, axis=0)
    distances, indices = index1.search(uploaded_image_normalized, k)

    similar_images_urls = []
    for i in indices[0]:
        similar_images_urls.append(image_urls[i])

    return similar_images_urls

def perform_image_search_for_keyword(uploaded_image, img_urls):
    # Initialize CLIP feature extractor
    clip_extractor = CLIPFeatureExtractor()

    # Extract features of the uploaded image
    uploaded_image_features = clip_extractor.get_image_features([uploaded_image])
    uploaded_image_np = np.array(uploaded_image_features).astype('float32')

    l = len(img_urls)

    # Fetch features for all URLs
    image_features = []
    for url in img_urls:
        result = solr.search('id:"{}"'.format(url), fl='image_features')
        for res in result:
            image_features.append(convert_str_to_list(res['image_features']))

    # Convert features to a suitable numpy array for FAISS
    image_features_array = [item for sublist in image_features for item in sublist]
    image_features_np = np.array(image_features_array).astype('float32')

    # Build an index for FAISS
    index2 = faiss.IndexFlatL2(image_features_np.shape[1])
    index2.add(image_features_np)

    # Perform a similarity search using FAISS
    k = l  # Number of similar images to retrieve
    distances, indices = index2.search(uploaded_image_np, k)

    similar_images_urls = []
    for i in indices[0]:
        similar_images_urls.append(img_urls[i])

    return similar_images_urls

if __name__=="__main__":
    
    keywords = [ "apple", "ball", "banana", "bed", "bicycle", "bird", "book", "bottle", "box", "bridge",
    "bus", "butterfly", "cake", "camera", "car", "cat", "chair", "clock", "cloud", "coffee",
    "computer", "cookie", "cow", "cup", "desk", "dog", "door", "dragon", "dress", "duck",
    "elephant", "eye", "fish", "flag", "flower", "frog", "guitar", "hammer", "hat", "heart",
    "house", "ice cream", "island", "jacket", "key", "kite", "lamp", "leaf", "leg", "lemon",
    "lion", "lock", "map", "milk", "moon", "mouse", "mountain", "mug", "mushroom", "nose",
    "ocean", "orange", "owl", "panda", "pear", "pen", "pencil", "phone", "pig", "pillow",
    "pizza", "plane", "plant", "potato", "rabbit", "rainbow", "robot", "rocket", "rose",
    "sandwich", "scarf", "scissors", "shark", "sheep", "shirt", "shoe", "snail", "snake",
    "snowflake", "socks", "spider", "spoon", "star", "strawberry", "sun", "table", "teapot",
    "teddy bear", "television", "tent", "tiger", "toothbrush", "toothpaste", "train", "tree",
    "truck", "turtle", "umbrella", "unicorn", "vase", "violin", "watch", "watermelon", "whale",
    "wheel", "window", "wine glass", "wolf", "yoga", "zebra"]

    for k in keywords:
        img_urls, img_data = fetch_alamy_image_urls(k)
        search_results = clip_process_data(k, img_urls, img_data)  # Fetch search results using process_data
        index_data(k, search_results)  # Index keyword and results
        print(k, "indexed")