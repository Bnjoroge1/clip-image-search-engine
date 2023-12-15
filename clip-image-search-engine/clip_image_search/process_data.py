from concurrent.futures import ThreadPoolExecutor
from clip_image_search.clip_feature_extractor import CLIPFeatureExtractor
from clip_image_search.similarity import calculate_similarity
from clip_image_search.utils import load_image_from_url

# Initialize CLIP feature extractor
clip_extractor = CLIPFeatureExtractor()

# Function to process image data and perform similarity ranking
def clip_process_data(search_query, image_urls, images_data):
    image_embeddings = []
    caption_embeddings = []
    list_of_urls_and_features = []

    def process_image(image_url, image_info):
        try:
            image_data = load_image_from_url(image_url)
            
            if image_data is not None:
                # Extract embeddings using CLIP feature extractor for the current image
                image_embedding = clip_extractor.get_image_features(image_data)
                image_embeddings.append(image_embedding)
                t = (image_url, image_embedding)
                list_of_urls_and_features.append(t)
        
                # Extract embeddings for image captions from images_data
                caption = image_info['caption']
                caption_embedding = clip_extractor.get_text_features(caption)
                caption_embeddings.append(caption_embedding)
            else:
                # Handle the case when image data loading fails
                print(f"Failed to load image data from URL: {image_url}")
        except Exception as e:
            # Handle exceptions related to image data loading or processing
            print(f"Error processing image from URL {image_url}: {str(e)}")

    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        for image_url, image_info in zip(image_urls, images_data):
            executor.submit(process_image, image_url, image_info)

    # Get text features (embedding) for the search query using CLIP feature extractor
    text_embedding = clip_extractor.get_text_features(search_query)

    # Calculate similarity scores between text embedding and image/caption embeddings
    similarity_scores = []
    for img_emb, cap_emb in zip(image_embeddings, caption_embeddings):
        # Calculate similarity scores for image embeddings and caption embeddings
        similarity_score_img = calculate_similarity(img_emb, text_embedding)
        similarity_score_caption = calculate_similarity(cap_emb, text_embedding)
        # Combine the similarity scores from image and caption
        similarity_scores.append((similarity_score_img + similarity_score_caption) / 2)

    # Rank images based on similarity scores
    ranked_images = [url for _, url in sorted(zip(similarity_scores, image_urls), reverse=True)]
    return ranked_images, list_of_urls_and_features