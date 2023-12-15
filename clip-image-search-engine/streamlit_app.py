import sys
sys.path.append('C:\\Users\\hp\\Downloads\\image-search-engine')
import streamlit as st
from PIL import Image as PILImage
from io import BytesIO
import base64
import requests
import time
from scripts.index_data import search_and_index_new_keyword
from scripts.index_data import perform_image_search
from scripts.index_data import perform_image_search_for_keyword

# Streamlit UI
def main():
    st.title("CLIP Image Search Engine")
    st.write("This search engine uses CLIP (Contrastive Language-Image Pretraining) to perform image searches based on text input, image input, or a combination of both. You can upload an image or enter text queries to find relevant images.")

    # Sidebar for user input
    st.sidebar.header("Search Options")
    search_option = st.sidebar.selectbox("Choose Search Option", ["Text Input", "Image Input", "Text + Image Input"])

    if search_option == "Text Input":
        text_query = st.sidebar.text_input("Enter Text Query")
        if st.sidebar.button("Search"):
            if len(text_query)>0:
                st.write("Performing search based on text input...")
                start_time = time.time()  # Record start time
                img = search_and_index_new_keyword(text_query)  
                end_time = time.time()  # Record end time
                st.write(f"Time taken: {int(end_time - start_time)} seconds")  # Display time taken
                # Display images on the main section
                display_images(img)
            else:
                st.warning("Please enter a text.")

    elif search_option == "Image Input":
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        if st.sidebar.button("Search"):
            if uploaded_file is not None:
                st.write("Performing search based on uploaded image...")
                start_time = time.time()  # Record start time
                uploaded_image = PILImage.open(uploaded_file)
                img = perform_image_search(uploaded_image)
                end_time = time.time()  # Record end time
                st.write(f"Time taken: {int(end_time - start_time)} seconds")  # Display time taken
                # Display images on the main section
                display_images(img)
            else:
                st.warning("Please upload an image.")
    
    elif search_option == "Text + Image Input":
        text_query = st.sidebar.text_input("Enter Text Query")
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        if st.sidebar.button("Search"):
            if (uploaded_file is not None) and (len(text_query)>0):
                st.write("Performing search based on text and image input...")
                start_time = time.time()  # Record start time
                uploaded_image = PILImage.open(uploaded_file)
                img = search_and_index_new_keyword(text_query) 
                result = perform_image_search_for_keyword(uploaded_image, img)
                end_time = time.time()  # Record end time
                st.write(f"Time taken: {int(end_time - start_time)} seconds")  # Display time taken
                # Display images on the main section
                display_images(result)
            else:
                if len(text_query)==0:
                    st.warning("Please enter a text.")
                if uploaded_file is None:
                    st.warning("Please upload an image.")

def crop_image_caption(image, percentage):
    width, height = image.size
    crop_height = int(height * percentage)
    cropped_image = image.crop((0, 0, width, height - crop_height))
    return cropped_image

def pil_to_b64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_images(images):
    st.header("Search Results")
    if not images:
        st.write("No results found.")
    else:
        columns = 2  # Number of columns in the grid
        image_size = 300  # Set the size of images (adjust as needed)
        caption_percentage = 0.2  # Percentage of image height to be cropped from the bottom

        for i in range(0, len(images), columns):
            col_images = images[i:i + columns]
            col1, col2 = st.columns(2)

            for url in col_images:
                response = requests.get(url)
                if response.status_code != 200: 
                    continue
                uploaded_image = PILImage.open(requests.get(url, stream=True).raw)
                cropped_image = crop_image_caption(uploaded_image, caption_percentage)
                cropped_b64 = pil_to_b64(cropped_image)
                img = f'<a href="{url}" target="_blank"><img src="data:image/png;base64,{cropped_b64}" style="width: {image_size}px; height: {image_size}px; object-fit: cover;"></a>'
                col1.markdown(img, unsafe_allow_html=True)

                # Display second image in the second column
                if len(col_images) == 1:
                    col2.markdown("", unsafe_allow_html=True)
                else:
                    col_images.pop(0)
                    url = col_images[0]
                    uploaded_image = PILImage.open(requests.get(url, stream=True).raw)
                    cropped_image = crop_image_caption(uploaded_image, caption_percentage)
                    cropped_b64 = pil_to_b64(cropped_image)
                    img = f'<a href="{url}" target="_blank"><img src="data:image/png;base64,{cropped_b64}" style="width: {image_size}px; height: {image_size}px; object-fit: cover;"></a>'
                    col2.markdown(img, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()