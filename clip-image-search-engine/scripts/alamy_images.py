import requests
import xml.etree.ElementTree as ET

def fetch_alamy_image_urls(search_query):
    encoded_query = search_query.replace(' ', '%20')
    alamy_url = f"https://www.alamy.com/xml-search-results.asp?qt={encoded_query}&pn=1&ps=1000"

    try:
        response = requests.get(alamy_url)
        if response.status_code == 200:
            xml_content = response.content
            tree = ET.ElementTree(ET.fromstring(xml_content))
            
            images_data = []
            image_urls = []
            root = tree.getroot()
            for image in root.findall('.//I'):
                ar_filename = image.get('AR')
                
                # Construct image URL using the AR identifier
                image_url = f"https://c7.alamy.com/zooms/9/1/{ar_filename}.jpg"  
                image_urls.append(image_url)
                
                # Image data
                image_data = {
                    'ID': image.get('ID'),
                    'file_name': ar_filename,
                    'size': f"{image.get('PIX_X')} x {image.get('PIX_Y')}",
                    'type': image.get('TYPE'),
                    'caption': image.get('CAPTION'),
                    'date_taken': image.get('DATETAKEN')
                }
                images_data.append(image_data)
                
            # Returning the image URLs
            return image_urls, images_data
        else:
            return []

    except requests.RequestException:
        return []
