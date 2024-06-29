import requests

# URL of the API endpoint
url = 'http://localhost:5000/upload'  # Adjust the port if your server runs on a different one

# Path to the image you want to upload
image_path = 'input.jpg'

# Open the image file in binary mode
with open(image_path, 'rb') as image_file:
    # Define the name of the file as a dictionary
    files = {'file': (image_path, image_file)}
    # Send the POST request with the image file
    response = requests.post(url, files=files)

# Print the response from the server
print(response.text)