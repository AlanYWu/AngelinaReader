import requests

def upload_file(url, file_path):
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file)}
        response = requests.post(url, files=files)
        return response

if __name__ == '__main__':
    url = 'http://localhost:12306/image_to_braille'
    file_path = 'input.jpg'  # Update this path to the file you want to upload
    response = upload_file(url, file_path)
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.text}')