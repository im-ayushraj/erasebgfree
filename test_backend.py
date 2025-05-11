import requests
import os
from PIL import Image
import io

BASE_URL = 'http://localhost:5000'

def test_remove_background():
    print("Testing background removal functionality...")
    
    # Test 1: Valid image upload
    print("\nTest 1: Uploading valid image...")
    test_image_path = 'test_image.jpg'
    
    # Create a test image if it doesn't exist
    if not os.path.exists(test_image_path):
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_image_path)
    
    with open(test_image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{BASE_URL}/remove-bg', files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        # Test image retrieval
        image_url = response.json()['processed_image_url']
        image_response = requests.get(f'{BASE_URL}{image_url}')
        print(f"Image retrieval status: {image_response.status_code}")
    
    # Test 2: Invalid file type
    print("\nTest 2: Uploading invalid file type...")
    with open('test.txt', 'w') as f:
        f.write('test content')
    
    with open('test.txt', 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{BASE_URL}/remove-bg', files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 3: No file
    print("\nTest 3: Uploading no file...")
    response = requests.post(f'{BASE_URL}/remove-bg')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    if os.path.exists('test.txt'):
        os.remove('test.txt')

if __name__ == '__main__':
    test_remove_background() 