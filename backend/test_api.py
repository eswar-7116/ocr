# test_api.py
import requests
import time
import os
import json
import sys

# --- Configuration ---
# 🚨 IMPORTANT: Change FILE_NAME to one of your sample files (e.g., 'sampleid.png')
FILE_NAME = "./sampleid.png" 
FILE_PATH = os.path.join(os.getcwd(), FILE_NAME) 

url = 'http://127.0.0.1:5000/api/extract'

if not os.path.exists(FILE_PATH):
    print(f"Error: Sample file not found at {FILE_PATH}. Please ensure '{FILE_NAME}' is in the backend directory.")
    sys.exit(1) # Exit if the test file is missing

# Check for requests dependency
try:
    import requests 
except ImportError:
    print("Error: 'requests' library not found. Please install it with: pip install requests")
    sys.exit(1)

# --- Test Execution ---

print(f"Testing API with {FILE_NAME}. Measuring CPU inference time...")
start_time = time.time()

try:
    with open(FILE_PATH, 'rb') as f:
        files = {'document': (FILE_NAME, f, 'image/png')}
        data = {'doc_type': 'printed'}
        
        # Set a very long timeout (5 minutes) just in case the CPU is still very slow
        response = requests.post(url, files=files, data=data, timeout=300) 

    end_time = time.time()
    print(f"\n--- API Test Result ---")
    print(f"Status Code: {response.status_code}")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        print("\nSUCCESS: Extraction complete!")
        # Print only the extracted fields for clarity
        try:
            result_json = response.json()
            print("Extracted Fields (for speed test):")
            print(json.dumps(result_json['extracted_fields'], indent=4))
        except Exception:
             print("Raw response (could not parse as JSON):", response.text)
    else:
        print(f"API Error: {response.json()}")

except requests.exceptions.Timeout:
    print(f"\nERROR: The request timed out after 300 seconds. Inference is too slow for deployment.")
except Exception as e:
    print(f"\nAn unexpected error occurred during the request: {e}")