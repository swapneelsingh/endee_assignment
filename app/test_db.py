import requests

# Endee default port
BASE_URL = "http://localhost:8080/api/v1"

def test_endee_connection():
    print("Attempting to connect to Endee Database...")
    try:
        response = requests.get(f"{BASE_URL}/index/list")
        if response.status_code == 200:
            print("✅ SUCCESS! Python is talking to Endee.")
            print("Database Response:", response.json())
        else:
            print(f"⚠️ Reached the server, but got status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect.")

if __name__ == "__main__":
    test_endee_connection()