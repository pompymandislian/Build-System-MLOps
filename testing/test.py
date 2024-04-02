import requests

def test_endpoint():
    url = "http://localhost:8000/predict/"
    
    # Data yang akan diuji
    data = {
        "transaction": 100,
        "age": 35,
        "tenure": 2,
        "num_pages_visited": 5,
        "has_credit_card": True,
        "items_in_cart": 3
    }
    
    # Pencetakan informasi debug
    print("URL:", url)
    
    # Kirim permintaan POST ke endpoint
    try:
        response = requests.post(url, json=data)
        
        # Pastikan respons memiliki status code 200 OK
        if response.status_code == 200:
            print("Response:", response.json())
            # Pastikan respons memiliki kunci "prediction"
            assert "prediction" in response.json()
        else:
            print("Error:", response.status_code, response.text)
    
    # Tangani kesalahan saat mengirim permintaan
    except requests.exceptions.RequestException as e:
        print("Error:", e)

if __name__ == "__main__":
    test_endpoint()
