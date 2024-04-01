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
    
    # Kirim permintaan POST ke endpoint
    response = requests.post(url, json=data)
    
    # Pastikan respons memiliki status code 200 OK
    assert response.status_code == 200
    
    # Pastikan respons memiliki kunci "prediction"
    assert "prediction" in response.json()

if __name__ == "__main__":
    test_endpoint()
