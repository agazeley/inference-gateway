import pytest
import requests

@pytest.fixture
def base_url():
    url = "http://localhost:3000"
    timeout = 300
    interval = 15
    iterations = int(timeout / interval)
    for _ in range(iterations):
        response = requests.get(f"{url}/readyz")
        if response.status_code == 200:
            break
        time.sleep(interval)
    return url

def test_health_check(base_url):
    response = requests.get(f"{base_url}/healthz")
    assert response.status_code == 200

@pytest.mark.parametrize("payload", [
    {
        "text": "The future of artificial intelligence",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    },
    {
        "text": "Exploring the universe",
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 0.8
    }
])
def test_generate_text(base_url, payload):
    response = requests.post(f"{base_url}/api/v1/inference", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert all(key in data for key in ["text", "model", "metadata"]), data


def test_post_transactions(base_url):
    url = f"{base_url}/api/v1/transactions"
    payload = {
        "transaction": {
            "prompt": "Test prompt",
            "response": "Test response"
        }
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 202, response.text
    assert response.json().get("id") != 0

def test_get_transactions(base_url):
    url = f"{base_url}/api/v1/transactions"
    response = requests.get(url)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data["transactions"], list), data
    assert isinstance(data["count"], int), data
