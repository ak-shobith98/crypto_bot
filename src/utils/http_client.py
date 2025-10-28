# File: src/utils/http_client.py

import requests
import time

def safe_get(url, retries=3, delay=2, timeout=8):
    """
    Wrapper for requests.get with retries and safe JSON parsing.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            print(f"[WARN] Attempt {attempt}: HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            print(f"[ERROR] Attempt {attempt} failed: {e}")
        time.sleep(delay)
    print(f"[ERROR] All {retries} attempts failed for {url}")
    return {}
