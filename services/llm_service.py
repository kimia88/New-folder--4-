import requests
import json
import re
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class QService:
    BASE_URL = "https://qwen-qwq-32b-preview.hf.space"

    def __init__(self, session_hash):
        self.session_hash = session_hash

    def predict(self, text):
        url = f"{self.BASE_URL}/run/predict"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        data = {
            "data": [{"files": [], "text": text}, [[{"id": None, "elem_id": None, "elem_classes": None, "name": None, "text": text, "flushing": None, "avatar": "", "files": []}, [{"id": None, "elem_id": None, "elem_classes": None, "name": None, "text": "", "flushing": None, "avatar": "", "files": []}, None, None]]], None],
            "event_data": None,
            "fn_index": 1,
            "trigger_id": 5,
            "session_hash": self.session_hash
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print("❌ Predict error:", e)
            return None

    def send_request(self, text):
        self.predict(text)
        url = f"{self.BASE_URL}/queue/join"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        data = {
            "data": [[[{"id": None, "elem_id": None, "elem_classes": None, "name": None, "text": text, "flushing": None, "avatar": "", "files": []}, None]], None, 0],
            "event_data": None,
            "fn_index": 2,
            "trigger_id": 5,
            "session_hash": self.session_hash
        }
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
            if res.status_code == 200:
                return res.json()
            else:
                print("❌ Join error:", res.status_code)
                return None
        except requests.exceptions.RequestException as e:
            print("❌ Join request error:", e)
            return None

    def get_response(self, wait_timeout=60, retries=3):
        url = f"{self.BASE_URL}/queue/data?session_hash={self.session_hash}"
        headers = {
            "Accept": "text/event-stream",
            "User-Agent": "Mozilla/5.0"
        }
        for attempt in range(1, retries + 1):
            try:
                start_time = time.time()
                response = requests.get(url, headers=headers, stream=True, verify=False, timeout=wait_timeout)
                for line in response.iter_lines():
                    if time.time() - start_time > wait_timeout:
                        print("⏱ Timeout while waiting for model response.")
                        return "پاسخی از مدل دریافت نشد"
                    if line and line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("msg") == "process_completed":
                                output_data = data.get("output", {}).get("data", [])
                                if output_data and isinstance(output_data[0], list) and len(output_data[0]) > 0:
                                    last_text = output_data[0][0][1][0]["text"]
                                    cleaned_text = re.sub(r'<summary>.*?</summary>', '', last_text)
                                    return cleaned_text
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSONDecodeError on attempt {attempt}: {e}")
                            print(f"خطا در پردازش پاسخ دریافتی، تلاش مجدد...")
                            # برای این خطا ادامه بده و تلاش کن
                            continue
                print("⚠️ پاسخ کامل دریافت نشد، تلاش مجدد...")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error on attempt {attempt}: {e}")
            # Backoff نمایی قبل از تلاش مجدد
            sleep_time = 2 ** attempt
            print(f"⏳ انتظار {sleep_time} ثانیه قبل از تلاش بعدی...")
            time.sleep(sleep_time)

        return "خطا در دریافت پاسخ از مدل پس از چند تلاش"

    def ask(self, prompt):
        self.send_request(prompt)
        print("⏳ در انتظار پاسخ از مدل...")
        time.sleep(5)  # منتظر بمون تا مدل شروع به تولید خروجی کنه
        return self.get_response()

