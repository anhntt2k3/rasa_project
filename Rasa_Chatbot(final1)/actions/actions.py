import requests
from rasa_sdk import Action

class ActionAskDeepseek(Action):
    def name(self):
        return "action_ask_deepseek"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")
        print(f"📥 Câu hỏi nhận được từ người dùng: {user_message}")  # Debug

        url = "http://localhost:8000/query"
        payload = {"question": user_message}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            print(f"📡 Gửi API: {payload}")  # Debug 

            if response.status_code == 200:
                answer = response.json().get("answer", "Xin lỗi, tôi không có câu trả lời.")
            else:
                answer = f"Lỗi API: {response.json()}"

        except Exception as e:
            answer = f"Lỗi kết nối đến DeepSeek API: {str(e)}"

        dispatcher.utter_message(text=answer)
        return []
    