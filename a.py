# import requests

# deepseek_api_key = <Your API Key>
# url = "https://openrouter.ai/api/v1/chat/completions"

# headers = {
#     "Authorization": f"Bearer {deepseek_api_key}",
#     "Content-Type": "application/json"
# }

# data = {
#     "model": "deepseek/deepseek-r1-distill-llama-70b:free",
#     "messages": [{"role": "user", "content": "Xin chào!"}],
#     "temperature": 0.7
# }

# response = requests.post(url, headers=headers, json=data)

# if response.status_code == 200:
#     print("✅ API Key hợp lệ.")
#     print("🔹 Phản hồi từ DeepSeek:", response.json())
# else:
#     print(f"❌ API Key không hợp lệ. Mã lỗi: {response.status_code} - {response.text}")
