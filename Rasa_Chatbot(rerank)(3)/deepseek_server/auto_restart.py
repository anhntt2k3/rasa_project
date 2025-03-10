# import os
# import subprocess
# import time

# # Thiết lập giới hạn số lượt chat tối đa
# MAX_TURNS = 50
# turns_count = 0
# chat_locked = False

# def restart_services():
#     global turns_count, chat_locked
#     print("\n🔄 Đang khởi động lại DeepSeek Server và Rasa...")

#     # Đóng các tiến trình cũ (tùy theo hệ điều hành)
#     try:
#         if os.name == "posix":  # Linux/WSL
#             os.system("pkill -f deepseek_server")
#             os.system("pkill -f 'rasa run actions'")
#             os.system("pkill -f 'rasa shell'")
#         elif os.name == "nt":  # Windows
#             os.system("taskkill /F /IM deepseek_server.exe")
#             os.system("taskkill /F /IM rasa.exe")
#     except Exception as e:
#         print(f"⚠️ Lỗi khi dừng tiến trình: {e}")

#     # Đợi vài giây để chắc chắn tiến trình đã tắt
#     time.sleep(3)

#     # Khởi động lại các tiến trình
#     try:
#         subprocess.Popen("deepseek_server", shell=True)
#         subprocess.Popen("rasa run actions", shell=True)
#         subprocess.Popen("rasa shell", shell=True)
#     except Exception as e:
#         print(f"⚠️ Lỗi khi khởi động lại tiến trình: {e}")

#     # Reset bộ đếm
#     turns_count = 0
#     chat_locked = False
#     print("✅ Đã reset thành công DeepSeek Server và Rasa!")

# def check_and_lock():
#     global turns_count, chat_locked
#     if turns_count >= MAX_TURNS:
#         chat_locked = True
#         print("🚫 Đạt giới hạn hội thoại. Chatbot sẽ ngừng nhận tin nhắn mới cho đến khi bạn thoát Rasa shell hoặc gửi /stop.")

# def manual_reset():
#     global turns_count, chat_locked
#     print("🔄 Nhận lệnh reset. Đang khởi động lại hệ thống...")
#     restart_services()

# # Hàm cập nhật số lượt chat, gọi từ API chính
# def update_turns():
#     global turns_count
#     if not chat_locked:
#         turns_count += 1
#         check_and_lock()
