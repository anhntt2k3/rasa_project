@echo off
echo 🔥 Dừng tất cả tiến trình liên quan đến Rasa và DeepSeek...

:: Dừng DeepSeek Server
taskkill /F /IM uvicorn.exe /T

:: Dừng Rasa Actions
taskkill /F /IM rasa.exe /T

:: Dừng Rasa Shell
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq Rasa Shell*"
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq Rasa Actions*"
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq DeepSeek Server*"

echo ✅ Tất cả tiến trình đã được dừng!
exit