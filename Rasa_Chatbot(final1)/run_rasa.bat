@echo off
echo 🚀 Khởi động DeepSeek Server...
cd deepseek_server
start "DeepSeek Server" cmd /k uvicorn deepseek_server:app --host 0.0.0.0 --port 8000 --reload
cd..
timeout /t 3

echo 🚀 Khởi động Rasa Actions...
start "Rasa Actions" cmd /k rasa run actions
timeout /t 3

echo 🚀 Khởi động Rasa Shell...
start "Rasa Shell" cmd /k rasa shell
timeout /t 3

@REM echo 👀 Giám sát hệ thống (Auto Restart)...
@REM python deepseek_server/auto_restart.py

echo ✅ Hoàn thành!
pause
