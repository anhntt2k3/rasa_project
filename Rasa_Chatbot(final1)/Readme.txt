# Instructions for Running Rasa Chatbot with API Chatbot Server

# First, install the required dependencies using:
pip install -r requirements.txt

# Next, train the Rasa model by running:
rasa train

# Open a new terminal window, navigate to the `deepseek_server` directory, and start the DeepSeek Server:
cd deepseek_server
uvicorn deepseek_server:app --host 0.0.0.0 --port 8000 --reload

# Open another terminal window and start the Rasa action server:
rasa run actions

# Then, open a new terminal and launch the Rasa shell for testing:
rasa shell

# (Optional) If you want to evaluate the chatbot, open a new terminal, navigate to `deepseek_server`, and run:
cd deepseek_server
uvicorn evaluate_chatbot_ragas:app --host 0.0.0.0 --port 8001 --reload

**Note:** Before running the evaluation, ensure that the evaluation-related code in `deepseek_server.py` is uncommented.

# To enable API access, open another terminal window and run:
rasa run -m models --enable-api --cors "*" --debug

# To update the dependency list, run:
pipreqs .


### Start and Stop Quickly
# To start all required services quickly, you can run:
run_rasa.bat

# To stop all running services, use:
stop_rasa.bat