#!/bin/bash

# This script starts the FastAPI server and the Streamlit UI for the Upbit Trader project.

# Activate the virtual environment
source venv/bin/activate

# Start the FastAPI server in the background
uvicorn server.api:app --host 127.0.0.1 --port 8000 --reload &
FASTAPI_PID=$!
echo "FastAPI server started with PID $FASTAPI_PID"

# Start the Streamlit UI in the background
streamlit run ui/streamlit_app.py &
STREAMLIT_PID=$!
echo "Streamlit UI started with PID $STREAMLIT_PID"

# Wait for user to terminate the script
echo "Services are running. Press [CTRL+C] to stop."
trap "kill $FASTAPI_PID $STREAMLIT_PID" SIGINT
wait
