"""Run the FastAPI app for runtime configuration.

Usage:
    uvicorn server.api:app --host 127.0.0.1 --port 8000 --reload

(Or run this module directly for development.)
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host="127.0.0.1", port=8000, reload=True)

