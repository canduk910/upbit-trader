from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from server import config

app = FastAPI(title="Upbit Trader Runtime API")


class ConfigPayload(BaseModel):
    config: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    cfg = config._config
    return {"config": cfg}


@app.post("/config")
def post_config(payload: ConfigPayload):
    new_cfg = payload.config
    # 기본적인 검증: 반드시 strategy_name과 market이 있어야 함
    if not isinstance(new_cfg, dict) or 'strategy_name' not in new_cfg or 'market' not in new_cfg:
        raise HTTPException(status_code=400, detail="Invalid config payload. 'strategy_name' and 'market' required.")

    success = config.save_config(new_cfg)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    # 저장 후 재로딩
    config.reload_config()
    return {"status": "saved"}


@app.post("/reload")
def reload_config():
    config.reload_config()
    return {"status": "reloaded"}

