import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from server.logger import log

# --- 1. .env 파일에서 민감 정보 로드 ---
# server 디렉토리 기준으로 .env 파일 경로 설정
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    log.info(".env file loaded successfully.")
else:
    log.warning("server/.env file not found. Please create one with your API keys.")

# 환경 변수에서 API 키 가져오기 (따옴표/공백 제거)
def _read_env(key: str) -> str | None:
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    return val or None

UPBIT_ACCESS_KEY = _read_env("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = _read_env("UPBIT_SECRET_KEY")
OPENAI_API_KEY = _read_env("OPENAI_API_KEY")
GEMINI_API_KEY = _read_env("GEMINI_API_KEY")

# --- 2. runtime/config.json 파일에서 설정 파라미터 로드 ---
def _get_runtime_config_path():
    project_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(project_root, 'runtime', 'config.json')


def _migrate_config_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy config keys to the current schema without mutating the original dict."""
    normalized = json.loads(json.dumps(cfg)) if isinstance(cfg, dict) else {}
    strategy_params = normalized.setdefault('strategy_params', {})
    vb_params = strategy_params.setdefault('VolatilityBreakout', {})
    legacy = normalized.pop('vb_target_vol_pct', None)
    if 'target_vol_pct' not in vb_params and legacy is not None:
        try:
            vb_params['target_vol_pct'] = float(legacy)
        except (TypeError, ValueError):
            vb_params['target_vol_pct'] = 30.0
    if 'target_vol_pct' not in vb_params:
        vb_params['target_vol_pct'] = 30.0
    ai_ensemble = normalized.setdefault('ai_ensemble', {}) if isinstance(normalized.get('ai_ensemble'), dict) else {}
    base_strategy = str(ai_ensemble.get('strategy', 'UNANIMOUS')).upper()
    buy_strategy = str(ai_ensemble.get('buy_strategy', '')).upper()
    sell_strategy = str(ai_ensemble.get('sell_strategy', '')).upper()
    if not buy_strategy:
        buy_strategy = base_strategy
    if not sell_strategy:
        sell_strategy = base_strategy
    ai_ensemble['buy_strategy'] = buy_strategy or 'UNANIMOUS'
    ai_ensemble['sell_strategy'] = sell_strategy or 'UNANIMOUS'
    try:
        ai_ensemble['average_threshold'] = float(ai_ensemble.get('average_threshold', 0.5))
    except (TypeError, ValueError):
        ai_ensemble['average_threshold'] = 0.5
    normalized['ai_ensemble'] = ai_ensemble
    return normalized


def load_config() -> Dict[str, Any]:
    """config.json 파일에서 설정을 로드하여 반환합니다."""
    config_path = _get_runtime_config_path()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        log.info("runtime/config.json file loaded successfully.")
        return _migrate_config_schema(config_data)
    except FileNotFoundError:
        log.error("runtime/config.json not found! Please create a configuration file in runtime directory.")
        return {}
    except json.JSONDecodeError:
        log.error("Error decoding runtime/config.json. Please check for syntax errors.")
        return {}


def save_config(new_config: Dict[str, Any]) -> bool:
    """config.json 파일에 설정을 저장합니다. 안전을 위해 기존 파일을 백업합니다.

    :param new_config: 저장할 설정 딕셔너리
    :return: 저장 성공 여부
    """
    config_path = _get_runtime_config_path()
    backup_path = config_path + '.bak'
    try:
        # 기존 파일 백업
        if os.path.exists(config_path):
            os.replace(config_path, backup_path)
            log.info(f"Existing config.json backed up to {backup_path}")

        normalized = _migrate_config_schema(new_config)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        log.info("New configuration saved to runtime/config.json")
        return True
    except Exception as e:
        log.error(f"Failed to save runtime/config.json: {e}")
        # 백업 복원 시도
        if os.path.exists(backup_path):
            os.replace(backup_path, config_path)
            log.info("Restored original config.json from backup due to failure.")
        return False


# 즉시 로드하여 전역 설정 변수로 노출
_config = load_config()


def _sync_globals_from_config(cfg: Dict[str, Any]):
    """로드된 `_config` 딕셔너리의 값을 모듈 전역 변수로 동기화합니다.
    이 함수는 `reload_config()` 호출 시 기존 전역 변수들이 최신 값으로 갱신되도록 보장합니다.
    """
    global STRATEGY_NAME, MARKET, TIMEFRAME, CANDLE_COUNT, LOOP_INTERVAL_SEC
    global MIN_ORDER_AMOUNT, TRADE_AMOUNT_KRW
    global USE_KELLY_CRITERION, KELLY_WIN_RATE, KELLY_PAYOFF_RATIO, KELLY_FRACTION
    global RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT
    global VB_K_VALUE, VB_TARGET_VOL_PCT, DM_WINDOW
    global ENSEMBLE_STRATEGY, OPENAI_MODEL, GEMINI_MODEL
    global BOT_ENABLED, BOT_INTERVAL_SEC, BOT_SELL_COOLDOWN_SEC
    global AI_ENS_BS, AI_ENS_SS, AI_ENS_AT

    STRATEGY_NAME = cfg.get("strategy_name", "RSI")
    MARKET = cfg.get("market", "KRW-BTC")
    TIMEFRAME = cfg.get("timeframe", "minute5")
    CANDLE_COUNT = cfg.get("candle_count", 200)
    LOOP_INTERVAL_SEC = cfg.get("loop_interval_sec", 5)

    _order_settings = cfg.get("order_settings", {})
    MIN_ORDER_AMOUNT = _order_settings.get("min_order_amount", 5500)
    TRADE_AMOUNT_KRW = _order_settings.get("trade_amount_krw", 6000)

    USE_KELLY_CRITERION = cfg.get("use_kelly_criterion", False)
    _kelly_settings = cfg.get("kelly_criterion", {})
    KELLY_WIN_RATE = _kelly_settings.get("win_rate", 0.5)
    KELLY_PAYOFF_RATIO = _kelly_settings.get("payoff_ratio", 1.0)
    KELLY_FRACTION = _kelly_settings.get("fraction", 0.5)

    _strategy_params = cfg.get("strategy_params", {})
    _rsi_params = _strategy_params.get("RSI", {})
    RSI_PERIOD = _rsi_params.get("period", 14)
    RSI_OVERSOLD = _rsi_params.get("oversold", 30)
    RSI_OVERBOUGHT = _rsi_params.get("overbought", 70)

    _vb_params = _strategy_params.get("VolatilityBreakout", {})
    VB_K_VALUE = _vb_params.get("k_value", 0.5)
    VB_TARGET_VOL_PCT = _vb_params.get("target_vol_pct", 30.0)

    _dm_params = _strategy_params.get("DualMomentum", {})
    DM_WINDOW = _dm_params.get("window", 12)

    _ai_ensemble_settings = cfg.get("ai_ensemble", {})
    ENSEMBLE_STRATEGY = _ai_ensemble_settings.get("strategy", "UNANIMOUS")
    OPENAI_MODEL = _ai_ensemble_settings.get("openai_model", "gpt-5.1-nano")
    GEMINI_MODEL = _ai_ensemble_settings.get("gemini_model", "gemini-2.5-flash")
    AI_ENS_BS = _ai_ensemble_settings.get("buy_strategy", ENSEMBLE_STRATEGY).upper()
    AI_ENS_SS = _ai_ensemble_settings.get("sell_strategy", ENSEMBLE_STRATEGY).upper()
    AI_ENS_AT = _ai_ensemble_settings.get("average_threshold", 0.5)

    bot_enabled = cfg.get("bot_enabled")
    if bot_enabled is None:
        bot_enabled = True
    BOT_ENABLED = bool(bot_enabled)
    BOT_INTERVAL_SEC = float(cfg.get("bot_interval_sec", 5.0))
    BOT_SELL_COOLDOWN_SEC = float(cfg.get("bot_sell_cooldown_sec", 120.0))


# 초기 로드 이후 전역 변수 동기화
_sync_globals_from_config(_config)

# expose some convenience getters
def get_setting(key: str, default=None):
    value = _config.get(key, default)
    if value is None:
        value = os.environ.get(key, default)
    return value


def reload_config():
    global _config
    _config = load_config()
    # 로드된 config로 전역 변수 동기화
    _sync_globals_from_config(_config)
    log.info("Configuration reloaded.")
    return _config


# --- 기존 전역 변수화 (하위 모듈 호환성 유지) ---
STRATEGY_NAME = _config.get("strategy_name", "RSI")
MARKET = _config.get("market", "KRW-BTC")
TIMEFRAME = _config.get("timeframe", "minute5")
CANDLE_COUNT = _config.get("candle_count", 200)
LOOP_INTERVAL_SEC = _config.get("loop_interval_sec", 5)

_order_settings = _config.get("order_settings", {})
MIN_ORDER_AMOUNT = _order_settings.get("min_order_amount", 5500)
TRADE_AMOUNT_KRW = _order_settings.get("trade_amount_krw", 6000)

USE_KELLY_CRITERION = _config.get("use_kelly_criterion", False)
_kelly_settings = _config.get("kelly_criterion", {})
KELLY_WIN_RATE = _kelly_settings.get("win_rate", 0.5)
KELLY_PAYOFF_RATIO = _kelly_settings.get("payoff_ratio", 1.0)
KELLY_FRACTION = _kelly_settings.get("fraction", 0.5)

_strategy_params = _config.get("strategy_params", {})
_rsi_params = _strategy_params.get("RSI", {})
RSI_PERIOD = _rsi_params.get("period", 14)
RSI_OVERSOLD = _rsi_params.get("oversold", 30)
RSI_OVERBOUGHT = _rsi_params.get("overbought", 70)

_vb_params = _config.get("strategy_params", {}).get("VolatilityBreakout", {})
VB_K_VALUE = _vb_params.get("k_value", 0.5)
VB_TARGET_VOL_PCT = _vb_params.get("target_vol_pct", 30.0)

_dm_params = _strategy_params.get("DualMomentum", {})
DM_WINDOW = _dm_params.get("window", 12)

_ai_ensemble_settings = _config.get("ai_ensemble", {})
ENSEMBLE_STRATEGY = _ai_ensemble_settings.get("strategy", "UNANIMOUS")
OPENAI_MODEL = _ai_ensemble_settings.get("openai_model", "gpt-5.1-nano")
GEMINI_MODEL = _ai_ensemble_settings.get("gemini_model", "gemini-2.5-flash")
AI_ENS_BS = _ai_ensemble_settings.get("buy_strategy", ENSEMBLE_STRATEGY).upper()
AI_ENS_SS = _ai_ensemble_settings.get("sell_strategy", ENSEMBLE_STRATEGY).upper()
AI_ENS_AT = _ai_ensemble_settings.get("average_threshold", 0.5)

BOT_ENABLED = _config.get("bot_enabled", True)
BOT_INTERVAL_SEC = float(_config.get("bot_interval_sec", 5.0))
BOT_SELL_COOLDOWN_SEC = float(_config.get("bot_sell_cooldown_sec", 120.0))

log.info(f"Configuration loaded: Strategy='{STRATEGY_NAME}', Market='{MARKET}'")
log.debug(f"Volatility breakout target volatility pct set to {VB_TARGET_VOL_PCT}")


def update_bot_control(bot_enabled: Optional[bool] = None, bot_interval_sec: Optional[float] = None, bot_sell_cooldown_sec: Optional[float] = None):
    """Update bot control values and persist them."""
    global _config
    updated = False
    if bot_enabled is not None:
        _config['bot_enabled'] = bool(bot_enabled)
        updated = True
    if bot_interval_sec is not None:
        _config['bot_interval_sec'] = float(bot_interval_sec)
        updated = True
    if bot_sell_cooldown_sec is not None:
        _config['bot_sell_cooldown_sec'] = float(bot_sell_cooldown_sec)
        updated = True
    if updated:
        save_config(_config)
        reload_config()
    return updated
