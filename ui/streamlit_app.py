import streamlit as st
import json
import requests
from pathlib import Path

RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / 'runtime' / 'config.json'
API_BASE = st.sidebar.text_input('API Base URL', 'http://127.0.0.1:8000')

st.title('Upbit Trader - Runtime Config Editor (Streamlit)')
st.markdown('Edit runtime configuration (runtime/config.json). After saving, the server will reload configuration.')

if not RUNTIME_CONFIG.exists():
    st.error('Runtime config not found: {}'.format(RUNTIME_CONFIG))
else:
    try:
        with open(RUNTIME_CONFIG, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        st.error('Failed to load runtime config: {}'.format(e))
        cfg = {}

    text = st.text_area('runtime/config.json', json.dumps(cfg, indent=2, ensure_ascii=False), height=480)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Save locally'):
            try:
                new_cfg = json.loads(text)
                with open(RUNTIME_CONFIG, 'w', encoding='utf-8') as f:
                    json.dump(new_cfg, f, ensure_ascii=False, indent=2)
                st.success('Saved to runtime/config.json')
            except json.JSONDecodeError as e:
                st.error('Invalid JSON format: {}'.format(e))
            except Exception as e:
                st.error('Failed to save locally: {}'.format(e))
    with col2:
        if st.button('Save & Push to Server (POST /config)'):
            try:
                new_cfg = json.loads(text)
            except json.JSONDecodeError as e:
                st.error('Invalid JSON: {}'.format(e))
            else:
                try:
                    resp = requests.post("{}/config".format(API_BASE), json={'config': new_cfg}, timeout=10)
                    if resp.status_code == 200:
                        st.success('Saved and server reloaded')
                    else:
                        st.error('Server responded with {}: {}'.format(resp.status_code, resp.text))
                except requests.RequestException as e:
                    st.error('Failed to call server API: {}'.format(e))

    st.markdown('---')
    if st.button('Reload config on server (POST /reload)'):
        try:
            resp = requests.post("{}/reload".format(API_BASE), timeout=5)
            if resp.status_code == 200:
                st.success('Server reloaded configuration')
            else:
                st.error('Server responded with {}: {}'.format(resp.status_code, resp.text))
        except requests.RequestException as e:
            st.error('Failed to call server API: {}'.format(e))

    st.markdown('**Notes**')
    st.markdown('- This tool edits the runtime JSON used by the bot. Ensure server is running and API is reachable.')
    st.markdown('- No authentication is implemented; restrict access to this UI in production.')
