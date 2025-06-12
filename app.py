import streamlit as st
import json
import os
import tempfile
import time
from pathlib import Path
import threading
import queue

from patent_extractor import StreamingPatentExtractor

# ページ設定
st.set_page_config(
    page_title="特許PDF構造化ツール（ストリーミング対応）",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .json-display {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        background-color: #f5f5f5;
        height: 500px;
        overflow: auto;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .streaming-output {
        border: 1px solid #d0e0ff;
        border-radius: 5px;
        padding: 15px;
        background-color: #f8fafe;
        height: 400px;
        overflow-y: auto;
        font-family: monospace;
        white-space: pre-wrap;
        line-height: 1.4;
    }
    .stat-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center;
    }
    .blue-stat {
        background-color: #f0f5ff;
        border: 1px solid #d0e0ff;
    }
    .green-stat {
        background-color: #f0fff5;
        border: 1px solid #d0ffe0;
    }
    .orange-stat {
        background-color: #fff5f0;
        border: 1px solid #ffe0d0;
    }
    .title-area {
        text-align: center;
        margin-bottom: 2rem;
    }
    .progress-indicator {
        padding: 10px;
        border-radius: 5px;
        background-color: #e8f4f8;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    .streaming-status {
        font-weight: bold;
        color: #0066cc;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# モデルオプション（デフォルトの選択肢）
DEFAULT_MODEL_OPTIONS = {
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4-vision-preview"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
}

# セッション状態の初期化
if 'streaming_output' not in st.session_state:
    st.session_state.streaming_output = ""
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# 環境変数からAPIキーを取得する関数
def get_api_key_from_env(provider):
    if provider == "Google Gemini":
        return os.environ.get("GOOGLE_API_KEY", "")
    elif provider == "OpenAI":
        return os.environ.get("OPENAI_API_KEY", "")
    elif provider == "Anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    return ""

# プロバイダーからモデル名のプレフィックスを推定する関数
def get_model_prefix(provider):
    if provider == "Google Gemini":
        return "gemini-"
    elif provider == "OpenAI":
        return "gpt-"
    elif provider == "Anthropic":
        return "claude-"
    return ""

# モデル名が有効かチェックする関数
def is_valid_model_name(model_name, provider):
    """モデル名が指定されたプロバイダーに適しているかをチェック"""
    if not model_name:
        return False
    
    model_lower = model_name.lower()
    
    if provider == "Google Gemini":
        return "gemini" in model_lower
    elif provider == "OpenAI":
        return "gpt" in model_lower or "openai" in model_lower
    elif provider == "Anthropic":
        return "claude" in model_lower
    
    return True

# APIから利用可能なモデル一覧を取得する関数
@st.cache_data(ttl=3600)
def fetch_available_models(provider, api_key):
    """APIから利用可能なモデル一覧を取得"""
    try:
        if not api_key:
            return []
        
        if provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            models = genai.list_models()
            return [model.name.replace('models/', '') for model in models 
                   if 'generateContent' in model.supported_generation_methods]
        
        elif provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            model_names = [model.id for model in models.data 
                          if 'gpt' in model.id.lower() or 'vision' in model.id.lower()]
            return sorted(model_names, reverse=True)
        
        elif provider == "Anthropic":
            return [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022"
            ]
        
        return []
        
    except Exception as e:
        st.error(f"モデル一覧の取得に失敗しました: {str(e)}")
        return []

# モデル一覧を取得してキャッシュする関数
def get_models_with_cache(provider, api_key):
    """キャッシュを使用してモデル一覧を取得"""
    if not api_key:
        return DEFAULT_MODEL_OPTIONS.get(provider, [])
    
    try:
        available_models = fetch_available_models(provider, api_key)
        if available_models:
            return available_models
        else:
            return DEFAULT_MODEL_OPTIONS.get(provider, [])
    except:
        return DEFAULT_MODEL_OPTIONS.get(provider, [])

# JSONスキーマをロードする関数
def load_schema(file_path=None, file_content=None):
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file_content:
            return json.loads(file_content)
        return {}
    except Exception as e:
        st.error(f"JSONスキーマの読み込みエラー: {str(e)}")
        return {}

# PDFファイルを一時ファイルとして保存する関数
def save_upload_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"ファイル保存エラー: {str(e)}")
        return None

# ストリーミング処理用関数
def stream_processing_thread(pdf_path, model_name, api_key, schema, prompt, temperature, max_tokens, output_queue):
    """別スレッドでストリーミング処理を実行"""
    try:
        extractor = StreamingPatentExtractor(
            model_name=model_name,
            api_key=api_key,
            json_schema=schema,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        def stream_callback(chunk):
            output_queue.put(('chunk', chunk))
        
        full_output = ""
        for chunk in extractor.process_patent_pdf_stream(pdf_path, stream_callback):
            full_output += chunk
        
        # 最終結果をJSONとして解析
        try:
            final_json = extractor._extract_json_from_text(full_output)
            output_queue.put(('complete', final_json))
        except Exception as e:
            output_queue.put(('complete', {"error": f"JSON解析エラー: {str(e)}", "raw_output": full_output}))
            
    except Exception as e:
        output_queue.put(('error', str(e)))

# リセット関数
def reset_session_state():
    """セッション状態をリセット"""
    st.session_state.streaming_output = ""
    st.session_state.is_processing = False
    st.session_state.processing_complete = False
    st.session_state.final_result = None
    st.session_state.chunk_count = 0
    st.session_state.start_time = None

# タイトルと説明
st.markdown('<div class="title-area">', unsafe_allow_html=True)
st.title("🧩 特許PDF構造化ツール（ストリーミング対応）")
st.markdown("""
特許PDFからマルチモーダル生成AIを使用して構造化JSONを**リアルタイムストリーミング**で抽出します。
Gemini、GPT、Claudeなどの最新モデルを利用して特許データを解析できます。
""")
st.markdown('</div>', unsafe_allow_html=True)

# サイドバー - 設定エリア
with st.sidebar:
    st.header("⚙️ 設定")
    
    # APIプロバイダーとモデル選択
    provider = st.selectbox(
        "AIプロバイダー（Google Geminiを推奨）",
        options=list(DEFAULT_MODEL_OPTIONS.keys())
    )
    
    # APIキー入力
    api_key = st.text_input(
        f"{provider} APIキー",
        value=get_api_key_from_env(provider),
        type="password",
        help="APIキーを入力してください。環境変数から自動的に取得することも可能です。"
    )
    
    # モデル選択方法のラジオボタン
    model_input_type = st.radio(
        "モデル選択方法",
        options=["利用可能なモデルから選択", "カスタムモデル名を入力"],
        horizontal=True
    )
    
    if model_input_type == "利用可能なモデルから選択":
        with st.spinner("利用可能なモデルを取得中..."):
            available_models = get_models_with_cache(provider, api_key)
        
        if available_models:
            if api_key:
                st.info(f"✅ {len(available_models)}個の利用可能なモデルを取得しました")
            else:
                st.info(f"ℹ️ デフォルトモデル一覧を表示中（APIキーを入力すると最新一覧を取得）")
            
            model_name = st.selectbox(
                "モデル",
                options=available_models,
                help="APIから取得した利用可能なモデル一覧です"
            )
            
            if st.button("🔄 モデル一覧を更新", help="最新のモデル一覧を再取得します"):
                st.cache_data.clear()
                st.rerun()
                
        else:
            st.error("利用可能なモデルを取得できませんでした。APIキーを確認してください。")
            model_name = ""
            
    else:
        model_placeholder = get_model_prefix(provider) + "model-name"
        model_name = st.text_input(
            "モデル名",
            placeholder=model_placeholder,
            help=f"{provider}の任意のモデル名を入力してください（例: {model_placeholder}）"
        )
        
        if model_name and not is_valid_model_name(model_name, provider):
            st.warning(f"⚠️ 入力されたモデル名「{model_name}」は{provider}のモデルではない可能性があります。")
        
        with st.expander("💡 モデル名のヒント"):
            if provider == "Google Gemini":
                st.markdown("""
                **Google Geminiモデル例:**
                - `gemini-1.5-pro` - 高性能モデル
                - `gemini-1.5-flash` - 高速モデル
                - `gemini-2.0-flash-exp` - 実験的モデル
                """)
            elif provider == "OpenAI":
                st.markdown("""
                **OpenAIモデル例:**
                - `gpt-4o` - 最新のマルチモーダルモデル
                - `gpt-4-vision-preview` - ビジョン対応モデル
                - `gpt-4-turbo` - 高速モデル
                """)
            elif provider == "Anthropic":
                st.markdown("""
                **Anthropicモデル例:**
                - `claude-3-opus-20240229` - 最高性能モデル
                - `claude-3-sonnet-20240229` - バランス型モデル
                - `claude-3-haiku-20240307` - 高速モデル
                - `claude-3-5-sonnet-20241022` - 最新Sonnetモデル
                """)
    
    if model_input_type == "カスタムモデル名を入力" and not model_name:
        st.error("モデル名を入力してください")
    
    if model_name:
        st.success(f"選択されたモデル: **{model_name}**")
    
    # スキーマタイプの選択
    schema_type = st.radio(
        "JSONスキーマ",
        options=["デフォルトスキーマを使用", "カスタムスキーマをアップロード", "スキーマを直接入力"],
        index=0
    )
    
    schema = {}
    if schema_type == "カスタムスキーマをアップロード":
        uploaded_schema = st.file_uploader(
            "JSONスキーマファイル",
            type=["json"],
            help="JSONスキーマファイルをアップロードしてください"
        )
        if uploaded_schema:
            schema_content = uploaded_schema.getvalue().decode("utf-8")
            schema = load_schema(file_content=schema_content)
            if schema:
                st.success("スキーマを読み込みました")
    
    elif schema_type == "スキーマを直接入力":
        schema_text = st.text_area(
            "JSONスキーマを入力",
            height=200,
            help="JSONスキーマを直接入力してください"
        )
        if schema_text:
            try:
                schema = json.loads(schema_text)
                st.success("スキーマの形式が正しいです")
            except json.JSONDecodeError:
                st.error("JSONの形式が正しくありません")
    
    else:
        default_schema_path = "default_schema.json"
        if os.path.exists(default_schema_path):
            schema = load_schema(file_path=default_schema_path)
            st.success("デフォルトスキーマを読み込みました")
        else:
            st.info("デフォルトスキーマが見つからないため、空のスキーマを使用します")
    
    # 詳細設定
    with st.expander("詳細設定", expanded=False):
        custom_prompt = st.text_area(
            "カスタムプロンプト",
            value="",
            height=100,
            help="生成AIへのカスタム指示を入力できます"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="高い値: より多様な出力、低い値: より確定的な出力"
        )
        
        max_tokens = st.number_input(
            "最大トークン数",
            min_value=1024,
            max_value=65535,
            value=32768,
            step=1024,
            help="生成AIが出力できる最大トークン数"
        )
    
    # スキーマの詳細表示
    if schema and st.checkbox("スキーマ詳細を表示"):
        st.json(schema)
    
    # リセットボタン
    if st.button("🔄 リセット", help="処理状態をリセットします"):
        reset_session_state()
        st.rerun()

# メインエリア - 2列レイアウト
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 入力")
    
    # PDF アップロード
    uploaded_pdf = st.file_uploader(
        "特許PDFをアップロード",
        type=["pdf"],
        help="処理する特許PDFファイルをアップロードしてください"
    )
    
    if uploaded_pdf:
        st.success(f"ファイル名: {uploaded_pdf.name}")
        
        # PDFの表示
        with st.expander("PDFプレビュー", expanded=False):
            try:
                import base64
                pdf_base64 = base64.b64encode(uploaded_pdf.getvalue()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="500" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            except:
                st.info("PDFプレビューを表示できませんでした")
    
    # 処理ボタン
    process_button = st.button(
        "🚀 ストリーミング処理開始",
        disabled=not (uploaded_pdf and api_key and model_name) or st.session_state.is_processing,
        help="特許PDFをストリーミング処理して構造化JSONを生成します"
    )

with col2:
    st.header("📥 ストリーミング出力")
    
    # 処理状況の表示
    if st.session_state.is_processing or st.session_state.processing_complete:
        # 統計情報
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            elapsed_time = (time.time() - st.session_state.start_time) if st.session_state.start_time else 0
            st.markdown(f'<div class="stat-box blue-stat"><h4>{elapsed_time:.1f}s</h4>経過時間</div>', unsafe_allow_html=True)
        with col_stats2:
            st.markdown(f'<div class="stat-box green-stat"><h4>{st.session_state.chunk_count}</h4>受信チャンク</div>', unsafe_allow_html=True)
        with col_stats3:
            status_text = "完了" if st.session_state.processing_complete else "処理中"
            status_color = "green-stat" if st.session_state.processing_complete else "orange-stat"
            st.markdown(f'<div class="stat-box {status_color}"><h4>{status_text}</h4>ステータス</div>', unsafe_allow_html=True)
    
    # ストリーミング出力エリア
    if st.session_state.is_processing or st.session_state.streaming_output:
        st.markdown("### リアルタイムストリーミング出力")
        streaming_container = st.empty()
        streaming_container.markdown(f'<div class="streaming-output">{st.session_state.streaming_output}</div>', unsafe_allow_html=True)
    
    # 最終結果の表示
    if st.session_state.processing_complete and st.session_state.final_result:
        st.markdown("### 最終JSON結果")
        st.json(st.session_state.final_result)
        
        # ダウンロードボタン
        if uploaded_pdf:
            json_str = json.dumps(st.session_state.final_result, ensure_ascii=False, indent=2)
            output_filename = f"{Path(uploaded_pdf.name).stem}_streaming.json"
            
            st.download_button(
                label="📥 JSONをダウンロード",
                data=json_str.encode("utf-8"),
                file_name=output_filename,
                mime="application/json",
                help="抽出された構造化データをJSONファイルとしてダウンロードします"
            )
    
    # プレースホルダー
    if not st.session_state.is_processing and not st.session_state.processing_complete:
        st.info("PDFをアップロードして「ストリーミング処理開始」ボタンをクリックすると、ここにリアルタイムで抽出結果が表示されます")
        
        with st.expander("ストリーミング出力例"):
            st.markdown("""
            ```json
            {
              "publicationIdentifier": "WO2020123456A1",
              "FrontPage": {
                "title": "AI駆動特許データ抽出システム",
                "PublicationData": {
                  "PublicationNumber": "WO2020123456A1",
                  "PublicationDate": "2020-06-15"
                }
              }
            }
            ```
            
            **ストリーミングの特徴:**
            - ⚡ リアルタイム出力表示
            - 🔄 途中経過が見える
            - ⏱️ 即座に結果確認
            - 🛑 途中停止可能
            """)

# 処理開始
if process_button and uploaded_pdf and api_key and model_name and not st.session_state.is_processing:
    # セッション状態をリセット
    reset_session_state()
    
    # 処理状態を開始に設定
    st.session_state.is_processing = True
    st.session_state.start_time = time.time()
    
    # PDFを一時ファイルとして保存
    pdf_path = save_upload_file(uploaded_pdf)
    
    if pdf_path:
        # キューを作成してストリーミング処理を開始
        output_queue = queue.Queue()
        
        # バックグラウンドスレッドで処理開始
        thread = threading.Thread(
            target=stream_processing_thread,
            args=(
                pdf_path,
                model_name,
                api_key,
                schema,
                custom_prompt if custom_prompt else None,
                temperature,
                max_tokens,
                output_queue
            )
        )
        thread.daemon = True
        thread.start()
        
        # ストリーミング出力の監視
        while st.session_state.is_processing:
            try:
                # ノンブロッキングでキューからデータを取得
                msg_type, data = output_queue.get(timeout=0.1)
                
                if msg_type == 'chunk':
                    st.session_state.streaming_output += data
                    st.session_state.chunk_count += 1
                    
                elif msg_type == 'complete':
                    st.session_state.final_result = data
                    st.session_state.processing_complete = True
                    st.session_state.is_processing = False
                    
                elif msg_type == 'error':
                    st.error(f"処理エラー: {data}")
                    st.session_state.is_processing = False
                
                # UIを更新
                st.rerun()
                
            except queue.Empty:
                # タイムアウト - 処理継続中
                time.sleep(0.1)
                continue
            except Exception as e:
                st.error(f"ストリーミングエラー: {str(e)}")
                st.session_state.is_processing = False
                break
        
        # 一時ファイルの削除
        try:
            os.remove(pdf_path)
        except:
            pass

# 自動リフレッシュ（処理中のみ）
if st.session_state.is_processing:
    time.sleep(1)
    st.rerun()

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    特許PDF構造化ツール（ストリーミング対応） - マルチモーダル生成AIを使用してリアルタイムで特許文書から構造化情報を抽出<br>
    <small>Powered by StreamingPatentExtractor library Copyright (c) 2025 Pyxist Co.,Ltd</small>
</div>
""", unsafe_allow_html=True)
