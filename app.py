import streamlit as st
import json
import os
import tempfile
import time
from pathlib import Path

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
    .stat-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .blue-stat {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #2196f3;
    }
    .green-stat {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 1px solid #4caf50;
    }
    .orange-stat {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
        border: 1px solid #ff9800;
    }
    .title-area {
        text-align: center;
        margin-bottom: 2rem;
    }
    .processing-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        background-color: #e3f2fd;
        border-radius: 8px;
        margin: 10px 0;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# モデルオプション
DEFAULT_MODEL_OPTIONS = {
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4-vision-preview"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
}

# セッション状態の初期化
def init_session_state():
    """セッション状態を初期化"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'final_result' not in st.session_state:
        st.session_state.final_result = None
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'chunk_count' not in st.session_state:
        st.session_state.chunk_count = 0

init_session_state()

# 環境変数からAPIキーを取得
def get_api_key_from_env(provider):
    """環境変数からAPIキーを取得"""
    if provider == "Google Gemini":
        return os.environ.get("GOOGLE_API_KEY", "")
    elif provider == "OpenAI":
        return os.environ.get("OPENAI_API_KEY", "")
    elif provider == "Anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    return ""

# プロバイダーからモデル名のプレフィックスを推定
def get_model_prefix(provider):
    """プロバイダーからモデル名のプレフィックスを取得"""
    if provider == "Google Gemini":
        return "gemini-"
    elif provider == "OpenAI":
        return "gpt-"
    elif provider == "Anthropic":
        return "claude-"
    return ""

# モデル名が有効かチェック
def is_valid_model_name(model_name, provider):
    """モデル名の妥当性をチェック"""
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

# APIから利用可能なモデル一覧を取得
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
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307",
                "claude-3-5-haiku-20241022"
            ]
        
        return []
        
    except Exception as e:
        st.error(f"モデル一覧の取得に失敗しました: {str(e)}")
        return []

# モデル一覧を取得してキャッシュ
def get_models_with_cache(provider, api_key):
    """キャッシュを使用してモデル一覧を取得"""
    if not api_key:
        return DEFAULT_MODEL_OPTIONS.get(provider, [])
    
    try:
        available_models = fetch_available_models(provider, api_key)
        return available_models if available_models else DEFAULT_MODEL_OPTIONS.get(provider, [])
    except:
        return DEFAULT_MODEL_OPTIONS.get(provider, [])

# JSONスキーマをロード
def load_schema(file_path=None, file_content=None):
    """JSONスキーマをロード"""
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

# PDFファイルを一時ファイルとして保存
def save_upload_file(uploaded_file):
    """PDFファイルを一時ファイルとして保存"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"ファイル保存エラー: {str(e)}")
        return None

# セッション状態をリセット
def reset_session_state():
    """セッション状態をリセット"""
    st.session_state.messages = []
    st.session_state.is_processing = False
    st.session_state.processing_complete = False
    st.session_state.final_result = None
    st.session_state.start_time = None
    st.session_state.chunk_count = 0

# タイトルと説明
st.markdown('<div class="title-area">', unsafe_allow_html=True)
st.title("🧩 特許PDF構造化ツール（ストリーミング対応）")
st.markdown("""
特許PDFからマルチモーダル生成AIを使用して構造化JSONを**リアルタイムストリーミング**で抽出します。
Streamlitのチャット機能を使用してタイプライター効果でAIの処理過程を可視化します。
""")
st.markdown('</div>', unsafe_allow_html=True)

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # プロバイダー選択
    provider = st.selectbox(
        "AIプロバイダー（Google Geminiを推奨）",
        options=list(DEFAULT_MODEL_OPTIONS.keys())
    )
    
    # APIキー入力
    api_key = st.text_input(
        f"{provider} APIキー",
        value=get_api_key_from_env(provider),
        type="password",
        help="APIキーを入力してください"
    )
    
    # モデル選択
    model_input_type = st.radio(
        "モデル選択方法",
        options=["利用可能なモデルから選択", "カスタムモデル名を入力"],
        horizontal=True
    )
    
    if model_input_type == "利用可能なモデルから選択":
        with st.spinner("モデル一覧取得中..."):
            available_models = get_models_with_cache(provider, api_key)
        
        if available_models:
            if api_key:
                st.info(f"✅ {len(available_models)}個のモデルを取得")
            else:
                st.info("ℹ️ デフォルトモデル一覧")
            
            model_name = st.selectbox("モデル", options=available_models)
            
            if st.button("🔄 モデル一覧更新"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.error("モデル一覧を取得できませんでした")
            model_name = ""
    else:
        model_placeholder = get_model_prefix(provider) + "model-name"
        model_name = st.text_input(
            "モデル名",
            placeholder=model_placeholder,
            help=f"{provider}のモデル名を入力"
        )
        
        if model_name and not is_valid_model_name(model_name, provider):
            st.warning(f"⚠️ {model_name}は{provider}のモデルではない可能性があります")
    
    if model_name:
        st.success(f"選択モデル: **{model_name}**")
    
    # スキーマ設定
    schema_type = st.radio(
        "JSONスキーマ",
        options=["デフォルト", "カスタムファイル", "直接入力"],
        index=0
    )
    
    schema = {}
    if schema_type == "カスタムファイル":
        uploaded_schema = st.file_uploader("JSONスキーマファイル", type=["json"])
        if uploaded_schema:
            schema_content = uploaded_schema.getvalue().decode("utf-8")
            schema = load_schema(file_content=schema_content)
            if schema:
                st.success("スキーマ読み込み完了")
    
    elif schema_type == "直接入力":
        schema_text = st.text_area("JSONスキーマを入力", height=200)
        if schema_text:
            try:
                schema = json.loads(schema_text)
                st.success("スキーマ形式OK")
            except json.JSONDecodeError:
                st.error("JSON形式エラー")
    
    else:
        default_schema_path = "default_schema.json"
        if os.path.exists(default_schema_path):
            schema = load_schema(file_path=default_schema_path)
            st.success("デフォルトスキーマ使用")
        else:
            st.info("空のスキーマを使用")
    
    # 詳細設定
    with st.expander("詳細設定"):
        custom_prompt = st.text_area("カスタムプロンプト", height=100)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.number_input("最大トークン数", 1024, 65535, 32768, 1024)
    
    # リセットボタン
    if st.button("🔄 リセット"):
        reset_session_state()
        st.rerun()

# メインエリア
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 入力")
    
    uploaded_pdf = st.file_uploader(
        "特許PDFをアップロード",
        type=["pdf"],
        help="処理する特許PDFファイルをアップロード"
    )
    
    if uploaded_pdf:
        st.success(f"ファイル: {uploaded_pdf.name}")
        
        with st.expander("PDFプレビュー"):
            try:
                import base64
                pdf_base64 = base64.b64encode(uploaded_pdf.getvalue()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="500"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            except:
                st.info("プレビュー表示不可")
    
    # 処理ボタン
    process_disabled = not (uploaded_pdf and api_key and model_name) or st.session_state.is_processing
    
    if st.button(
        "🚀 ストリーミング処理開始",
        disabled=process_disabled,
        help="特許PDFをStreamlitチャット形式でストリーミング処理"
    ):
        if uploaded_pdf and api_key and model_name:
            # セッション状態をリセット
            reset_session_state()
            
            # ユーザーメッセージを追加
            st.session_state.messages.append({
                "role": "user", 
                "content": f"特許PDF「{uploaded_pdf.name}」を分析してください"
            })
            
            # 処理状態を開始に設定
            st.session_state.is_processing = True
            st.session_state.start_time = time.time()
            
            st.rerun()

with col2:
    st.header("📥 ストリーミング出力")
    
    # 統計情報表示
    if st.session_state.is_processing or st.session_state.processing_complete:
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            elapsed_time = (time.time() - st.session_state.start_time) if st.session_state.start_time else 0
            st.markdown(f'<div class="stat-box blue-stat"><h4>{elapsed_time:.1f}s</h4>経過時間</div>', 
                       unsafe_allow_html=True)
        
        with col_stats2:
            st.markdown(f'<div class="stat-box green-stat"><h4>{st.session_state.chunk_count}</h4>受信チャンク</div>', 
                       unsafe_allow_html=True)
        
        with col_stats3:
            status_text = "完了" if st.session_state.processing_complete else "処理中"
            status_color = "green-stat" if st.session_state.processing_complete else "orange-stat"
            st.markdown(f'<div class="stat-box {status_color}"><h4>{status_text}</h4>ステータス</div>', 
                       unsafe_allow_html=True)
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ストリーミング処理の実行
    if st.session_state.is_processing and uploaded_pdf:
        # 一時ファイルとしてPDFを保存
        pdf_path = save_upload_file(uploaded_pdf)
        
        if pdf_path:
            try:
                # エクストラクタの初期化
                extractor = StreamingPatentExtractor(
                    model_name=model_name,
                    api_key=api_key,
                    json_schema=schema,
                    user_prompt=custom_prompt if custom_prompt else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # アシスタントメッセージでストリーミング出力
                with st.chat_message("assistant"):
                    # ストリーミング処理の実行
                    response = st.write_stream(extractor.stream_patent_extraction(pdf_path))
                
                # レスポンスをメッセージ履歴に追加
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                # JSON解析と最終結果の設定
                try:
                    final_result = extractor._extract_json_from_text(response)
                    st.session_state.final_result = final_result
                except Exception as e:
                    st.session_state.final_result = {
                        "error": f"JSON解析エラー: {str(e)}", 
                        "raw_output": response
                    }
                
                # 処理完了
                st.session_state.is_processing = False
                st.session_state.processing_complete = True
                
                st.rerun()
                
            except Exception as e:
                st.error(f"処理エラー: {str(e)}")
                st.session_state.is_processing = False
                
            finally:
                # 一時ファイルの削除
                try:
                    os.remove(pdf_path)
                except:
                    pass
    
    # 最終結果の表示と処理
    if st.session_state.processing_complete and st.session_state.final_result:
        st.markdown("---")
        st.markdown("### 📋 構造化JSON結果")
        
        # エラーがない場合のみJSONを表示
        if "error" not in st.session_state.final_result:
            st.json(st.session_state.final_result)
            
            # ダウンロードボタン
            if uploaded_pdf:
                json_str = json.dumps(st.session_state.final_result, ensure_ascii=False, indent=2)
                output_filename = f"{Path(uploaded_pdf.name).stem}_streaming.json"
                
                st.download_button(
                    label="📥 JSONダウンロード",
                    data=json_str.encode("utf-8"),
                    file_name=output_filename,
                    mime="application/json",
                    help="抽出された構造化データをJSONファイルとしてダウンロード"
                )
                
                # 統計情報表示
                json_size = len(json_str)
                keys_count = len(st.session_state.final_result.keys())
                
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.metric("JSON文字数", f"{json_size:,}")
                with col_result2:
                    st.metric("トップレベル要素", keys_count)
        else:
            # エラーの場合
            st.error(f"処理エラー: {st.session_state.final_result.get('error', '不明なエラー')}")
            if "raw_output" in st.session_state.final_result:
                with st.expander("生出力を表示"):
                    st.text(st.session_state.final_result["raw_output"])
    
    # プレースホルダー（処理前）
    if not st.session_state.is_processing and not st.session_state.processing_complete and not st.session_state.messages:
        st.info("PDFをアップロードして処理を開始してください")
        
        with st.expander("💡 ストリーミング機能について"):
            st.markdown("""
            **🚀 Streamlitチャット形式ストリーミングの特徴:**
            
            - **⚡ リアルタイム表示**: AIが生成したテキストがタイプライター効果で表示
            - **💬 チャット形式**: 会話形式で処理過程を確認
            - **📊 進捗追跡**: 経過時間とチャンク数をリアルタイム監視
            - **🎯 即座のフィードバック**: 処理開始直後から結果が見える
            - **📱 レスポンシブ**: モバイルでも快適に利用可能
            
            **📋 対応機能:**
            - PDF特許文書の構造化抽出
            - 多言語対応（日本語・英語など）
            - 大容量ファイル対応
            - JSONスキーマによるカスタマイズ
            - エラーハンドリングとフォールバック
            
            **🔧 技術的特徴:**
            - `st.write_stream()` によるネイティブストリーミング
            - メモリ効率的な処理
            - 安定したエラー処理
            """)

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    特許PDF構造化ツール（Streamlitネイティブストリーミング対応） v3.0<br>
    <small>Powered by StreamingPatentExtractor with Streamlit Chat API | Copyright (c) 2025 Pyxist Co.,Ltd</small>
</div>
""", unsafe_allow_html=True)
        