import streamlit as st
import json
import os
import tempfile
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from typing import Dict, List, Any

from patent_extractor import PatentExtractor

# ページ設定
st.set_page_config(
    page_title="特許PDF構造化ツール",
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
    .page-progress {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
    }
    .error-page {
        background-color: #fff5f5;
        border-left: 4px solid #dc3545;
    }
    .success-page {
        background-color: #f0fff4;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# モデルオプション（デフォルトの選択肢）
DEFAULT_MODEL_OPTIONS = {
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4-vision-preview"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
}

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
    
    return True  # その他の場合は通す

# APIから利用可能なモデル一覧を取得する関数
@st.cache_data(ttl=3600)  # 1時間キャッシュ
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
            # GPTモデルと Vision対応モデルのみをフィルタ
            model_names = [model.id for model in models.data 
                          if 'gpt' in model.id.lower() or 'vision' in model.id.lower()]
            return sorted(model_names, reverse=True)  # 新しいモデルが上に来るようにソート
        
        elif provider == "Anthropic":
            # AnthropicのAPIはモデル一覧を提供していないため、既知のモデルを返す
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
            # APIで取得できない場合はデフォルトを使用
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

# PDFを1ページずつ分割して処理する関数
def split_pdf_to_pages(pdf_path: str) -> List[str]:
    """PDFを1ページずつ分割して一時ファイルのパスリストを返す"""
    page_paths = []
    try:
        reader = PdfReader(pdf_path)
        
        for page_num in range(len(reader.pages)):
            # 新しいPDFライターを作成
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            
            # 一時ファイルとして保存
            temp_path = tempfile.mktemp(suffix=f"_page_{page_num + 1}.pdf")
            with open(temp_path, "wb") as output_file:
                writer.write(output_file)
            
            page_paths.append(temp_path)
        
        return page_paths
    except Exception as e:
        st.error(f"PDF分割エラー: {str(e)}")
        return []

# 1ページのPDFを処理する関数
def process_single_page(page_path: str, page_num: int, model_name: str, api_key: str, 
                       schema: Dict, prompt: str = None, temperature: float = 0.1, 
                       max_tokens: int = 4096) -> Dict:
    """1ページのPDFを処理"""
    try:
        extractor = PatentExtractor(
            model_name=model_name,
            api_key=api_key,
            json_schema=schema,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result = extractor.process_patent_pdf(page_path)
        return {
            "page_number": page_num,
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "page_number": page_num,
            "status": "error",
            "error": str(e),
            "data": {}
        }

# 複数ページの結果を統合する関数
def merge_page_results(page_results: List[Dict]) -> Dict:
    """複数ページの結果を統合してひとつのJSONを作成"""
    merged_result = {
        "processing_summary": {
            "total_pages": len(page_results),
            "successful_pages": sum(1 for r in page_results if r["status"] == "success"),
            "failed_pages": sum(1 for r in page_results if r["status"] == "error"),
            "page_details": []
        }
    }
    
    # 各ページの詳細を記録
    for result in page_results:
        page_detail = {
            "page_number": result["page_number"],
            "status": result["status"]
        }
        if result["status"] == "error":
            page_detail["error"] = result["error"]
        merged_result["processing_summary"]["page_details"].append(page_detail)
    
    # 成功したページのデータを統合
    successful_results = [r["data"] for r in page_results if r["status"] == "success"]
    
    if not successful_results:
        merged_result["error"] = "すべてのページで処理に失敗しました"
        return merged_result
    
    # 基本的な統合戦略
    # 1. publicationIdentifierは最初に見つかったものを使用
    # 2. セクションは配列として統合
    # 3. 単一値は最初に見つかったものを使用
    
    for i, result in enumerate(successful_results):
        if i == 0:
            # 最初の結果をベースとして使用
            for key, value in result.items():
                if key not in merged_result:
                    merged_result[key] = value
        else:
            # 後続の結果をマージ
            for key, value in result.items():
                if key in merged_result:
                    # 既存のキーがある場合の統合ロジック
                    if isinstance(merged_result[key], dict) and isinstance(value, dict):
                        # 辞書の場合は再帰的にマージ
                        merged_result[key] = merge_dict_values(merged_result[key], value)
                    elif isinstance(merged_result[key], list) and isinstance(value, list):
                        # リストの場合は結合
                        merged_result[key].extend(value)
                    # 単一値の場合は最初の値を保持（上書きしない）
                else:
                    # 新しいキーの場合は追加
                    merged_result[key] = value
    
    return merged_result

def merge_dict_values(dict1: Dict, dict2: Dict) -> Dict:
    """辞書の値を再帰的にマージ"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dict_values(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key].extend(value)
            # 単一値の場合は最初の値を保持
        else:
            result[key] = value
    
    return result

# ページ単位でPDFを処理する関数
def process_pdf_by_pages(pdf_path: str, model_name: str, api_key: str, schema: Dict, 
                        prompt: str = None, temperature: float = 0.1, 
                        max_tokens: int = 4096, progress_container=None) -> Dict:
    """PDFを1ページずつ処理して結果を統合"""
    try:
        # PDFを1ページずつ分割
        page_paths = split_pdf_to_pages(pdf_path)
        
        if not page_paths:
            return {"error": "PDFの分割に失敗しました"}
        
        total_pages = len(page_paths)
        page_results = []
        
        # 進捗表示用のコンテナ
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
        
        try:
            # 各ページを処理
            for i, page_path in enumerate(page_paths):
                page_num = i + 1
                
                # 進捗更新
                if progress_container:
                    progress = (i + 1) / total_pages
                    progress_bar.progress(progress)
                    status_text.text(f"ページ {page_num}/{total_pages} を処理中...")
                
                # ページ処理
                result = process_single_page(
                    page_path, page_num, model_name, api_key, schema, 
                    prompt, temperature, max_tokens
                )
                
                page_results.append(result)
                
                # 進捗表示の更新
                if progress_container:
                    with progress_container:
                        if result["status"] == "success":
                            st.markdown(f'<div class="page-progress success-page">✅ ページ {page_num}: 処理完了</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="page-progress error-page">❌ ページ {page_num}: エラー - {result["error"]}</div>', unsafe_allow_html=True)
        
        finally:
            # 一時ファイルの削除
            for page_path in page_paths:
                try:
                    os.remove(page_path)
                except:
                    pass
        
        # 進捗完了
        if progress_container:
            progress_bar.progress(1.0)
            status_text.text("全ページの処理が完了しました")
        
        # 結果を統合
        merged_result = merge_page_results(page_results)
        
        return merged_result
        
    except Exception as e:
        return {"error": f"処理エラー: {str(e)}"}

# PDFファイルを一時ファイルとして保存する関数
def save_upload_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"ファイル保存エラー: {str(e)}")
        return None

# タイトルと説明
st.markdown('<div class="title-area">', unsafe_allow_html=True)
st.title("🧩 特許PDF構造化ツール")
st.markdown("""
特許PDFからマルチモーダル生成AIを使用して構造化JSONを抽出します。
Gemini、GPT、Claudeなどの最新モデルを利用して特許データを解析できます。
**📄 ページ単位処理**: PDFを1ページずつ処理して結果を統合します。
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
    
    # APIキー入力（モデル一覧取得のために先に配置）
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
        # 利用可能なモデルを取得
        with st.spinner("利用可能なモデルを取得中..."):
            available_models = get_models_with_cache(provider, api_key)
        
        if available_models:
            # モデル一覧の情報表示
            if api_key:
                st.info(f"✅ {len(available_models)}個の利用可能なモデルを取得しました")
            else:
                st.info(f"ℹ️ デフォルトモデル一覧を表示中（APIキーを入力すると最新一覧を取得）")
            
            model_name = st.selectbox(
                "モデル",
                options=available_models,
                help="APIから取得した利用可能なモデル一覧です"
            )
            
            # モデル更新ボタン
            if st.button("🔄 モデル一覧を更新", help="最新のモデル一覧を再取得します"):
                st.cache_data.clear()  # キャッシュをクリア
                st.rerun()  # アプリを再実行
                
        else:
            st.error("利用可能なモデルを取得できませんでした。APIキーを確認してください。")
            model_name = ""
            
    else:
        # カスタムモデル名を直接入力
        model_placeholder = get_model_prefix(provider) + "model-name"
        model_name = st.text_input(
            "モデル名",
            placeholder=model_placeholder,
            help=f"{provider}の任意のモデル名を入力してください（例: {model_placeholder}）"
        )
        
        # モデル名の妥当性チェック
        if model_name and not is_valid_model_name(model_name, provider):
            st.warning(f"⚠️ 入力されたモデル名「{model_name}」は{provider}のモデルではない可能性があります。")
        
        # よく使われるモデルのヒント表示
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
    
    # モデル名が空の場合の警告
    if model_input_type == "カスタムモデル名を入力" and not model_name:
        st.error("モデル名を入力してください")
    
    # 選択されたモデルの情報表示
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
    
    else:  # デフォルトスキーマを使用
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
    
    # スキーマの詳細表示（オプション）
    if schema and st.checkbox("スキーマ詳細を表示"):
        st.json(schema)

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
        
        # PDFの基本情報を表示
        try:
            # PDFを一時保存してページ数を取得
            temp_pdf_path = save_upload_file(uploaded_pdf)
            if temp_pdf_path:
                reader = PdfReader(temp_pdf_path)
                page_count = len(reader.pages)
                os.remove(temp_pdf_path)
                
                st.info(f"📄 総ページ数: {page_count} ページ")
                st.info("🔄 このファイルは1ページずつ処理されます")
        except Exception as e:
            st.warning(f"PDFの情報取得に失敗: {str(e)}")
        
        # PDFの表示
        with st.expander("PDFプレビュー", expanded=False):
            pdf_display = f'<iframe src="data:application/pdf;base64,{uploaded_pdf.getvalue().hex()}" width="100%" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    
    # 処理ボタン
    process_button = st.button(
        "ページ単位で処理開始",
        disabled=not (uploaded_pdf and api_key and model_name),
        help="特許PDFを1ページずつ処理して構造化JSONを生成します"
    )

with col2:
    st.header("📥 出力")
    
    # PDFを処理
    if process_button and uploaded_pdf and api_key and model_name:
        with st.status("ページ単位で処理中...", expanded=True) as status:
            # アップロードされたPDFを一時ファイルとして保存
            pdf_path = save_upload_file(uploaded_pdf)
            
            if pdf_path:
                # 進捗表示用のコンテナ
                progress_container = st.container()
                
                # ページ単位で処理
                result = process_pdf_by_pages(
                    pdf_path=pdf_path,
                    model_name=model_name,
                    api_key=api_key,
                    schema=schema,
                    prompt=custom_prompt if custom_prompt else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    progress_container=progress_container
                )
                
                # 処理完了
                if "error" in result:
                    status.update(label=f"エラー: {result['error']}", state="error")
                else:
                    status.update(label="全ページの処理完了！", state="complete")
                    
                    # 処理結果の統計情報
                    processing_summary = result.get("processing_summary", {})
                    total_pages = processing_summary.get("total_pages", 0)
                    successful_pages = processing_summary.get("successful_pages", 0)
                    failed_pages = processing_summary.get("failed_pages", 0)
                    
                    # 統計情報の表示
                    st.markdown("### 処理結果概要")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.markdown(f'<div class="stat-box blue-stat"><h3>{total_pages}</h3>総ページ数</div>', unsafe_allow_html=True)
                    with col_stats2:
                        st.markdown(f'<div class="stat-box green-stat"><h3>{successful_pages}</h3>成功ページ</div>', unsafe_allow_html=True)
                    with col_stats3:
                        st.markdown(f'<div class="stat-box orange-stat"><h3>{failed_pages}</h3>失敗ページ</div>', unsafe_allow_html=True)
                    
                    # 処理詳細の表示
                    if processing_summary.get("page_details"):
                        with st.expander("ページ別処理詳細", expanded=False):
                            for detail in processing_summary["page_details"]:
                                if detail["status"] == "success":
                                    st.success(f"ページ {detail['page_number']}: 処理成功")
                                else:
                                    st.error(f"ページ {detail['page_number']}: {detail.get('error', '不明なエラー')}")
                    
                    # 統合された構造化データのキー数
                    data_keys = [k for k in result.keys() if k != "processing_summary"]
                    sections_count = sum(1 for k in data_keys if isinstance(result[k], dict))
                    
                    st.markdown("### 抽出データ概要")
                    col_data1, col_data2 = st.columns(2)
                    with col_data1:
                        st.markdown(f'<div class="stat-box blue-stat"><h3>{len(data_keys)}</h3>データ要素</div>', unsafe_allow_html=True)
                    with col_data2:
                        st.markdown(f'<div class="stat-box green-stat"><h3>{sections_count}</h3>セクション数</div>', unsafe_allow_html=True)
                    
                    # 統合結果の表示（processing_summaryを除く）
                    display_result = {k: v for k, v in result.items() if k != "processing_summary"}
                    
                    # JSON結果の表示
                    st.markdown("### 統合JSON出力")
                    st.json(display_result)
                    
                    # ダウンロードボタン
                    json_str = json.dumps(result, ensure_ascii=False, indent=2)
                    output_filename = f"{Path(uploaded_pdf.name).stem}_merged.json"
                    
                    col_download1, col_download2 = st.columns(2)
                    with col_download1:
                        st.download_button(
                            label="📥 統合JSONをダウンロード",
                            data=json_str.encode("utf-8"),
                            file_name=output_filename,
                            mime="application/json",
                            help="ページ別処理結果を統合したJSONファイルをダウンロードします"
                        )
                    
                    with col_download2:
                        # データのみ（処理情報を除く）のダウンロード
                        clean_json_str = json.dumps(display_result, ensure_ascii=False, indent=2)
                        clean_filename = f"{Path(uploaded_pdf.name).stem}_data_only.json"
                        st.download_button(
                            label="📄 データのみダウンロード",
                            data=clean_json_str.encode("utf-8"),
                            file_name=clean_filename,
                            mime="application/json",
                            help="処理情報を除いた構造化データのみをダウンロードします"
                        )
                
                # 一時ファイルの削除
                try:
                    os.remove(pdf_path)
                except:
                    pass
    else:
        # 処理前のプレースホルダー
        st.info("PDFをアップロードして「ページ単位で処理開始」ボタンをクリックすると、ここに統合された構造化JSONが表示されます")
        
        # デモ表示（オプション）
        with st.expander("出力例"):
            example_output = {
                "processing_summary": {
                    "total_pages": 10,
                    "successful_pages": 9,
                    "failed_pages": 1,
                    "page_details": [
                        {"page_number": 1, "status": "success"},
                        {"page_number": 2, "status": "success"},
                        {"page_number": 3, "status": "error", "error": "画像が不鮮明"},
                        {"page_number": 4, "status": "success"}
                    ]
                },
                "publicationIdentifier": "WO2020123456A1",
                "FrontPage": {
                    "title": "AI駆動特許データ抽出システム",
                    "PublicationData": {
                        "PublicationNumber": "WO2020123456A1",
                        "PublicationDate": "2020-06-15",
                        "PublicationKind": "A1"
                    },
                    "Applicants": {"Applicant": [{"Name": "サンプル株式会社"}]},
                    "Inventors": {"Inventor": [{"Name": "発明 太郎"}]},
                    "Abstract": {"Paragraph": [{"content": "これは特許の要約サンプルです..."}]}
                },
                "Claims": {
                    "Claim": [
                        {"id": "claim1", "number": 1, "Text": {"content": "AIを使用して特許文書から構造化データを抽出する方法..."}},
                        {"id": "claim2", "number": 2, "Text": {"content": "請求項1に記載の方法において..."}}
                    ]
                },
                "Description": {
                    "TechnicalField": {"Paragraph": [{"content": "本発明は、特許文書処理の分野に関する..."}]},
                    "BackgroundArt": {"Paragraph": [{"content": "従来の特許文書処理では..."}]},
                    "SummaryOfInvention": {"Paragraph": [{"content": "本発明の目的は..."}]}
                }
            }
            st.json(example_output)

# フッター
st.markdown("---")

# 処理方法の説明
with st.expander("📖 ページ単位処理について", expanded=False):
    st.markdown("""
    ### 🔄 ページ単位処理の特徴
    
    **処理フロー:**
    1. **PDF分割**: アップロードされたPDFを1ページずつ分割（pypdfライブラリ使用）
    2. **個別処理**: 各ページを独立してAIモデルで解析
    3. **リアルタイム進捗**: 各ページの処理状況をリアルタイムで表示
    4. **結果統合**: 全ページの処理結果を統合してひとつのJSONを生成
    5. **エラー耐性**: 一部のページでエラーが発生しても他のページの処理を継続
    
    **利点:**
    - 📊 **進捗の可視化**: どのページまで処理が完了したかリアルタイムで確認
    - 🛡️ **エラー耐性**: 一部ページの失敗が全体に影響しない
    - 🔧 **柔軟な統合**: 複数ページのデータを適切に結合
    - 💾 **メモリ効率**: 大きなPDFも効率的に処理
    
    **統合ルール:**
    - 📝 **基本情報**: 最初に見つかった値を使用（公開番号等）
    - 📚 **配列データ**: 全ページのデータを結合（請求項、段落等）
    - 🏗️ **構造化データ**: 辞書は再帰的にマージ
    - ⚠️ **エラー情報**: 処理概要に失敗ページの詳細を記録
    """)

st.markdown("""
<div style="text-align: center; color: #666;">
    特許PDF構造化ツール - マルチモーダル生成AIを使用した特許文書の構造化データ抽出<br>
    <small>Powered by patent-extractor library Copyright (c) 2025 Pyxist Co.,Ltd</small><br>
    <small>🔄 ページ単位処理機能搭載 - pypdf (MIT License) を使用した安定したPDF処理</small>
</div>
""", unsafe_allow_html=True)