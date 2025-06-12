import streamlit as st
import json
import os
import tempfile
import time
from pathlib import Path

from patent_extractor_structured import PatentExtractor

# ページ設定
st.set_page_config(
    page_title="特許PDF構造化ツール",
    page_icon="⚡",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .metric-row {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    .metric-card {
        flex: 1;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .performance-info {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# モデルオプション
MODEL_OPTIONS = {
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4-vision-preview"],
    "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
}

def get_api_key_from_env(provider):
    env_vars = {
        "Google Gemini": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY", 
        "Anthropic": "ANTHROPIC_API_KEY"
    }
    return os.environ.get(env_vars.get(provider, ""), "")

def load_schema(file_path=None, file_content=None):
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file_content:
            return json.loads(file_content)
        return {}
    except Exception as e:
        st.error(f"スキーマ読み込みエラー: {str(e)}")
        return {}

def process_pdf(pdf_path, model_name, api_key, schema, prompt=None, temperature=0.1, max_tokens=4096, max_workers=8):
    try:
        extractor = PatentExtractor(
            model_name=model_name,
            api_key=api_key,
            json_schema=schema,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers
        )
        return extractor.process_patent_pdf(pdf_path)
    except Exception as e:
        return {"error": str(e)}

def save_upload_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"ファイル保存エラー: {str(e)}")
        return None

def display_performance_metrics(processing_info):
    if not processing_info:
        return
    
    total_time = processing_info.get("total_time_seconds", 0)
    field_timing = processing_info.get("field_timing", {})
    workers = processing_info.get("parallel_workers", 1)
    
    st.markdown("### パフォーマンス情報")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総処理時間", f"{total_time:.1f}秒")
    with col2:
        st.metric("並列ワーカー", f"{workers}")
    with col3:
        st.metric("処理フィールド", f"{len(field_timing)}")
    with col4:
        estimated_sequential = sum(field_timing.values()) if field_timing else total_time
        speedup = estimated_sequential / total_time if total_time > 0 else 1
        st.metric("高速化", f"{speedup:.1f}x")
    
    if field_timing:
        with st.expander("フィールド別処理時間"):
            for field, timing in sorted(field_timing.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{field}**: {timing:.2f}秒")

# メインUI
st.title("⚡ 特許PDF構造化ツール")
st.markdown("特許PDFから構造化JSONを並列処理で高速抽出")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    
    # 並列処理設定
    max_workers = st.slider("並列ワーカー数", 1, 16, 8, help="同時処理数")
    
    # AIモデル設定
    provider = st.selectbox("AIプロバイダー", list(MODEL_OPTIONS.keys()))
    model_name = st.selectbox("モデル", MODEL_OPTIONS[provider])
    api_key = st.text_input(f"{provider} APIキー", value=get_api_key_from_env(provider), type="password")
    
    # スキーマ設定
    schema_type = st.radio("JSONスキーマ", ["デフォルト", "ファイルアップロード", "直接入力"])
    
    schema = {}
    if schema_type == "ファイルアップロード":
        uploaded_schema = st.file_uploader("スキーマファイル", type=["json"])
        if uploaded_schema:
            schema = load_schema(file_content=uploaded_schema.getvalue().decode("utf-8"))
            if schema:
                st.success("スキーマ読み込み完了")
    elif schema_type == "直接入力":
        schema_text = st.text_area("JSONスキーマ", height=150)
        if schema_text:
            try:
                schema = json.loads(schema_text)
                st.success("スキーマ形式OK")
            except:
                st.error("JSON形式エラー")
    else:
        if os.path.exists("default_schema.json"):
            schema = load_schema(file_path="default_schema.json")
            st.success("デフォルトスキーマ使用")
    
    # 詳細設定
    with st.expander("詳細設定"):
        custom_prompt = st.text_area("カスタムプロンプト", height=80)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.number_input("最大トークン数", 1024, 32768, 4096, 1024)

# メイン処理エリア
col1, col2 = st.columns([1, 1])

with col1:
    st.header("入力")
    
    uploaded_pdf = st.file_uploader("特許PDFファイル", type=["pdf"])
    
    if uploaded_pdf:
        st.success(f"ファイル: {uploaded_pdf.name}")
    
    process_button = st.button(
        f"並列処理開始 ({max_workers}workers)",
        disabled=not (uploaded_pdf and api_key and model_name),
        use_container_width=True
    )

with col2:
    st.header("出力")
    
    if process_button and uploaded_pdf and api_key and model_name:
        pdf_path = save_upload_file(uploaded_pdf)
        
        if pdf_path:
            with st.spinner(f"{model_name}で並列処理中..."):
                start_time = time.time()
                result = process_pdf(
                    pdf_path=pdf_path,
                    model_name=model_name,
                    api_key=api_key,
                    schema=schema,
                    prompt=custom_prompt if custom_prompt else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_workers=max_workers
                )
                
                if "error" in result:
                    st.error(f"処理エラー: {result['error']}")
                else:
                    st.success("処理完了！")
                    
                    # パフォーマンス情報表示
                    if "_processing_info" in result:
                        display_performance_metrics(result["_processing_info"])
                    
                    # 結果統計
                    clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
                    total_fields = len(clean_result)
                    successful_fields = len([v for v in clean_result.values() if not (isinstance(v, dict) and 'error' in v)])
                    
                    st.write(f"**抽出結果**: {successful_fields}/{total_fields} フィールド成功")
                    
                    # JSON表示
                    st.json(clean_result)
                    
                    # ダウンロード
                    json_str = json.dumps(clean_result, indent=2, ensure_ascii=False)
                    output_filename = f"{Path(uploaded_pdf.name).stem}.json"
                    
                    st.download_button(
                        "JSONダウンロード",
                        data=json_str.encode("utf-8"),
                        file_name=output_filename,
                        mime="application/json",
                        use_container_width=True
                    )
            
            # 一時ファイル削除
            try:
                os.remove(pdf_path)
            except:
                pass
    else:
        st.info("PDFをアップロードして処理開始ボタンを押してください")
        
        # 簡単な例
        with st.expander("出力例"):
            example = {
                "publicationIdentifier": "WO2024123456A1",
                "FrontPage": {"title": "AI特許抽出システム"},
                "Claims": {"Claim": [{"id": "1", "Text": {"content": "AIを使用する方法..."}}]},
                "Description": {"TechnicalField": {"Paragraph": [{"content": "AI分野に関する..."}]}}
            }
            st.json(example)

# フッター
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "特許PDF構造化ツール - 並列処理で高速データ抽出"
    "</div>", 
    unsafe_allow_html=True
)