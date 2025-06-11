import streamlit as st
import json
import os
import tempfile
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from typing import Dict, List
from patent_extractor import PatentExtractor

# ページ設定
st.set_page_config(page_title="特許PDF構造化ツール", page_icon="📄", layout="wide")

# カスタムCSS
st.markdown("""
<style>
    .stat-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center; }
    .blue-stat { background-color: #f0f5ff; border: 1px solid #d0e0ff; }
    .green-stat { background-color: #f0fff5; border: 1px solid #d0ffe0; }
    .orange-stat { background-color: #fff5f0; border: 1px solid #ffe0d0; }
    .page-badge { display: inline-block; padding: 4px 8px; margin: 2px; border-radius: 12px; 
                  font-size: 0.8em; font-weight: 500; text-align: center; }
    .page-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .page-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .page-processing { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
</style>
""", unsafe_allow_html=True)

# モデルオプション
DEFAULT_MODELS = {
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4o-mini"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
}

@st.cache_data(ttl=3600)
def get_models(provider, api_key):
    """利用可能なモデルを取得"""
    if not api_key:
        return DEFAULT_MODELS.get(provider, [])
    try:
        if provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return [m.name.replace('models/', '') for m in genai.list_models() 
                   if 'generateContent' in m.supported_generation_methods]
        elif provider == "OpenAI":
            from openai import OpenAI
            models = OpenAI(api_key=api_key).models.list()
            return sorted([m.id for m in models.data if 'gpt' in m.id.lower()], reverse=True)
        elif provider == "Anthropic":
            return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    except:
        pass
    return DEFAULT_MODELS.get(provider, [])

def load_schema(file_path=None, content=None):
    """JSONスキーマを読み込み"""
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif content:
            return json.loads(content)
    except Exception as e:
        st.error(f"スキーマ読み込みエラー: {e}")
    return {}

def split_pdf_chunks(pdf_path, chunk_size=2, overlap=1):
    """PDFを重複チャンクに分割"""
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        for i, start in enumerate(range(0, total_pages, chunk_size - overlap)):
            end = min(start + chunk_size, total_pages)
            writer = PdfWriter()
            
            for page_idx in range(start, end):
                writer.add_page(reader.pages[page_idx])
            
            temp_path = tempfile.mktemp(suffix=f"_chunk_{i+1}.pdf")
            with open(temp_path, "wb") as f:
                writer.write(f)
            
            chunks.append({
                "id": i + 1,
                "path": temp_path,
                "pages": list(range(start + 1, end + 1)),
                "start": start + 1,
                "end": end
            })
    except Exception as e:
        st.error(f"PDF分割エラー: {e}")
    return chunks

def process_chunk(chunk, model_name, api_key, schema, prompt=None, temperature=0.1, max_tokens=32768):
    """チャンクを処理"""
    try:
        extractor = PatentExtractor(model_name, api_key, schema, prompt, temperature, max_tokens)
        result = extractor.process_patent_pdf(chunk["path"])
        return {"id": chunk["id"], "pages": chunk["pages"], "status": "success", "data": result}
    except Exception as e:
        return {"id": chunk["id"], "pages": chunk["pages"], "status": "error", "error": str(e), "data": {}}

def merge_text(text1, text2):
    """テキストをマージ"""
    if not text1 or not text2:
        return text1 or text2
    
    text1, text2 = text1.strip(), text2.strip()
    
    # 継続パターンをチェック
    continues = (not text1.endswith(('.', '。', '!', '？')) or 
                text1.endswith((',', '、', ';')) or 
                text2.startswith(('が', 'を', 'に', 'の', 'は', 'と', 'で')) or 
                (text2 and text2[0].islower()))
    
    return f"{text1} {text2}" if continues else f"{text1}\n\n{text2}"

def merge_items(list1, list2):
    """リストアイテムをマージ（重複除去）"""
    result = list1.copy()
    for item in list2:
        if not any(str(item) in str(existing) for existing in result):
            result.append(item)
    return result

def merge_dicts(dict1, dict2):
    """辞書を再帰的にマージ"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = merge_items(result[key], value)
            elif isinstance(result[key], str) and isinstance(value, str):
                result[key] = merge_text(result[key], value)
        else:
            result[key] = value
    return result

def merge_chunks(chunk_results):
    """チャンク結果を統合"""
    successful = [r for r in chunk_results if r["status"] == "success"]
    if not successful:
        return {"error": "すべてのチャンクで処理失敗"}
    
    # 処理概要
    result = {
        "processing_summary": {
            "total_chunks": len(chunk_results),
            "successful_chunks": len(successful),
            "failed_chunks": len(chunk_results) - len(successful)
        }
    }
    
    # データをマージ
    for chunk in sorted(successful, key=lambda x: x["id"]):
        for key, value in chunk["data"].items():
            if key not in result:
                result[key] = value
            else:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    result[key] = merge_items(result[key], value)
                elif isinstance(result[key], str) and isinstance(value, str):
                    result[key] = merge_text(result[key], value)
    
    return result

def process_pdf(pdf_path, model_name, api_key, schema, prompt=None, temperature=0.1, 
               max_tokens=32768, chunk_size=2, overlap=1, progress_container=None):
    """PDF全体を処理"""
    chunks = split_pdf_chunks(pdf_path, chunk_size, overlap)
    if not chunks:
        return {"error": "PDF分割失敗"}
    
    results = []
    total = len(chunks)
    
    if progress_container:
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        badge_container = progress_container.empty()
    
    try:
        for i, chunk in enumerate(chunks):
            if progress_container:
                progress_bar.progress((i + 1) / total)
                status_text.text(f"チャンク {chunk['id']}/{total} (P{chunk['start']}-{chunk['end']}) 処理中...")
            
            result = process_chunk(chunk, model_name, api_key, schema, prompt, temperature, max_tokens)
            results.append(result)
            
            # バッジ表示更新
            if progress_container:
                badges = []
                for j, res in enumerate(results):
                    pages = f"P{res['pages'][0]}-{res['pages'][-1]}"
                    status_class = "page-success" if res["status"] == "success" else "page-error"
                    symbol = "✓" if res["status"] == "success" else "✗"
                    badges.append(f'<span class="page-badge {status_class}">C{res["id"]} ({pages}) {symbol}</span>')
                
                # 未処理チャンク
                for k in range(len(results), total):
                    chunk_info = chunks[k]
                    pages = f"P{chunk_info['start']}-{chunk_info['end']}"
                    class_name = "page-processing" if k == len(results) else ""
                    symbol = "..." if k == len(results) else ""
                    badges.append(f'<span class="page-badge {class_name}">C{chunk_info["id"]} ({pages}) {symbol}</span>')
                
                badge_container.markdown(f'<div>{"".join(badges)}</div>', unsafe_allow_html=True)
        
        if progress_container:
            status_text.text("処理完了")
    
    finally:
        # 一時ファイル削除
        for chunk in chunks:
            try:
                os.remove(chunk["path"])
            except:
                pass
    
    return merge_chunks(results)

# UI
st.title("🧩 特許PDF構造化ツール")
st.markdown("特許PDFからマルチモーダル生成AIを使用して構造化JSONを抽出します。")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    # プロバイダー選択
    provider = st.selectbox("AIプロバイダー", list(DEFAULT_MODELS.keys()))
    
    # APIキー
    api_key_env = {"Google Gemini": "GOOGLE_API_KEY", "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY"}
    api_key = st.text_input(f"{provider} APIキー", value=os.environ.get(api_key_env[provider], ""), type="password")
    
    # モデル選択
    available_models = get_models(provider, api_key)
    model_name = st.selectbox("モデル", available_models) if available_models else ""
    
    # スキーマ
    schema_type = st.radio("JSONスキーマ", ["デフォルト", "ファイル", "直接入力"])
    schema = {}
    
    if schema_type == "ファイル":
        uploaded_schema = st.file_uploader("JSONスキーマファイル", type=["json"])
        if uploaded_schema:
            schema = load_schema(content=uploaded_schema.getvalue().decode("utf-8"))
    elif schema_type == "直接入力":
        schema_text = st.text_area("JSONスキーマ", height=150)
        if schema_text:
            schema = load_schema(content=schema_text)
    else:
        schema = load_schema("default_schema.json")
    
    # 詳細設定
    with st.expander("詳細設定"):
        prompt = st.text_area("カスタムプロンプト", height=80)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.number_input("最大トークン数", 1024, 65535, 32768, 1024)
        
        st.markdown("### チャンク設定")
        chunk_size = st.selectbox("チャンクサイズ", [1, 2, 3, 4, 5], index=1)
        overlap = st.selectbox("重複サイズ", [0, 1, 2], index=1)

# メイン
col1, col2 = st.columns(2)

with col1:
    st.header("📤 入力")
    uploaded_pdf = st.file_uploader("特許PDFをアップロード", type=["pdf"])
    
    if uploaded_pdf:
        st.success(f"ファイル: {uploaded_pdf.name}")
        try:
            temp_path = tempfile.mktemp(suffix=".pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            reader = PdfReader(temp_path)
            st.info(f"📄 総ページ数: {len(reader.pages)}")
            os.remove(temp_path)
        except:
            pass
    
    process_button = st.button("処理開始", disabled=not (uploaded_pdf and api_key and model_name))

with col2:
    st.header("📥 出力")
    
    if process_button and uploaded_pdf and api_key and model_name:
        with st.status("処理中...", expanded=True) as status:
            # PDF保存
            pdf_path = tempfile.mktemp(suffix=".pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            # 処理実行
            progress_container = st.container()
            result = process_pdf(pdf_path, model_name, api_key, schema, prompt, 
                               temperature, max_tokens, chunk_size, overlap, progress_container)
            
            # 後処理
            os.remove(pdf_path)
            
            if "error" in result:
                status.update(label=f"エラー: {result['error']}", state="error")
            else:
                status.update(label="処理完了", state="complete")
                
                # 統計表示
                summary = result.get("processing_summary", {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="stat-box blue-stat"><h3>{summary.get("total_chunks", 0)}</h3>総チャンク</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="stat-box green-stat"><h3>{summary.get("successful_chunks", 0)}</h3>成功</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="stat-box orange-stat"><h3>{summary.get("failed_chunks", 0)}</h3>失敗</div>', unsafe_allow_html=True)
                
                # 結果表示
                display_data = {k: v for k, v in result.items() if k != "processing_summary"}
                st.markdown("### JSON出力")
                st.json(display_data)
                
                # ダウンロード
                json_str = json.dumps(result, ensure_ascii=False, indent=2)
                filename = f"{Path(uploaded_pdf.name).stem}_processed.json"
                st.download_button("📥 JSONダウンロード", json_str.encode("utf-8"), filename, "application/json")
    else:
        st.info("PDFをアップロードして処理を開始してください")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
特許PDF構造化ツール - チャンク重複処理によるAI文書解析<br>
<small>Powered by patent-extractor library</small>
</div>
""", unsafe_allow_html=True)