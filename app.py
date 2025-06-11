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
    """チャンクを処理（Gemini安全性フィルタ対策）"""
    try:
        # より安全なプロンプトを作成
        safe_prompt = create_safe_prompt(prompt, schema)
        
        extractor = PatentExtractor(model_name, api_key, schema, safe_prompt, temperature, max_tokens)
        
        # Geminiの安全性設定を調整する場合のパラメータ
        # (patent_extractorライブラリがサポートしている場合)
        raw_result = extractor.process_patent_pdf(chunk["path"])
        
        # レスポンスの安全性チェック
        if raw_result is None or raw_result == {}:
            # 空のレスポンスの場合は基本情報のみ抽出
            fallback_result = create_fallback_extraction(chunk)
            return {"id": chunk["id"], "pages": chunk["pages"], "status": "partial", 
                   "data": fallback_result, "warning": "安全性フィルタのため部分的な抽出"}
        
        # JSONレスポンスのクリーニング
        cleaned_result = clean_json_response(raw_result)
        
        return {"id": chunk["id"], "pages": chunk["pages"], "status": "success", "data": cleaned_result}
        
    except Exception as e:
        error_msg = str(e)
        
        # Gemini特有のエラーを特定
        if "finish_reason is 2" in error_msg or "SAFETY" in error_msg:
            # 安全性フィルタによるブロックの場合
            fallback_result = create_fallback_extraction(chunk)
            return {"id": chunk["id"], "pages": chunk["pages"], "status": "blocked", 
                   "data": fallback_result, "error": "安全性フィルタによりブロック"}
        elif "response.text" in error_msg:
            # Geminiレスポンスエラーの場合
            return {"id": chunk["id"], "pages": chunk["pages"], "status": "error", 
                   "error": "Gemini APIレスポンスエラー", "data": {}}
        else:
            # その他のエラー
            return {"id": chunk["id"], "pages": chunk["pages"], "status": "error", 
                   "error": error_msg, "data": {}}

def create_safe_prompt(original_prompt, schema):
    """安全性フィルタを回避するためのプロンプト作成"""
    safe_prompt = f"""
    以下は学術的な特許文書の情報抽出タスクです。
    
    タスク: 特許文書から構造化された情報を抽出し、JSONフォーマットで出力してください。
    
    抽出対象:
    - 特許番号や出願情報
    - 技術分野
    - 発明の背景
    - 課題と解決手段
    - 効果
    - 実施例
    
    出力形式: 有効なJSONのみ
    
    注意事項:
    - 学術的・教育的目的での使用
    - 公開されている特許情報の構造化
    - 研究目的での情報整理
    
    {original_prompt or ""}
    """
    
    return safe_prompt

def create_fallback_extraction(chunk):
    """安全性フィルタブロック時のフォールバック抽出"""
    return {
        "processing_info": {
            "chunk_id": chunk["id"],
            "pages": chunk["pages"],
            "status": "limited_extraction",
            "reason": "安全性フィルタのため基本情報のみ抽出"
        },
        "basic_info": {
            "document_type": "特許文書",
            "pages_processed": len(chunk["pages"]),
            "extraction_level": "minimal"
        }
    }

def handle_gemini_safety_settings():
    """Gemini安全性設定のガイダンス"""
    return """
    Gemini安全性フィルタ対策:
    
    1. プロンプトの調整
       - 学術的・教育的目的を明記
       - 技術文書として扱うことを強調
       
    2. モデル選択
       - gemini-1.5-flash より gemini-1.5-pro を推奨
       - より高度なコンテキスト理解
       
    3. 処理方法
       - チャンクサイズを小さく (1-2ページ)
       - 複雑な内容を分割処理
    """

def merge_chunks_with_safety_handling(chunk_results):
    """安全性ブロックを考慮したチャンク統合"""
    successful = [r for r in chunk_results if r["status"] in ["success", "partial"]]
    blocked = [r for r in chunk_results if r["status"] == "blocked"]
    failed = [r for r in chunk_results if r["status"] == "error"]
    
    if not successful:
        return {"error": "すべてのチャンクで処理に失敗しました"}
    
    # 処理概要（詳細）
    result = {
        "processing_summary": {
            "total_chunks": len(chunk_results),
            "successful_chunks": len([r for r in chunk_results if r["status"] == "success"]),
            "partial_chunks": len([r for r in chunk_results if r["status"] == "partial"]),
            "blocked_chunks": len(blocked),
            "failed_chunks": len(failed),
            "safety_filter_issues": len(blocked) > 0
        }
    }
    
    # 安全性フィルタの警告
    if blocked:
        result["safety_warning"] = {
            "message": "一部のチャンクが安全性フィルタによりブロックされました",
            "blocked_pages": [r["pages"] for r in blocked],
            "recommendation": "より小さなチャンクサイズまたは異なるモデルを試してください"
        }
    
    # データをマージ（成功および部分成功のみ）
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
    
    # 全体の重複除去処理
    result = clean_duplicates(result)
    
    return result

def clean_json_response(response):
    """AIレスポンスのJSONクリーニング"""
    if isinstance(response, dict):
        return response  # 既にパース済みの場合
    
    if isinstance(response, str):
        try:
            # 基本的なJSONクリーニング
            cleaned = response.strip()
            
            # マークダウンコードブロックの除去
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            # 前後の空白除去
            cleaned = cleaned.strip()
            
            # JSONとして解析
            parsed = json.loads(cleaned)
            return parsed
            
        except json.JSONDecodeError:
            # JSONパースに失敗した場合は空の辞書を返す
            st.warning(f"JSON解析に失敗しました。空のデータを返します。")
            return {}
    
    return response or {}

def safe_json_dumps(obj, **kwargs):
    """安全なJSON文字列化"""
    try:
        return json.dumps(obj, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError) as e:
        # シリアライズできないオブジェクトがある場合の対処
        st.warning(f"JSON変換エラー: {e}")
        return json.dumps({"error": "JSON変換に失敗しました"}, ensure_ascii=False, **kwargs)

def merge_text(text1, text2):
    """テキストをマージ（重複除去・安全性向上）"""
    # 入力値の型チェックと安全な変換
    str1 = str(text1) if text1 is not None else ""
    str2 = str(text2) if text2 is not None else ""
    
    if not str1 or not str2:
        return str1 or str2
    
    str1, str2 = str1.strip(), str2.strip()
    
    # 完全一致の場合は片方を返す
    if str1 == str2:
        return str1
    
    # 一方が他方に含まれる場合は長い方を返す
    if str1 in str2:
        return str2
    if str2 in str1:
        return str1
    
    # 空白区切りで分割して重複を除去
    try:
        words1 = str1.split()
        words2 = str2.split()
        
        # 大部分が重複している場合（類似度80%以上）は長い方を採用
        common_words = set(words1) & set(words2)
        total_words = set(words1) | set(words2)
        if total_words and len(common_words) / len(total_words) > 0.8:
            return str1 if len(str1) > len(str2) else str2
        
        # 継続パターンをチェック
        continues = (not str1.endswith(('.', '。', '!', '？')) or 
                    str1.endswith((',', '、', ';')) or 
                    str2.startswith(('が', 'を', 'に', 'の', 'は', 'と', 'で')) or 
                    (str2 and str2[0].islower()))
        
        return f"{str1} {str2}" if continues else f"{str1}\n\n{str2}"
    except Exception:
        # エラーが発生した場合は安全に結合
        return f"{str1} {str2}"

def merge_items(list1, list2):
    """リストアイテムをマージ（重複除去・エラー対策）"""
    if not isinstance(list1, list):
        list1 = []
    if not isinstance(list2, list):
        list2 = []
    
    if not list1:
        return list2
    if not list2:
        return list1
    
    result = list1.copy()
    
    for item in list2:
        is_duplicate = False
        
        try:
            for existing in result:
                if isinstance(item, dict) and isinstance(existing, dict):
                    # 辞書の場合：キーと値の組み合わせで判定
                    if (item.get('id') and existing.get('id') and item['id'] == existing['id']) or \
                       (item.get('number') and existing.get('number') and item['number'] == existing['number']) or \
                       (item.get('content') and existing.get('content') and 
                        item.get('content', '') == existing.get('content', '')):
                        is_duplicate = True
                        # より完全な情報で既存を更新
                        if len(str(item)) > len(str(existing)):
                            idx = result.index(existing)
                            result[idx] = item
                        break
                elif isinstance(item, str) and isinstance(existing, str):
                    # 文字列の場合：内容で判定
                    if item == existing or item in existing or existing in item:
                        is_duplicate = True
                        # より長い文字列で更新
                        if len(item) > len(existing):
                            idx = result.index(existing)
                            result[idx] = item
                        break
                elif str(item) == str(existing):
                    is_duplicate = True
                    break
        except Exception:
            # 比較エラーの場合は重複していないとみなす
            pass
        
        if not is_duplicate:
            result.append(item)
    
    return result

def merge_dicts(dict1, dict2):
    """辞書を再帰的にマージ（エラー対策強化）"""
    if not isinstance(dict1, dict):
        dict1 = {}
    if not isinstance(dict2, dict):
        dict2 = {}
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        try:
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    result[key] = merge_items(result[key], value)
                elif isinstance(result[key], str) and isinstance(value, str):
                    result[key] = merge_text(result[key], value)
                else:
                    # 型が異なる場合は文字列として結合
                    result[key] = merge_text(str(result[key]), str(value))
            else:
                result[key] = value
        except Exception as e:
            # マージエラーの場合は新しい値を採用
            st.warning(f"マージエラー (キー: {key}): {e}")
            result[key] = value
    
    return result

def merge_text(text1, text2):
    """テキストをマージ（重複除去）"""
    if not text1 or not text2:
        return text1 or text2
    
    text1, text2 = text1.strip(), text2.strip()
    
    # 完全一致の場合は片方を返す
    if text1 == text2:
        return text1
    
    # 一方が他方に含まれる場合は長い方を返す
    if text1 in text2:
        return text2
    if text2 in text1:
        return text1
    
    # 空白区切りで分割して重複を除去
    words1 = text1.split()
    words2 = text2.split()
    
    # 大部分が重複している場合（類似度80%以上）は長い方を採用
    common_words = set(words1) & set(words2)
    total_words = set(words1) | set(words2)
    if total_words and len(common_words) / len(total_words) > 0.8:
        return text1 if len(text1) > len(text2) else text2
    
    # 継続パターンをチェック
    continues = (not text1.endswith(('.', '。', '!', '？')) or 
                text1.endswith((',', '、', ';')) or 
                text2.startswith(('が', 'を', 'に', 'の', 'は', 'と', 'で')) or 
                (text2 and text2[0].islower()))
    
    return f"{text1} {text2}" if continues else f"{text1}\n\n{text2}"

def merge_items(list1, list2):
    """リストアイテムをマージ（重複除去）"""
    if not list1:
        return list2
    if not list2:
        return list1
    
    result = list1.copy()
    
    for item in list2:
        # アイテムの重複チェック
        is_duplicate = False
        
        for existing in result:
            if isinstance(item, dict) and isinstance(existing, dict):
                # 辞書の場合：キーと値の組み合わせで判定
                if ('id' in item and 'id' in existing and item['id'] == existing['id']) or \
                   ('number' in item and 'number' in existing and item['number'] == existing['number']) or \
                   ('content' in item and 'content' in existing and 
                    item.get('content', '') == existing.get('content', '')):
                    is_duplicate = True
                    # より完全な情報で既存を更新
                    if len(str(item)) > len(str(existing)):
                        idx = result.index(existing)
                        result[idx] = item
                    break
            elif isinstance(item, str) and isinstance(existing, str):
                # 文字列の場合：内容で判定
                if item == existing or item in existing or existing in item:
                    is_duplicate = True
                    # より長い文字列で更新
                    if len(item) > len(existing):
                        idx = result.index(existing)
                        result[idx] = item
                    break
            elif str(item) == str(existing):
                is_duplicate = True
                break
        
        if not is_duplicate:
            result.append(item)
    
    return result

def clean_duplicates(data):
    """データ全体から重複を除去（エラー対策強化）"""
    try:
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                try:
                    if isinstance(value, str):
                        # 文字列から重複パターンを除去
                        cleaned_text = remove_repeated_patterns(value)
                        cleaned[key] = cleaned_text
                    elif isinstance(value, (dict, list)):
                        cleaned[key] = clean_duplicates(value)
                    else:
                        cleaned[key] = value
                except Exception as e:
                    # 個別キーの処理でエラーが発生した場合は元の値を保持
                    st.warning(f"データクリーニングエラー (キー: {key}): {e}")
                    cleaned[key] = value
            return cleaned
        elif isinstance(data, list):
            try:
                return [clean_duplicates(item) for item in data]
            except Exception:
                # リスト処理でエラーの場合は元のリストを返す
                return data
        else:
            return data
    except Exception:
        # 全体処理でエラーの場合は元のデータを返す
        return data

def remove_repeated_patterns(text):
    """文字列から重複パターンを除去（エラー対策）"""
    try:
        if not isinstance(text, str) or not text.strip():
            return text
        
        # 空白で分割
        words = text.split()
        if len(words) <= 1:
            return text
        
        # 連続する重複単語を除去
        cleaned_words = [words[0]]
        for word in words[1:]:
            if word != cleaned_words[-1]:
                cleaned_words.append(word)
        
        # 同じフレーズの繰り返しを検出して除去
        result_text = ' '.join(cleaned_words)
        
        # より長いパターンの重複を検出
        for pattern_length in range(min(len(cleaned_words) // 2, 10), 0, -1):
            try:
                pattern = cleaned_words[:pattern_length]
                pattern_str = ' '.join(pattern)
                
                # パターンが複数回繰り返されているかチェック
                if len(pattern_str) > 0 and result_text.count(pattern_str) > 1:
                    # 最初の出現のみを残す
                    parts = result_text.split(pattern_str)
                    if len(parts) > 2:  # パターンが2回以上出現
                        result_text = pattern_str.join([parts[0], parts[1]]) + pattern_str
                        break
            except Exception:
                continue
        
        return result_text.strip()
    except Exception:
        # エラーの場合は元のテキストを返す
        return str(text) if text is not None else ""

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
    """チャンク結果を統合（安全性フィルタ対応）"""
    return merge_chunks_with_safety_handling(chunk_results)

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
                json_str = safe_json_dumps(result, indent=2)
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