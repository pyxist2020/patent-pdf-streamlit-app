import streamlit as st
import json
import os
import tempfile
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 新しい並列処理システムをインポート
try:
    from patent_extractor import PatentExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False
    st.error("❌ 抽出エンジンが見つかりません。patent_extractor.py を確認してください。")

# ページ設定
st.set_page_config(
    page_title="🚀 並列特許PDF構造化ツール",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
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
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .performance-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .domain-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        background: #4CAF50;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    .success-metrics {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .error-metrics {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# デフォルトモデルオプション
DEFAULT_MODEL_OPTIONS = {
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
    "OpenAI": ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
}

def get_api_key_from_env(provider):
    """環境変数からAPIキーを取得"""
    env_vars = {
        "Google Gemini": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY", 
        "Anthropic": "ANTHROPIC_API_KEY"
    }
    return os.environ.get(env_vars.get(provider, ""), "")

@st.cache_data(ttl=3600)
def fetch_available_models(provider, api_key):
    """APIから利用可能なモデル一覧を取得（キャッシュ付き）"""
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
                          if any(keyword in model.id.lower() for keyword in ['gpt', 'vision', 'turbo'])]
            return sorted(model_names, reverse=True)
        
        elif provider == "Anthropic":
            # AnthropicのAPIはモデル一覧を提供していないため、既知のモデルを返す
            return [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022", 
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        
        return []
        
    except Exception as e:
        st.warning(f"⚠️ モデル一覧の取得に失敗: {str(e)}")
        return []

def get_models_with_cache(provider, api_key):
    """キャッシュを使用してモデル一覧を取得"""
    if not api_key:
        return DEFAULT_MODEL_OPTIONS.get(provider, [])
    
    try:
        available_models = fetch_available_models(provider, api_key)
        return available_models if available_models else DEFAULT_MODEL_OPTIONS.get(provider, [])
    except:
        return DEFAULT_MODEL_OPTIONS.get(provider, [])

def load_schema(file_path=None, file_content=None):
    """スキーマファイルを読み込み"""
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
                st.success(f"✅ スキーマ読み込み完了: {len(schema.get('properties', {}))} プロパティ")
                return schema
        elif file_content:
            schema = json.loads(file_content)
            st.success(f"✅ スキーマ解析完了: {len(schema.get('properties', {}))} プロパティ")
            return schema
        return {}
    except Exception as e:
        st.error(f"❌ スキーマ読み込みエラー: {str(e)}")
        return {}

def detect_domain_only(pdf_path, model_name, api_key):
    """ドメイン検出のみ実行"""
    try:
        if not EXTRACTOR_AVAILABLE:
            return {"error": "抽出エンジンが利用できません"}
            
        extractor = PatentExtractor(
            model_name=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2048
        )
        return extractor.detect_domain_parallel(pdf_path)
    except Exception as e:
        return {"error": str(e)}

def process_pdf_parallel(pdf_path, model_name, api_key, schema, prompt=None, temperature=0.1, max_tokens=8192, max_workers=8):
    """並列処理でPDFを処理"""
    try:
        if not EXTRACTOR_AVAILABLE:
            return {"error": "抽出エンジンが利用できません"}
            
        extractor = PatentExtractor(
            model_name=model_name,
            api_key=api_key,
            json_schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers
        )
        return extractor.process_patent_parallel(pdf_path)
    except Exception as e:
        return {"error": str(e)}

def save_upload_file(uploaded_file):
    """アップロードファイルを一時保存"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"❌ ファイル保存エラー: {str(e)}")
        return None

def display_domain_info(domain_info):
    """ドメイン情報を表示"""
    if not domain_info or "error" in domain_info:
        return
    
    domain = domain_info.get("primary_domain", "unknown")
    
    # ドメインバッジ
    domain_colors = {
        "chemical": "#FF6B6B",
        "biotechnology": "#4ECDC4", 
        "mechanical": "#45B7D1",
        "electrical": "#96CEB4",
        "software": "#FFEAA7",
        "general": "#DDA0DD"
    }
    
    color = domain_colors.get(domain, "#DDA0DD")
    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
        🔍 検出ドメイン: <strong>{domain.upper()}</strong>
    </div>
    """, unsafe_allow_html=True)

def display_performance_metrics(processing_metadata):
    """パフォーマンス指標を表示"""
    if not processing_metadata:
        return
    
    st.markdown("### 📊 処理パフォーマンス")
    
    total_time = processing_metadata.get("total_time_seconds", 0)
    workers = processing_metadata.get("parallel_workers", 1)
    stats = processing_metadata.get("processing_stats", {})
    field_timing = processing_metadata.get("field_timing", {})
    
    # メトリクス表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ 総処理時間", f"{total_time:.1f}秒")
    
    with col2:
        st.metric("🚀 並列ワーカー", f"{workers}")
    
    with col3:
        successful = stats.get("successful_fields", 0)
        total = stats.get("total_fields", 0)
        success_rate = (successful / max(1, total)) * 100
        st.metric("✅ 成功率", f"{success_rate:.1f}%")
    
    with col4:
        if field_timing:
            estimated_sequential = sum(field_timing.values())
            speedup = estimated_sequential / total_time if total_time > 0 else 1
            st.metric("⚡ 高速化", f"{speedup:.1f}x")
        else:
            st.metric("⚡ 高速化", "N/A")
    
    # 詳細タイミング
    if field_timing:
        with st.expander("📈 フィールド別処理時間"):
            # データフレーム作成
            timing_df = pd.DataFrame([
                {"フィールド": field, "処理時間(秒)": timing}
                for field, timing in sorted(field_timing.items(), key=lambda x: x[1], reverse=True)
            ])
            
            # グラフ表示
            fig = px.bar(timing_df, x="フィールド", y="処理時間(秒)", 
                        title="フィールド別処理時間",
                        color="処理時間(秒)",
                        color_continuous_scale="viridis")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # テーブル表示
            st.dataframe(timing_df, use_container_width=True)

def display_validation_results(result, schema):
    """バリデーション結果を表示"""
    if not schema:
        return
    
    st.markdown("### 🔍 スキーマ検証結果")
    
    # 必須フィールドチェック
    required_fields = schema.get("required", [])
    schema_properties = schema.get("properties", {})
    
    present_required = [f for f in required_fields if f in result and result[f] is not None]
    missing_required = [f for f in required_fields if f not in result or result[f] is None]
    
    optional_present = [f for f in schema_properties if f not in required_fields and f in result and result[f] is not None]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="success-metrics">
            <h4>✅ 必須フィールド</h4>
            <h2>{len(present_required)}/{len(required_fields)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📋 オプション</h4>
            <h2>{len(optional_present)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_coverage = (len(present_required) + len(optional_present)) / max(1, len(schema_properties)) * 100
        coverage_class = "success-metrics" if total_coverage > 80 else "error-metrics" if total_coverage < 50 else "metric-card"
        st.markdown(f"""
        <div class="{coverage_class}">
            <h4>📊 カバレッジ</h4>
            <h2>{total_coverage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 不足フィールド
    if missing_required:
        with st.expander("❌ 不足必須フィールド"):
            for field in missing_required:
                st.error(f"• {field}")

def display_extraction_summary(result):
    """抽出結果サマリーを表示"""
    clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
    
    st.markdown("### 📋 抽出結果サマリー")
    
    # フィールド統計
    total_fields = len(clean_result)
    successful_fields = len([v for v in clean_result.values() if not (isinstance(v, dict) and 'error' in v)])
    error_fields = total_fields - successful_fields
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 総フィールド", total_fields)
    with col2:
        st.metric("✅ 成功", successful_fields)
    with col3:
        st.metric("❌ エラー", error_fields)
    
    # 主要フィールドの内容チェック
    key_fields_status = {}
    key_fields = ["publicationIdentifier", "FrontPage", "Claims", "Description", "ChemicalStructureLibrary", "BiologicalSequenceLibrary"]
    
    for field in key_fields:
        if field in clean_result:
            value = clean_result[field]
            if isinstance(value, dict) and "error" in value:
                key_fields_status[field] = "❌ エラー"
            elif value is None:
                key_fields_status[field] = "⚪ 空"
            else:
                key_fields_status[field] = "✅ 成功"
        else:
            key_fields_status[field] = "➖ なし"
    
    with st.expander("🔍 主要フィールド状況"):
        for field, status in key_fields_status.items():
            st.write(f"**{field}**: {status}")

# メインUI
st.title("🚀 並列特許PDF構造化ツール")
st.markdown("**AI並列処理による高速特許データ抽出システム**")

if not EXTRACTOR_AVAILABLE:
    st.error("❌ 抽出エンジンが利用できません。システム管理者に連絡してください。")
    st.stop()

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 処理設定")
    
    # 並列処理設定
    st.subheader("🚀 並列処理")
    max_workers = st.slider("並列ワーカー数", 1, 32, 8, help="同時処理するフィールド数")
    
    processing_mode = st.radio(
        "処理モード",
        ["🔍 ドメイン検出のみ", "⚡ 完全並列抽出"],
        help="ドメイン検出のみは高速、完全抽出は詳細データを取得"
    )
    
    # AIモデル設定
    st.subheader("🤖 AIモデル")
    provider = st.selectbox("プロバイダー", list(DEFAULT_MODEL_OPTIONS.keys()))
    api_key = st.text_input(
        f"{provider} APIキー", 
        value=get_api_key_from_env(provider), 
        type="password",
        help="環境変数からの自動取得もサポート"
    )
    
    # モデル取得と選択
    if api_key:
        with st.spinner("🔄 利用可能なモデルを取得中..."):
            available_models = get_models_with_cache(provider, api_key)
        
        if available_models:
            st.success(f"✅ {len(available_models)}個のモデルを取得")
            model_name = st.selectbox("モデル", available_models)
            
            # モデル更新ボタン
            if st.button("🔄 モデル一覧を更新"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.error("❌ モデル取得に失敗しました")
            model_name = ""
    else:
        st.warning("⚠️ APIキーを入力してください")
        model_name = ""
    
    # スキーマ設定
    st.subheader("📋 JSONスキーマ")
    schema_type = st.radio("スキーマ設定", ["🎯 デフォルト", "📁 ファイル", "✏️ 直接入力"])
    
    schema = {}
    if schema_type == "📁 ファイル":
        uploaded_schema = st.file_uploader("スキーマファイル", type=["json"])
        if uploaded_schema:
            schema = load_schema(file_content=uploaded_schema.getvalue().decode("utf-8"))
    elif schema_type == "✏️ 直接入力":
        schema_text = st.text_area("JSONスキーマ", height=150, placeholder='{"type": "object", "properties": {...}}')
        if schema_text:
            try:
                schema = json.loads(schema_text)
                st.success("✅ スキーマ形式OK")
            except:
                st.error("❌ JSON形式エラー")
    else:
        # デフォルトスキーマ
        default_paths = ["default_schema.json", "schema.json", "./default_schema.json"]
        for path in default_paths:
            if os.path.exists(path):
                schema = load_schema(file_path=path)
                break
        if not schema:
            st.warning("⚠️ デフォルトスキーマが見つかりません")
    
    # 詳細設定
    with st.expander("⚙️ 詳細設定"):
        custom_prompt = st.text_area("カスタムプロンプト", height=80, placeholder="追加の抽出指示...")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, help="0.0=決定的, 1.0=創造的")
        max_tokens = st.selectbox("最大トークン数", [2048, 4096, 8192, 16384, 32768], index=2)

# メイン処理エリア
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 入力")
    
    uploaded_pdf = st.file_uploader(
        "特許PDFファイル", 
        type=["pdf"],
        help="日本語・英語の特許PDFに対応"
    )
    
    if uploaded_pdf:
        st.success(f"✅ ファイル: {uploaded_pdf.name} ({uploaded_pdf.size:,} bytes)")
        
        # ファイル情報
        with st.expander("📄 ファイル詳細"):
            st.write(f"**ファイル名**: {uploaded_pdf.name}")
            st.write(f"**サイズ**: {uploaded_pdf.size:,} bytes")
            st.write(f"**タイプ**: {uploaded_pdf.type}")
    
    # 処理ボタン
    process_enabled = uploaded_pdf and api_key and model_name
    
    if processing_mode == "🔍 ドメイン検出のみ":
        process_button = st.button(
            "🔍 ドメイン検出実行",
            disabled=not process_enabled,
            use_container_width=True,
            help="高速でドメインのみ検出"
        )
    else:
        process_button = st.button(
            f"⚡ 並列抽出開始 ({max_workers}workers)",
            disabled=not process_enabled,
            use_container_width=True,
            help="完全な並列データ抽出"
        )
    
    if not process_enabled:
        missing = []
        if not uploaded_pdf: missing.append("PDFファイル")
        if not api_key: missing.append("APIキー")
        if not model_name: missing.append("モデル選択")
        st.warning(f"⚠️ 不足: {', '.join(missing)}")

with col2:
    st.header("📤 出力")
    
    if process_button and uploaded_pdf and api_key and model_name:
        pdf_path = save_upload_file(uploaded_pdf)
        
        if pdf_path:
            # プログレスバー
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if processing_mode == "🔍 ドメイン検出のみ":
                    # ドメイン検出のみ
                    status_text.text("🔍 ドメイン検出中...")
                    progress_bar.progress(50)
                    
                    result = detect_domain_only(pdf_path, model_name, api_key)
                    progress_bar.progress(100)
                    
                    if "error" in result:
                        st.error(f"❌ 検出エラー: {result['error']}")
                    else:
                        st.success("✅ ドメイン検出完了！")
                        display_domain_info(result)
                        
                        # 結果表示
                        st.json(result)
                        
                        # ダウンロード
                        json_str = json.dumps(result, indent=2, ensure_ascii=False)
                        st.download_button(
                            "📥 ドメイン情報ダウンロード",
                            data=json_str.encode("utf-8"),
                            file_name=f"{Path(uploaded_pdf.name).stem}_domain.json",
                            mime="application/json",
                            use_container_width=True
                        )
                
                else:
                    # 完全並列抽出
                    status_text.text(f"⚡ {model_name}で並列処理中...")
                    progress_bar.progress(25)
                    
                    start_time = time.time()
                    result = process_pdf_parallel(
                        pdf_path=pdf_path,
                        model_name=model_name,
                        api_key=api_key,
                        schema=schema,
                        prompt=custom_prompt if custom_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_workers=max_workers
                    )
                    
                    progress_bar.progress(100)
                    
                    if "error" in result:
                        st.error(f"❌ 処理エラー: {result['error']}")
                    else:
                        st.success("✅ 並列処理完了！")
                        
                        # ドメイン情報表示
                        if "_processing_metadata" in result:
                            domain = result["_processing_metadata"].get("domain_detected")
                            if domain:
                                display_domain_info({"primary_domain": domain})
                        
                        # パフォーマンス指標
                        if "_processing_metadata" in result:
                            display_performance_metrics(result["_processing_metadata"])
                        
                        # 抽出結果サマリー
                        display_extraction_summary(result)
                        
                        # スキーマ検証
                        if schema:
                            display_validation_results(result, schema)
                        
                        # 結果表示
                        clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
                        
                        with st.expander("📋 完全な抽出結果"):
                            st.json(clean_result)
                        
                        # ダウンロードボタン
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            json_str = json.dumps(clean_result, indent=2, ensure_ascii=False)
                            output_filename = f"{Path(uploaded_pdf.name).stem}_extracted.json"
                            
                            st.download_button(
                                "📥 抽出結果ダウンロード",
                                data=json_str.encode("utf-8"),
                                file_name=output_filename,
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # メタデータ付き完全版
                            full_json_str = json.dumps(result, indent=2, ensure_ascii=False)
                            metadata_filename = f"{Path(uploaded_pdf.name).stem}_full.json"
                            
                            st.download_button(
                                "📊 メタデータ付きダウンロード",
                                data=full_json_str.encode("utf-8"),
                                file_name=metadata_filename,
                                mime="application/json",
                                use_container_width=True
                            )
                
            finally:
                # 一時ファイル削除
                try:
                    os.remove(pdf_path)
                except:
                    pass
                
                status_text.empty()
                progress_bar.empty()
    
    else:
        # 待機状態の情報表示
        st.info("📋 PDFをアップロードして処理を開始してください")
        
        # 使用例表示
        with st.expander("💡 出力例"):
            if processing_mode == "🔍 ドメイン検出のみ":
                example = {
                    "primary_domain": "chemical",
                    "structural_elements": ["chemical_structures", "tables", "figures"],
                    "extraction_priorities": ["ChemicalStructureLibrary", "Claims", "Description"],
                    "complexity_level": "high"
                }
            else:
                example = {
                    "publicationIdentifier": "WO2024123456A1",
                    "FrontPage": {
                        "PublicationData": {"PublicationNumber": "WO2024123456A1"},
                        "Abstract": {"Paragraph": [{"content": "AI特許抽出システム..."}]}
                    },
                    "Claims": {"Claim": [{"id": "1", "Text": {"content": "AIを使用する方法..."}}]},
                    "ChemicalStructureLibrary": {"Compound": []}
                }
            st.json(example)

# フッター情報
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("**🚀 機能**")
    st.markdown("• 並列処理による高速抽出")
    st.markdown("• ドメイン自動検出")
    st.markdown("• スキーマ準拠")

with col_info2:
    st.markdown("**📊 対応ドメイン**") 
    st.markdown("• 化学・医薬")
    st.markdown("• バイオテクノロジー")
    st.markdown("• 機械・電子")

with col_info3:
    st.markdown("**⚡ パフォーマンス**")
    st.markdown("• 最大32並列ワーカー")
    st.markdown("• リアルタイム進捗表示")
    st.markdown("• 詳細統計レポート")

st.markdown(
    "<div style='text-align: center; color: #666; margin-top: 20px;'>"
    "🔬 並列特許PDF構造化ツール - AI駆動高速データ抽出システム"
    "</div>", 
    unsafe_allow_html=True
)