# 特許PDF構造化ツール - Streamlitアプリ

特許PDFからマルチモーダル生成AI（Gemini、GPT、Claude）を使用して構造化JSONを抽出するStreamlitウェブアプリケーションです。

![App Screenshot](https://via.placeholder.com/800x400?text=Patent+PDF+Extractor+App)

## 🚀 機能

- **特許PDFアップロード**: ドラッグ&ドロップでPDFを簡単アップロード
- **マルチAIサポート**: Google Gemini、OpenAI GPT、Anthropic Claudeに対応
- **動的モデル選択**: APIから最新の利用可能なモデル一覧を自動取得
- **カスタムモデル**: 任意のモデル名を直接入力可能
- **JSONスキーマ**: カスタムスキーマのアップロード・直接入力・デフォルト使用
- **AI設定調整**: temperature、max_tokensのリアルタイム調整
- **結果ダウンロード**: 構造化されたJSONの即座ダウンロード
- **レスポンシブUI**: PC・タブレット・スマホ対応

## 🎥 デモ

### ライブデモ
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)

### 使用例

1. **AIプロバイダー選択** → Google Gemini / OpenAI / Anthropic
2. **APIキー入力** → 自動的に最新モデル一覧を取得
3. **モデル選択** → 利用可能なモデルから選択 or カスタム入力
4. **PDF特許アップロード** → ファイル選択またはドラッグ&ドロップ
5. **処理実行** → ワンクリックで構造化JSON生成
6. **結果ダウンロード** → JSON形式で即座にダウンロード

## 🛠️ インストール・実行方法

### 方法1: Pythonで直接実行

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/patent-pdf-streamlit-app.git
cd patent-pdf-streamlit-app

# 依存関係をインストール
pip install -r requirements.txt

# アプリを実行
streamlit run app.py
```

### 方法2: Dockerで実行

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/patent-pdf-streamlit-app.git
cd patent-pdf-streamlit-app

# Dockerで実行
docker-compose up -d

# ブラウザでアクセス
# http://localhost:8501
```

### 方法3: Streamlit Cloudにデプロイ

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)

1. このリポジトリをフォーク
2. [Streamlit Cloud](https://share.streamlit.io/) にログイン
3. 「New app」→ このリポジトリを選択
4. `app.py` を指定してデプロイ

## 🔧 設定

### 環境変数

APIキーは以下の環境変数で設定できます：

```bash
# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Streamlit設定

`.streamlit/config.toml` でアプリをカスタマイズ：

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 50
```

## 📚 サポートされるAIモデル

### Google Gemini
- `gemini-1.5-pro` - 高性能モデル
- `gemini-1.5-flash` - 高速モデル  
- `gemini-2.0-flash-exp` - 実験的最新モデル

### OpenAI
- `gpt-4o` - 最新マルチモーダルモデル
- `gpt-4-vision-preview` - Vision対応
- `gpt-4-turbo` - 高速処理

### Anthropic
- `claude-3-5-sonnet-20241022` - 最新Sonnet
- `claude-3-opus-20240229` - 最高性能
- `claude-3-sonnet-20240229` - バランス型

## 🎛️ カスタマイズ

### カスタムJSONスキーマ

以下の3つの方法でスキーマを設定可能：

1. **デフォルトスキーマ**: そのまま使用
2. **ファイルアップロード**: `.json` ファイルをアップロード
3. **直接入力**: テキストエリアにJSONを直接入力

### AI生成パラメータ

- **Temperature** (0.0-1.0): 出力の多様性を制御
- **Max Tokens** (1024-32768): 出力の最大長を制御

## 🐳 Docker設定

### 単体実行
```bash
docker build -t patent-pdf-app .
docker run -p 8501:8501 patent-pdf-app
```

### 環境変数付き実行
```bash
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY="your-key" \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  patent-pdf-app
```

## 🤝 コントリビューション

バグ報告、機能リクエスト、プルリクエストを歓迎します！

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## 🐛 トラブルシューティング

### よくある問題

**Q: モデル一覧が取得できない**
- APIキーが正しいか確認
- ネットワーク接続を確認
- 「🔄 モデル一覧を更新」ボタンを試す

**Q: PDFの処理が失敗する**
- PDFファイルが破損していないか確認
- 選択したモデルがPDF処理に対応しているか確認
- APIの使用制限に達していないか確認

**Q: アプリが起動しない**
```bash
# 依存関係を再インストール
pip install --upgrade -r requirements.txt

# Streamlitを最新版に更新
pip install --upgrade streamlit
```

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 🙏 謝辞

- [Streamlit](https://streamlit.io/) - 素晴らしいウェブアプリフレームワーク
- [Google Generative AI](https://ai.google.dev/) - Geminiモデル
- [OpenAI](https://openai.com/) - GPTモデル  
- [Anthropic](https://anthropic.com/) - Claudeモデル

---

⭐ このプロジェクトが役に立ったら、スターをつけていただけると嬉しいです！
