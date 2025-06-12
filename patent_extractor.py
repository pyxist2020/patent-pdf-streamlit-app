import os
import json
import base64
import logging
from typing import Dict, Any, Optional, Iterator, Callable
from pathlib import Path

import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streaming-patent-extractor")

class StreamingPatentExtractor:
    """特許PDFから構造化JSONを抽出するライブラリ（ストリーミング専用）"""
    
    def __init__(
        self, 
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        """
        初期化
        
        Args:
            model_name: 使用する生成AIモデル名
            api_key: API認証キー
            json_schema: JSONスキーマ（辞書形式）
            user_prompt: カスタムプロンプト
            temperature: 生成AI の temperature 設定値 (0.0〜1.0)
            max_tokens: 生成AI の最大トークン数
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(self._get_env_var_name(model_name))
        self.schema = json_schema or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # デフォルトプロンプト
        self.prompt = user_prompt or """
        添付のPDFは特許文書です。提供されたJSONスキーマに従って情報を構造化してください。
        フロントページ情報、請求項、詳細な説明など、スキーマに記載されたすべてのセクションを含めてください。
        スキーマ構造に正確に従い、すべての見出し、小見出しを抽出し、特許文書の階層構造を維持してください。
        化学式、図、表についてはそれらの識別子と参照情報を含めてください。
        結果は有効なJSONとして出力し、マークダウン形式や説明文は含めないでください。
        """
        
        # クライアント初期化
        self._init_client()
    
    def _get_env_var_name(self, model_name: str) -> str:
        """モデル名に応じた環境変数名を取得"""
        if "gemini" in model_name.lower():
            return "GOOGLE_API_KEY"
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            return "OPENAI_API_KEY"
        elif "claude" in model_name.lower():
            return "ANTHROPIC_API_KEY"
        return "API_KEY"
    
    def _init_client(self):
        """AIクライアントを初期化"""
        if not self.api_key:
            raise ValueError(f"API key not provided for model {self.model_name}")
        
        if "gemini" in self.model_name.lower():
            genai.configure(api_key=self.api_key)
            self.client = genai
            logger.info(f"Initialized Google Generative AI client with model {self.model_name}")
        elif "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model {self.model_name}")
        elif "claude" in self.model_name.lower():
            self.client = Anthropic(api_key=self.api_key)
            logger.info(f"Initialized Anthropic client with model {self.model_name}")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def process_patent_pdf_stream(
        self, 
        pdf_path: str, 
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """
        特許PDFを処理し構造化情報をストリーミング出力
        
        Args:
            pdf_path: PDFファイルのパス
            callback: 各チャンクを受信したときに呼び出されるコールバック関数
            
        Yields:
            生成されたテキストのチャンク
        """
        logger.info(f"Processing PDF with streaming: {pdf_path}")
        
        try:
            # モデルタイプに応じた処理
            if "gemini" in self.model_name.lower():
                yield from self._process_with_gemini_stream(pdf_path, callback)
            elif "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
                yield from self._process_with_openai_stream(pdf_path, callback)
            elif "claude" in self.model_name.lower():
                yield from self._process_with_anthropic_stream(pdf_path, callback)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            error_chunk = json.dumps({
                "error": str(e),
                "publicationIdentifier": Path(pdf_path).stem
            })
            yield error_chunk
    
    def _process_with_gemini_stream(
        self, 
        pdf_path: str, 
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """Geminiモデルでストリーミング処理"""
        # 生成設定
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # スキーマをJSONとして整形
        schema_json = json.dumps(self.schema, indent=2)
        
        # モデル初期化
        model = self.client.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        # プロンプト準備
        full_prompt = f"""
        {self.prompt}
        
        JSON SCHEMA:
        ```json
        {schema_json}
        ```
        
        Extract information from the patent PDF and format it according to this schema.
        Return ONLY valid JSON that conforms to the schema without any explanations or markdown formatting.
        """
        
        # PDFデータ読み込み
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        # ストリーミング生成
        response = model.generate_content(
            contents=[
                full_prompt,
                {"mime_type": "application/pdf", "data": pdf_data}
            ],
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                if callback:
                    callback(chunk.text)
                yield chunk.text
    
    def _process_with_openai_stream(
        self, 
        pdf_path: str, 
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """OpenAIモデルでストリーミング処理"""
        # スキーマをJSONとして整形
        schema_json = json.dumps(self.schema, indent=2)
        
        # PDFをbase64エンコード
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
        
        # OpenAI API ストリーミング呼び出し
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a patent analysis assistant. Extract structured information from patents according to the provided JSON schema."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            {self.prompt}
                            
                            JSON SCHEMA:
                            ```json
                            {schema_json}
                            ```
                            
                            Extract information from the patent PDF and format it according to this schema.
                            Return ONLY valid JSON that conforms to the schema without any explanations or markdown formatting.
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if callback:
                    callback(content)
                yield content
    
    def _process_with_anthropic_stream(
        self, 
        pdf_path: str, 
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """Anthropicモデルでストリーミング処理"""
        # スキーマをJSONとして整形
        schema_json = json.dumps(self.schema, indent=2)
        
        # PDFをbase64エンコード
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Anthropic API ストリーミング呼び出し
        with self.client.messages.stream(
            model=self.model_name,
            system="You are a patent analysis assistant. Extract structured information from the patent PDF according to the provided JSON schema.",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            {self.prompt}
                            
                            JSON SCHEMA:
                            ```json
                            {schema_json}
                            ```
                            
                            Extract information from the patent PDF and format it according to this schema.
                            Return ONLY valid JSON that conforms to the schema without any explanations or markdown formatting.
                            """
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data
                            }
                        }
                    ]
                }
            ]
        ) as stream:
            for text in stream.text_stream:
                if callback:
                    callback(text)
                yield text
    
    def extract_json_from_stream(self, stream: Iterator[str]) -> Dict[str, Any]:
        """
        ストリームからJSONを抽出
        
        Args:
            stream: テキストストリーム
            
        Returns:
            抽出されたJSON辞書
        """
        full_text = "".join(stream)
        return self._extract_json_from_text(full_text)
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """テキストからJSONを抽出"""
        try:
            if "```json" in text:
                json_block = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_block)
            
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                return json.loads(json_text)
            
            raise ValueError("No valid JSON found in response")
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return {"error": "Failed to parse JSON from AI response"}


def main():
    """コマンドライン実行用のメイン関数（ストリーミング専用）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract structured information from patent PDFs with streaming')
    parser.add_argument('pdf_path', help='Path to the patent PDF file')
    parser.add_argument('--model', default='gemini-1.5-pro', help='AI model to use')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--schema', help='Path to the JSON schema file')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--prompt', help='Custom prompt')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum tokens to generate')
    parser.add_argument('--quiet', action='store_true', help='Suppress real-time output (only save to file)')
    
    args = parser.parse_args()
    
    # スキーマ読み込み
    schema = None
    if args.schema:
        with open(args.schema, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    
    # エクストラクタの初期化
    extractor = StreamingPatentExtractor(
        model_name=args.model,
        api_key=args.api_key,
        json_schema=schema,
        user_prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # ストリーミング処理
    print("Starting streaming extraction...")
    full_output = ""
    
    def progress_callback(chunk: str):
        """進捗表示用コールバック"""
        if not args.quiet:
            print(chunk, end='', flush=True)
    
    try:
        for chunk in extractor.process_patent_pdf_stream(args.pdf_path, progress_callback):
            full_output += chunk
        
        if not args.quiet:
            print("\n\nExtraction completed!")
        
        # 結果の保存
        if args.output:
            try:
                result_json = extractor._extract_json_from_text(full_output)
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)
                print(f"Output saved to {args.output}")
            except Exception as e:
                print(f"Error saving JSON: {e}")
                # Raw output を保存
                with open(args.output + '.raw', 'w', encoding='utf-8') as f:
                    f.write(full_output)
                print(f"Raw output saved to {args.output}.raw")
        else:
            # 標準出力にJSON結果を表示
            try:
                result_json = extractor._extract_json_from_text(full_output)
                print("\n" + "="*50)
                print("EXTRACTED JSON:")
                print("="*50)
                print(json.dumps(result_json, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"\nError parsing final JSON: {e}")
                print("Raw output:")
                print(full_output)
                
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user")
        if args.output and full_output:
            # 部分的な結果でも保存
            with open(args.output + '.partial', 'w', encoding='utf-8') as f:
                f.write(full_output)
            print(f"Partial output saved to {args.output}.partial")
    except Exception as e:
        print(f"\nError during extraction: {e}")


if __name__ == "__main__":
    main()
