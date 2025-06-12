import os
import json
import base64
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("patent-extractor")

class PatentExtractor:
    """特許PDFから構造化JSONを抽出するライブラリ（並列処理専用）"""
    
    def __init__(
        self, 
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_workers: int = 8
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
            max_workers: 並列処理数（デフォルト8に増加）
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(self._get_env_var_name(model_name))
        self.schema = json_schema or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        
        # デフォルトプロンプト
        self.prompt = user_prompt or """
        添付のPDFは特許文書です。提供されたJSONスキーマに従って情報を構造化してください。
        フロントページ情報、請求項、詳細な説明など、スキーマに記載されたすべてのセクションを含めてください。
        スキーマ構造に正確に従い、すべての見出し、小見出しを抽出し、特許文書の階層構造を維持してください。
        化学式、図、表についてはそれらの識別子と参照情報を含めてください。
        """
        
        # フィールド定義（依存関係を最小化して並列度を最大化）
        self.field_definitions = {
            # Wave 1: 基本情報（完全並列）
            "FrontPage": {
                "description": "フロントページ情報（出願情報、発明者、出願人、分類、要約）を抽出",
                "wave": 1,
                "dependencies": []
            },
            "Claims": {
                "description": "特許請求の範囲を抽出",
                "wave": 1,
                "dependencies": []
            },
            "Description": {
                "description": "技術分野、背景技術、発明の概要、発明の詳細な説明を抽出",
                "wave": 1,
                "dependencies": []
            },
            # Wave 2: 専門データ（基本情報を参照するが並列実行）
            "ChemicalStructureLibrary": {
                "description": "特許中のすべての化学構造を抽出",
                "wave": 2,
                "dependencies": ["Claims", "Description"]
            },
            "BiologicalSequenceLibrary": {
                "description": "特許中のすべての生物学的配列を抽出",
                "wave": 2,
                "dependencies": ["Claims", "Description"]
            },
            "Tables": {
                "description": "特許中のすべての表を抽出",
                "wave": 2,
                "dependencies": ["Description"]
            },
            "Figures": {
                "description": "特許中のすべての図を抽出",
                "wave": 2,
                "dependencies": ["Description"]
            },
            "IndustrialApplicability": {
                "description": "産業上の利用可能性を抽出",
                "wave": 2,
                "dependencies": []
            },
            # Wave 3: 高度な分析（Wave 2の結果を活用）
            "Examples": {
                "description": "実施例を抽出",
                "wave": 3,
                "dependencies": ["Description", "ChemicalStructureLibrary", "Tables"]
            },
            "InternationalSearchReport": {
                "description": "国際調査報告を抽出",
                "wave": 3,
                "dependencies": ["FrontPage"]
            },
            "PatentFamilyInformation": {
                "description": "特許ファミリー情報を抽出",
                "wave": 3,
                "dependencies": ["FrontPage"]
            },
            "FrontPageContinuation": {
                "description": "フロントページ続き情報を抽出",
                "wave": 3,
                "dependencies": ["FrontPage"]
            }
        }
        
        # クライアント初期化
        self._init_client()
        
        # 共有データ用のロック（高速化のためReadWriteLock風の実装）
        self._data_lock = threading.RLock()
        self._shared_data = {}
        
        # パフォーマンス測定
        self._timing_data = {}
    
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
    
    def _get_field_schema(self, field_name: str) -> Dict[str, Any]:
        """特定フィールドのスキーマを取得（プロンプト用）"""
        if field_name in self.schema.get("properties", {}):
            field_property = self.schema["properties"][field_name]
            
            # $refを展開してdefinitionsをインライン化
            expanded_property = self._expand_definitions(field_property)
            
            return {
                "field_name": field_name,
                "schema": expanded_property
            }
        
        return {
            "field_name": field_name,
            "schema": {"type": "object"}
        }
    
    def _expand_definitions(self, schema_part: Any) -> Any:
        """$refを展開してdefinitionsをインライン化（プロンプト用）"""
        if isinstance(schema_part, dict):
            if "$ref" in schema_part:
                ref_path = schema_part["$ref"]
                if ref_path.startswith("#/definitions/"):
                    def_name = ref_path.replace("#/definitions/", "")
                    definitions = self.schema.get("definitions", {})
                    if def_name in definitions:
                        # 循環参照を避けるため、深度制限
                        if not hasattr(self, '_expansion_depth'):
                            self._expansion_depth = 0
                        
                        if self._expansion_depth >= 3:
                            return {"type": "object", "description": f"Reference to {def_name}"}
                        
                        self._expansion_depth += 1
                        result = self._expand_definitions(definitions[def_name])
                        self._expansion_depth -= 1
                        return result
                return schema_part
            else:
                expanded = {}
                for key, value in schema_part.items():
                    expanded[key] = self._expand_definitions(value)
                return expanded
        elif isinstance(schema_part, list):
            return [self._expand_definitions(item) for item in schema_part]
        else:
            return schema_part
    
    def _create_schema_prompt(self, field_name: str, schema_info: Dict[str, Any]) -> str:
        """スキーマ情報からプロンプト用の説明を生成"""
        schema = schema_info.get("schema", {})
        
        # スキーマから構造の説明を生成
        def describe_schema(s, indent=0):
            if isinstance(s, dict):
                if s.get("type") == "object":
                    props = s.get("properties", {})
                    if props:
                        lines = []
                        for prop_name, prop_schema in props.items():
                            required = " (required)" if prop_name in s.get("required", []) else ""
                            lines.append("  " * indent + f"- {prop_name}{required}: {describe_schema(prop_schema, indent+1)}")
                        return "object with properties:\n" + "\n".join(lines)
                    else:
                        return "object"
                elif s.get("type") == "array":
                    items = s.get("items", {})
                    return f"array of {describe_schema(items, indent)}"
                elif s.get("type") in ["string", "integer", "number", "boolean"]:
                    enum_values = s.get("enum")
                    if enum_values:
                        return f"{s['type']} (one of: {', '.join(map(str, enum_values))})"
                    return s["type"]
                else:
                    return s.get("type", "unknown")
            return str(s)
        
        schema_description = describe_schema(schema)
        
        return f"""
Extract the {field_name} section and return it in the following JSON structure:

{field_name}: {schema_description}

Important:
- Return valid JSON only
- Follow the exact structure shown above
- Include all available information from the PDF
- Use null for missing values
- Ensure proper JSON formatting
"""
    
    def _create_field_prompt(self, field_name: str, dependency_context: str = "") -> str:
        """フィールド専用のプロンプトを作成"""
        field_prompts = {
            "FrontPage": """
            PDFの最初のページ（フロントページ）から以下の情報を抽出してください：
            - 公開番号、公開日、出願番号、出願日
            - 発明者情報（名前、住所）
            - 出願人情報（名前、住所）
            - 国際特許分類（IPC）
            - 要約（Abstract）
            - 優先権データがあれば含める
            正確性を重視し、フロントページのレイアウトに従って情報を抽出してください。
            """,
            "Claims": """
            特許請求の範囲（Claims）セクションから全ての請求項を抽出してください。
            各請求項には番号とテキストを含めてください。
            化学構造や表への参照も含めてください。
            独立請求項と従属請求項の関係も明確にしてください。
            """,
            "Description": """
            発明の詳細な説明から以下のセクションを抽出してください：
            - 技術分野（Technical Field）
            - 背景技術（Background Art）
            - 発明の概要（Summary of Invention）
            - 発明の詳細な説明（Detailed Description）
            各セクションの構造と内容を維持し、階層構造を正確に抽出してください。
            """,
            "ChemicalStructureLibrary": """
            特許文書全体から化学構造、化学式、化合物を抽出してください。
            化合物番号、SMILES、分子式、化学名などの情報を含めてください。
            化学構造画像への参照も含めてください。
            各化合物の用途や特性も記載があれば含めてください。
            """,
            "BiologicalSequenceLibrary": """
            特許文書全体から生物学的配列（タンパク質、DNA、RNA）を抽出してください。
            配列ID（SEQ ID NO）、配列情報、生物種、機能情報を含めてください。
            配列リストセクションがあれば優先的に参照してください。
            """,
            "Tables": """
            特許文書全体から表を抽出してください。
            表の構造、ヘッダー、データ、キャプションを完全に抽出してください。
            表番号と位置情報も含めてください。
            数値データの単位や注釈も忘れずに含めてください。
            """,
            "Figures": """
            特許文書全体から図を抽出してください。
            図番号、キャプション、参照情報を含めてください。
            図の説明文も可能な限り抽出してください。
            """,
            "Examples": """
            実施例セクションから全ての実施例を抽出してください。
            実施例番号、タイトル、詳細な説明を含めてください。
            化学構造、表、図への参照も含めてください。
            実験条件、結果、考察も含めてください。
            """,
            "IndustrialApplicability": """
            産業上の利用可能性に関するセクションを抽出してください。
            適用分野、利用方法、産業への影響を含めてください。
            """,
            "InternationalSearchReport": """
            国際調査報告書の情報を抽出してください。
            引用文献、調査分野、見解書の内容を含めてください。
            """,
            "PatentFamilyInformation": """
            特許ファミリー情報を抽出してください。
            関連特許、ファミリー構成、優先権情報を含めてください。
            """,
            "FrontPageContinuation": """
            フロントページの続き情報を抽出してください。
            指定国、F-Term、その他の分類情報を含めてください。
            """
        }
        
        base_prompt = field_prompts.get(field_name, f"{field_name}を抽出してください。")
        
        if dependency_context:
            return f"{base_prompt}\n\n{dependency_context}"
        
        return base_prompt
    
    def process_patent_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        特許PDFを並列処理で高速抽出
        
        Args:
            pdf_path: PDFファイルのパス
            
        Returns:
            構造化された特許情報を含む辞書
        """
        start_time = time.time()
        logger.info(f"Starting parallel processing of PDF: {pdf_path}")
        
        try:
            result = self._process_parallel_waves(pdf_path)
            
            # 公開番号がない場合、ファイル名から設定
            if "publicationIdentifier" not in result:
                result["publicationIdentifier"] = Path(pdf_path).stem
            
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            
            # パフォーマンス情報を追加
            result["_processing_info"] = {
                "total_time_seconds": total_time,
                "field_timing": self._timing_data,
                "parallel_workers": self.max_workers,
                "model_used": self.model_name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                "error": str(e),
                "publicationIdentifier": Path(pdf_path).stem
            }
    
    def _process_parallel_waves(self, pdf_path: str) -> Dict[str, Any]:
        """Wave単位での並列処理（最大並列度を実現）"""
        logger.info("Starting wave-based parallel processing")
        
        # Waveごとにフィールドをグループ化
        wave_groups = {}
        for field_name, field_info in self.field_definitions.items():
            wave = field_info.get("wave", 999)
            if wave not in wave_groups:
                wave_groups[wave] = []
            wave_groups[wave].append(field_name)
        
        final_result = {}
        
        # Wave順に処理（各Wave内は完全並列）
        for wave in sorted(wave_groups.keys()):
            fields_in_wave = wave_groups[wave]
            wave_start_time = time.time()
            
            logger.info(f"Processing Wave {wave} with {len(fields_in_wave)} fields: {fields_in_wave}")
            
            # Wave内のすべてのフィールドを並列処理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_field = {
                    executor.submit(self._extract_field_with_timing, pdf_path, field_name): field_name
                    for field_name in fields_in_wave
                }
                
                # 結果を収集
                wave_results = {}
                for future in as_completed(future_to_field):
                    field_name = future_to_field[future]
                    try:
                        field_result = future.result()
                        if field_result and field_name in field_result:
                            wave_results[field_name] = field_result[field_name]
                            logger.info(f"✓ Wave {wave}: {field_name} completed")
                        else:
                            logger.warning(f"✗ Wave {wave}: {field_name} returned no data")
                            wave_results[field_name] = None
                    except Exception as e:
                        logger.error(f"✗ Wave {wave}: {field_name} failed: {e}")
                        wave_results[field_name] = {"error": str(e)}
                
                # Wave結果をマージ
                final_result.update(wave_results)
                
                # 共有データを一括更新（次Waveの依存関係のため）
                with self._data_lock:
                    self._shared_data.update(wave_results)
                
                wave_time = time.time() - wave_start_time
                logger.info(f"Wave {wave} completed in {wave_time:.2f} seconds")
        
        return final_result
    
    def _extract_field_with_timing(self, pdf_path: str, field_name: str) -> Dict[str, Any]:
        """タイミング測定付きフィールド抽出"""
        start_time = time.time()
        try:
            result = self._extract_field(pdf_path, field_name)
            
            processing_time = time.time() - start_time
            self._timing_data[field_name] = processing_time
            
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            self._timing_data[field_name] = processing_time
            raise e
    
    def _extract_field(self, pdf_path: str, field_name: str) -> Dict[str, Any]:
        """特定フィールドを抽出（プロンプトベース）"""
        schema_info = self._get_field_schema(field_name)
        field_prompt = self._create_field_prompt(field_name)
        schema_prompt = self._create_schema_prompt(field_name, schema_info)
        
        # 依存関係のコンテキストを高速取得
        dependency_context = self._get_dependency_context(field_name)
        
        full_prompt = f"{field_prompt}\n\n{schema_prompt}"
        if dependency_context:
            full_prompt += f"\n\n{dependency_context}"
        
        # モデルタイプに応じた処理（すべてプロンプトベース）
        if "gemini" in self.model_name.lower():
            return self._process_field_with_gemini_prompt(pdf_path, full_prompt)
        elif "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            return self._process_field_with_openai_prompt(pdf_path, full_prompt)
        elif "claude" in self.model_name.lower():
            return self._process_field_with_anthropic_prompt(pdf_path, full_prompt)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _get_dependency_context(self, field_name: str) -> str:
        """依存関係コンテキストを高速取得"""
        dependencies = self.field_definitions.get(field_name, {}).get("dependencies", [])
        
        if not dependencies:
            return ""
        
        # 読み込みロック（高速化）
        with self._data_lock:
            available_contexts = []
            for dep in dependencies:
                if dep in self._shared_data and self._shared_data[dep] is not None:
                    # 要約版のコンテキストを作成（大容量データを避ける）
                    dep_data = self._shared_data[dep]
                    if isinstance(dep_data, dict):
                        summary = self._create_context_summary(dep_data, dep)
                        available_contexts.append(f"【{dep}参考情報】\n{summary}")
            
            if available_contexts:
                return "\n\n以下の関連情報を参考にしてください：\n" + "\n".join(available_contexts)
        
        return ""
    
    def _create_context_summary(self, data: Dict[str, Any], field_name: str) -> str:
        """コンテキスト要約を作成（大容量データの問題を回避）"""
        if field_name == "Claims":
            # 請求項の要約
            claims = data.get("Claim", [])
            return f"請求項数: {len(claims)}項目"
        
        elif field_name == "Description":
            # 説明の要約
            sections = []
            for section_name in ["TechnicalField", "BackgroundArt", "SummaryOfInvention"]:
                if section_name in data:
                    sections.append(section_name)
            return f"含まれるセクション: {', '.join(sections)}"
        
        else:
            # その他の一般的な要約
            return f"データ構造: {list(data.keys())[:5]}"  # 最初の5つのキーのみ
    
    def _process_field_with_gemini_prompt(self, pdf_path: str, prompt: str) -> Dict[str, Any]:
        """Geminiでフィールドを処理（プロンプトベース）"""
        model = self.client.GenerativeModel(
            model_name=self.model_name,
            system_instruction="You are a patent analysis assistant. Extract specific field information from patent PDFs and return valid JSON only."
        )
        
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        response = model.generate_content(
            contents=[
                f"{prompt}\n\nReturn only valid JSON, no explanations or markdown.",
                {"mime_type": "application/pdf", "data": pdf_data}
            ],
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        
        return self._extract_json_from_text(response.text)
    
    def _process_field_with_openai_prompt(self, pdf_path: str, prompt: str) -> Dict[str, Any]:
        """OpenAIでフィールドを処理（プロンプトベース）"""
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a patent analysis assistant. Extract specific field information from patents and return valid JSON only. Do not include explanations or markdown formatting."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"{prompt}\n\nReturn only valid JSON."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return self._extract_json_from_text(response.choices[0].message.content)
    
    def _process_field_with_anthropic_prompt(self, pdf_path: str, prompt: str) -> Dict[str, Any]:
        """Anthropicでフィールドを処理（プロンプトベース）"""
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.messages.create(
            model=self.model_name,
            system="You are a patent analysis assistant. Extract specific field information from patent PDFs and return valid JSON only. Do not include explanations or markdown formatting.",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}\n\nReturn only valid JSON that matches the specified structure."
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
        )
        
        return self._extract_json_from_text(response.content[0].text)
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """テキストからJSONを抽出（Anthropic用）"""
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
    """コマンドライン実行用のメイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract structured information from patent PDFs with high-speed parallel processing')
    parser.add_argument('pdf_path', help='Path to the patent PDF file')
    parser.add_argument('--model', default='gemini-1.5-pro', help='AI model to use')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--schema', help='Path to the JSON schema file')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--prompt', help='Custom prompt')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum tokens to generate')
    parser.add_argument('--max-workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # スキーマ読み込み
    schema = None
    if args.schema:
        with open(args.schema, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    
    # エクストラクタの初期化と実行
    extractor = PatentExtractor(
        model_name=args.model,
        api_key=args.api_key,
        json_schema=schema,
        user_prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers
    )
    
    # PDFの処理
    result = extractor.process_patent_pdf(args.pdf_path)
    
    # 結果の出力
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Output saved to {args.output}")
        
        # パフォーマンス情報の表示
        if "_processing_info" in result:
            info = result["_processing_info"]
            print(f"\nPerformance Summary:")
            print(f"Total Time: {info['total_time_seconds']:.2f} seconds")
            print(f"Parallel Workers: {info['parallel_workers']}")
            print(f"Model: {info['model_used']}")
            print(f"\nField Processing Times:")
            for field, time_taken in info['field_timing'].items():
                print(f"  {field}: {time_taken:.2f}s")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()