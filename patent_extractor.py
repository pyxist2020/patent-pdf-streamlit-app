import os
import json
import base64
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

# Log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("parallel-patent-extractor")

class ProcessingWave(Enum):
    """Processing wave definitions for parallel execution"""
    WAVE_1_CORE = 1      # Core required fields
    WAVE_2_STRUCTURES = 2 # Structure libraries
    WAVE_3_METADATA = 3   # Metadata and references
    WAVE_4_VALIDATION = 4 # Validation and post-processing

@dataclass
class FieldConfig:
    """Configuration for each extractable field"""
    name: str
    wave: ProcessingWave
    dependencies: List[str] = field(default_factory=list)
    schema_ref: str = ""
    domain_specific: bool = False
    required: bool = False
    prompt_template: str = ""

class PatentExtractor:
    """Parallel patent extractor with full schema compliance and universal domain support"""
    
    def __init__(
        self, 
        model_name: str = "claude-3-sonnet-20241022",
        api_key: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_workers: int = 8
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(self._get_env_var_name(model_name))
        self.schema = json_schema or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        
        # Initialize field configurations based on schema
        self.field_configs = self._initialize_field_configs()
        
        # Initialize client
        self._init_client()
        
        # Shared data and locks for parallel processing
        self._data_lock = threading.RLock()
        self._shared_data = {}
        self._timing_data = {}
        self._processing_stats = {
            "total_fields": 0,
            "successful_fields": 0,
            "failed_fields": 0,
            "processing_times": {}
        }
        
        # Domain detection cache
        self._domain_cache = {}
    
    def _get_env_var_name(self, model_name: str) -> str:
        """Get environment variable name according to model name"""
        if "gemini" in model_name.lower():
            return "GOOGLE_API_KEY"
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            return "OPENAI_API_KEY"
        elif "claude" in model_name.lower():
            return "ANTHROPIC_API_KEY"
        return "API_KEY"
    
    def _init_client(self):
        """Initialize AI client"""
        if not self.api_key:
            raise ValueError(f"API key not provided for model {self.model_name}")
        
        if "gemini" in self.model_name.lower():
            genai.configure(api_key=self.api_key)
            self.client = genai
        elif "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            self.client = OpenAI(api_key=self.api_key)
        elif "claude" in self.model_name.lower():
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _initialize_field_configs(self) -> Dict[str, FieldConfig]:
        """Initialize field configurations based on schema"""
        configs = {}
        
        if not self.schema:
            return self._get_default_field_configs()
        
        # Get required fields from schema
        required_fields = set(self.schema.get("required", []))
        schema_properties = self.schema.get("properties", {})
        
        # Core required fields (Wave 1)
        for field_name in ["publicationIdentifier", "FrontPage", "Claims", "Description"]:
            if field_name in schema_properties:
                configs[field_name] = FieldConfig(
                    name=field_name,
                    wave=ProcessingWave.WAVE_1_CORE,
                    dependencies=[],
                    schema_ref=f"#/properties/{field_name}",
                    required=field_name in required_fields,
                    prompt_template=self._get_field_prompt_template(field_name)
                )
        
        # Structure libraries (Wave 2)
        structure_fields = [
            "ChemicalStructureLibrary", "BiologicalSequenceLibrary", 
            "Figures", "Tables"
        ]
        for field_name in structure_fields:
            if field_name in schema_properties:
                configs[field_name] = FieldConfig(
                    name=field_name,
                    wave=ProcessingWave.WAVE_2_STRUCTURES,
                    dependencies=["Claims", "Description"],
                    schema_ref=f"#/properties/{field_name}",
                    domain_specific=True,
                    prompt_template=self._get_field_prompt_template(field_name)
                )
        
        # Metadata fields (Wave 3)
        metadata_fields = [
            "IndustrialApplicability", "InternationalSearchReport",
            "PatentFamilyInformation", "FrontPageContinuation"
        ]
        for field_name in metadata_fields:
            if field_name in schema_properties:
                configs[field_name] = FieldConfig(
                    name=field_name,
                    wave=ProcessingWave.WAVE_3_METADATA,
                    dependencies=["FrontPage"],
                    schema_ref=f"#/properties/{field_name}",
                    prompt_template=self._get_field_prompt_template(field_name)
                )
        
        # Add definition-based fields
        definitions = self.schema.get("definitions", {})
        for def_name, def_schema in definitions.items():
            if def_name not in configs:
                wave = self._determine_definition_wave(def_name, def_schema)
                dependencies = self._infer_definition_dependencies(def_name, def_schema)
                
                configs[def_name] = FieldConfig(
                    name=def_name,
                    wave=wave,
                    dependencies=dependencies,
                    schema_ref=f"#/definitions/{def_name}",
                    prompt_template=self._get_definition_prompt_template(def_name, def_schema)
                )
        
        return configs
    
    def _get_default_field_configs(self) -> Dict[str, FieldConfig]:
        """Get default field configurations when no schema is provided"""
        return {
            "publicationIdentifier": FieldConfig(
                "publicationIdentifier", ProcessingWave.WAVE_1_CORE, [], "", True, True,
                "Extract the patent publication number (e.g., JP2022-105746A, WO2020162638A1)"
            ),
            "FrontPage": FieldConfig(
                "FrontPage", ProcessingWave.WAVE_1_CORE, [], "", False, True,
                "Extract front page information including publication data, application data, inventors, applicants, classification, and abstract"
            ),
            "Claims": FieldConfig(
                "Claims", ProcessingWave.WAVE_1_CORE, [], "", False, True,
                "Extract all patent claims with detailed structure and technical content"
            ),
            "Description": FieldConfig(
                "Description", ProcessingWave.WAVE_1_CORE, [], "", False, True,
                "Extract technical field, background art, summary of invention, detailed description, and examples"
            )
        }
    
    def _determine_definition_wave(self, def_name: str, def_schema: Dict) -> ProcessingWave:
        """Determine processing wave for schema definitions"""
        if any(keyword in def_name.lower() for keyword in ["chemical", "molecular", "compound"]):
            return ProcessingWave.WAVE_2_STRUCTURES
        elif any(keyword in def_name.lower() for keyword in ["biological", "sequence", "protein", "dna"]):
            return ProcessingWave.WAVE_2_STRUCTURES
        elif any(keyword in def_name.lower() for keyword in ["table", "figure", "image"]):
            return ProcessingWave.WAVE_2_STRUCTURES
        elif any(keyword in def_name.lower() for keyword in ["person", "organization", "claim"]):
            return ProcessingWave.WAVE_1_CORE
        else:
            return ProcessingWave.WAVE_3_METADATA
    
    def _infer_definition_dependencies(self, def_name: str, def_schema: Dict) -> List[str]:
        """Infer dependencies for schema definitions"""
        dependencies = []
        
        # Analyze schema references
        schema_str = json.dumps(def_schema)
        
        if "chemical" in def_name.lower() or "compound" in def_name.lower():
            dependencies.extend(["Claims", "Description"])
        elif "biological" in def_name.lower() or "sequence" in def_name.lower():
            dependencies.extend(["Claims", "Description"])
        elif "table" in def_name.lower():
            dependencies.append("Description")
        elif "figure" in def_name.lower():
            dependencies.append("Description")
        elif "person" in def_name.lower() or "organization" in def_name.lower():
            dependencies.append("FrontPage")
        
        return dependencies
    
    def _get_field_prompt_template(self, field_name: str) -> str:
        """Get prompt template for specific field"""
        templates = {
            "publicationIdentifier": """
Extract the patent publication identifier from the document header or front page.
Look for formats like: JP2022-105746A, WO2020162638A1, US10123456B2, etc.
Return as a simple string value.
""",
            
            "FrontPage": """
Extract comprehensive front page information including:
- Publication data (number, kind, date, office)
- Application data (number, international number, dates)
- Inventors (names and addresses)
- Applicants (names and addresses)  
- Classification data (IPC codes)
- Abstract (complete text)
Follow the FrontPageType schema structure exactly.
""",
            
            "Claims": """
Extract all patent claims with complete technical detail:
- Claim number and text
- Chemical structures with detailed molecular graphs
- Dependencies between claims
- Technical elements and parameters
- References to figures, tables, and sequences
Follow the ClaimsType schema structure exactly.
""",
            
            "Description": """
Extract the complete technical description including:
- Technical Field
- Background Art  
- Summary of Invention
- Detailed Description of Invention
- Examples with full procedural details
Include all chemical structures, figures, tables, and technical data.
Follow the DescriptionType schema structure exactly.
""",
            
            "ChemicalStructureLibrary": """
Extract ALL chemical structures with complete molecular graph representation:

FOR EACH CHEMICAL STRUCTURE:
1. Create detailed molecular graph with:
   - Nodes for each atom (element, charge, hybridization, coordinates)
   - Edges for each bond (order, type, stereochemistry)
   - Ring systems and aromaticity
   - Functional groups identification

2. Generate alternative representations:
   - SMILES (canonical and isomeric)
   - InChI and InChI Key
   - Molecular formula and weight

3. Extract patent context:
   - Role in invention (product, intermediate, etc.)
   - Synthesis information
   - Characterization data (NMR, IR, MS, etc.)
   - Activity data if present

4. Map references to claims and examples

Follow the ChemicalStructureLibrary schema structure exactly with complete ChemicalGraphType data.
""",
            
            "BiologicalSequenceLibrary": """
Extract ALL biological sequences with complete annotation:
- Protein sequences (amino acid sequences, lengths, properties)
- Nucleic acid sequences (DNA/RNA, sequence types, lengths)
- Sequence identifiers (SEQ ID NO, names, organisms)
- Functional information and roles
- References to claims and examples
Follow the BiologicalSequenceLibrary schema structure exactly.
""",
            
            "Tables": """
Extract ALL tables with complete structure and data:
- Table numbers, titles, and captions
- Complete row and column structure
- All cell content with proper data typing
- References and context within the patent
Follow the TablesType schema structure exactly.
""",
            
            "Figures": """
Extract ALL figures with complete information:
- Figure numbers and captions
- Figure types and content descriptions
- References to related text
Follow the FiguresType schema structure exactly.
"""
        }
        
        return templates.get(field_name, f"Extract {field_name} according to schema requirements.")
    
    def _get_definition_prompt_template(self, def_name: str, def_schema: Dict) -> str:
        """Get prompt template for schema definitions"""
        if "chemical" in def_name.lower() or "compound" in def_name.lower():
            return """
Extract detailed chemical structure information following the molecular graph schema.
Include complete atom and bond data, stereochemistry, and patent context.
"""
        elif "biological" in def_name.lower() or "sequence" in def_name.lower():
            return """
Extract detailed biological sequence information including sequence data,
identifiers, organism information, and functional annotations.
"""
        elif "table" in def_name.lower():
            return """
Extract complete table structure with all rows, columns, and cell data.
Properly type the data and maintain table relationships.
"""
        else:
            return f"Extract {def_name} information according to schema requirements."
    
    def detect_domain_parallel(self, pdf_path: str) -> Dict[str, Any]:
        """Parallel domain detection for adaptive processing"""
        
        domain_prompt = """
Quickly analyze this patent and determine:

1. Primary technical domain:
   - chemical (molecules, compounds, drugs, materials)
   - mechanical (devices, machines, systems)
   - electrical (circuits, electronics, software)
   - biotechnology (proteins, DNA, biological processes)
   - software (algorithms, methods, data processing)
   - other

2. Key structural elements present:
   - Chemical structures/formulas
   - Biological sequences
   - Mechanical drawings
   - Circuit diagrams
   - Flowcharts/algorithms
   - Tables with data
   - Figures/images

3. Extraction priorities:
   - High priority fields for this domain
   - Expected complexity level
   - Special processing requirements

Return JSON with domain classification and processing recommendations.
"""
        
        try:
            result = self._process_single_field(pdf_path, domain_prompt, "domain_detection")
            self._domain_cache[pdf_path] = result
            return result
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return {"primary_domain": "general", "extraction_priorities": []}
    
    def process_patent_parallel(self, pdf_path: str) -> Dict[str, Any]:
        """Main parallel processing pipeline"""
        start_time = time.time()
        logger.info(f"Starting parallel patent processing: {pdf_path}")
        
        try:
            # Step 1: Domain detection
            domain_info = self.detect_domain_parallel(pdf_path)
            logger.info(f"Domain detected: {domain_info.get('primary_domain', 'unknown')}")
            
            # Step 2: Parallel wave processing
            result = self._process_waves_parallel(pdf_path, domain_info)
            
            # Step 3: Validation and cleanup
            final_result = self._validate_and_cleanup(result)
            
            # Add processing metadata
            total_time = time.time() - start_time
            final_result["_processing_metadata"] = {
                "total_time_seconds": total_time,
                "domain_detected": domain_info.get("primary_domain"),
                "model_used": self.model_name,
                "parallel_workers": self.max_workers,
                "processing_stats": self._processing_stats,
                "field_timing": self._timing_data,
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_processed": Path(pdf_path).name
            }
            
            logger.info(f"Parallel processing completed in {total_time:.2f} seconds")
            return final_result
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            return {
                "error": str(e),
                "partial_data": self._shared_data.copy(),
                "processing_failed": True
            }
    
    def _process_waves_parallel(self, pdf_path: str, domain_info: Dict) -> Dict[str, Any]:
        """Process fields in parallel waves"""
        
        # Group fields by wave
        wave_groups = {}
        for field_name, config in self.field_configs.items():
            wave = config.wave
            if wave not in wave_groups:
                wave_groups[wave] = []
            wave_groups[wave].append(field_name)
        
        final_result = {}
        
        # Process waves sequentially, fields within wave in parallel
        for wave in sorted(wave_groups.keys(), key=lambda x: x.value):
            fields_in_wave = wave_groups[wave]
            wave_start_time = time.time()
            
            logger.info(f"Processing {wave.name} with {len(fields_in_wave)} fields: {fields_in_wave}")
            
            # Filter fields based on domain and requirements
            active_fields = self._filter_fields_for_domain(fields_in_wave, domain_info)
            
            if not active_fields:
                logger.info(f"No active fields for {wave.name}, skipping")
                continue
            
            # Process fields in parallel
            wave_results = self._process_wave_parallel(pdf_path, active_fields, domain_info)
            
            # Update shared data and final result
            with self._data_lock:
                self._shared_data.update(wave_results)
                final_result.update(wave_results)
            
            wave_time = time.time() - wave_start_time
            logger.info(f"{wave.name} completed in {wave_time:.2f} seconds")
        
        return final_result
    
    def _filter_fields_for_domain(self, fields: List[str], domain_info: Dict) -> List[str]:
        """Filter fields based on domain relevance"""
        domain = domain_info.get("primary_domain", "general")
        
        # Always include required fields
        required_fields = [f for f in fields if self.field_configs.get(f, FieldConfig("", ProcessingWave.WAVE_1_CORE)).required]
        
        # Add domain-specific fields
        domain_specific_fields = []
        
        if domain == "chemical":
            domain_specific_fields.extend([f for f in fields if "chemical" in f.lower() or "compound" in f.lower()])
        elif domain == "biotechnology":
            domain_specific_fields.extend([f for f in fields if "biological" in f.lower() or "sequence" in f.lower()])
        
        # Always include general fields
        general_fields = [f for f in fields if not self.field_configs.get(f, FieldConfig("", ProcessingWave.WAVE_1_CORE)).domain_specific]
        
        return list(set(required_fields + domain_specific_fields + general_fields))
    
    def _process_wave_parallel(self, pdf_path: str, fields: List[str], domain_info: Dict) -> Dict[str, Any]:
        """Process a wave of fields in parallel"""
        
        wave_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all field extraction tasks
            future_to_field = {
                executor.submit(self._extract_field_with_timing, pdf_path, field_name, domain_info): field_name
                for field_name in fields
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_field):
                field_name = future_to_field[future]
                try:
                    field_result = future.result(timeout=300)  # 5 minute timeout per field
                    
                    if field_result and field_name in field_result:
                        wave_results[field_name] = field_result[field_name]
                        self._processing_stats["successful_fields"] += 1
                        logger.info(f"✓ {field_name} completed successfully")
                    else:
                        wave_results[field_name] = None
                        self._processing_stats["failed_fields"] += 1
                        logger.warning(f"✗ {field_name} returned no data")
                        
                except Exception as e:
                    wave_results[field_name] = {"error": str(e)}
                    self._processing_stats["failed_fields"] += 1
                    logger.error(f"✗ {field_name} failed: {e}")
        
        return wave_results
    
    def _extract_field_with_timing(self, pdf_path: str, field_name: str, domain_info: Dict) -> Dict[str, Any]:
        """Extract field with timing measurement"""
        start_time = time.time()
        
        try:
            result = self._extract_single_field(pdf_path, field_name, domain_info)
            processing_time = time.time() - start_time
            self._timing_data[field_name] = processing_time
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._timing_data[field_name] = processing_time
            raise e
    
    def _extract_single_field(self, pdf_path: str, field_name: str, domain_info: Dict) -> Dict[str, Any]:
        """Extract a single field with schema compliance"""
        
        config = self.field_configs.get(field_name)
        if not config:
            raise ValueError(f"No configuration found for field: {field_name}")
        
        # Get schema information for this field
        field_schema = self._get_field_schema_info(field_name)
        
        # Create field-specific prompt
        prompt = self._create_field_prompt(field_name, config, field_schema, domain_info)
        
        # Add dependency context
        dependency_context = self._get_dependency_context(field_name)
        if dependency_context:
            prompt += f"\n\nDependency Context:\n{dependency_context}"
        
        # Process with AI model
        return self._process_single_field(pdf_path, prompt, field_name)
    
    def _get_field_schema_info(self, field_name: str) -> Dict[str, Any]:
        """Get schema information for field"""
        if not self.schema:
            return {}
        
        # Check in properties
        if field_name in self.schema.get("properties", {}):
            return {
                "type": "property",
                "schema": self.schema["properties"][field_name],
                "required": field_name in self.schema.get("required", [])
            }
        
        # Check in definitions
        if field_name in self.schema.get("definitions", {}):
            return {
                "type": "definition", 
                "schema": self.schema["definitions"][field_name]
            }
        
        return {}
    
    def _create_field_prompt(self, field_name: str, config: FieldConfig, field_schema: Dict, domain_info: Dict) -> str:
        """Create comprehensive field-specific prompt"""
        
        base_prompt = config.prompt_template or f"Extract {field_name} information from the patent."
        
        # Add schema structure information
        if field_schema:
            schema_description = self._generate_schema_description(field_schema.get("schema", {}))
            base_prompt += f"\n\nSchema Structure:\n{schema_description}"
        
        # Add domain-specific enhancements
        domain = domain_info.get("primary_domain", "general")
        if config.domain_specific and domain in ["chemical", "biotechnology"]:
            base_prompt += f"\n\nDomain-Specific Focus ({domain}):\n"
            base_prompt += self._get_domain_specific_instructions(field_name, domain)
        
        # Add output format requirements
        base_prompt += f"""

OUTPUT REQUIREMENTS:
- Return valid JSON matching the schema structure exactly
- Include all available information from the PDF
- Use proper data types (strings, numbers, arrays, objects)
- Set missing values to null
- Ensure proper nesting and field names
- For {field_name}, follow the schema specification precisely

Return only the JSON for the {field_name} field.
"""
        
        return base_prompt
    
    def _generate_schema_description(self, schema: Dict) -> str:
        """Generate human-readable schema description"""
        if not schema:
            return "No specific schema requirements."
        
        def describe_object(obj, indent=0):
            if not isinstance(obj, dict):
                return str(obj)
            
            lines = []
            obj_type = obj.get("type", "unknown")
            
            if obj_type == "object":
                lines.append("object with properties:")
                props = obj.get("properties", {})
                required = obj.get("required", [])
                
                for prop_name, prop_schema in props.items():
                    req_marker = " (required)" if prop_name in required else ""
                    prop_desc = describe_object(prop_schema, indent + 1)
                    lines.append("  " * (indent + 1) + f"- {prop_name}{req_marker}: {prop_desc}")
                    
            elif obj_type == "array":
                items_desc = describe_object(obj.get("items", {}), indent)
                lines.append(f"array of {items_desc}")
                
            else:
                enum_vals = obj.get("enum")
                if enum_vals:
                    lines.append(f"{obj_type} (one of: {', '.join(map(str, enum_vals))})")
                else:
                    lines.append(obj_type)
            
            return "\n".join(lines) if len(lines) > 1 else lines[0] if lines else "unknown"
        
        return describe_object(schema)
    
    def _get_domain_specific_instructions(self, field_name: str, domain: str) -> str:
        """Get domain-specific extraction instructions"""
        
        if domain == "chemical" and "chemical" in field_name.lower():
            return """
- Extract complete molecular graphs with all atoms and bonds
- Include stereochemistry (R/S, E/Z configurations)
- Generate SMILES and InChI if possible
- Map functional groups and ring systems
- Include synthesis and characterization data
- Note biological activity if mentioned
"""
        
        elif domain == "biotechnology" and "biological" in field_name.lower():
            return """
- Extract complete sequence data with annotations
- Include organism and functional information
- Map sequence features and domains
- Note expression and activity data
- Include assay protocols if described
"""
        
        return ""
    
    def _get_dependency_context(self, field_name: str) -> str:
        """Get dependency context for field extraction"""
        config = self.field_configs.get(field_name)
        if not config or not config.dependencies:
            return ""
        
        context_parts = []
        
        with self._data_lock:
            for dep in config.dependencies:
                if dep in self._shared_data and self._shared_data[dep] is not None:
                    dep_data = self._shared_data[dep]
                    summary = self._create_dependency_summary(dep, dep_data)
                    if summary:
                        context_parts.append(f"[{dep}] {summary}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _create_dependency_summary(self, dep_name: str, dep_data: Any) -> str:
        """Create summary of dependency data"""
        if not isinstance(dep_data, dict):
            return ""
        
        if dep_name == "Claims":
            claims = dep_data.get("Claim", [])
            return f"Contains {len(claims)} claims"
        elif dep_name == "Description":
            sections = [k for k in dep_data.keys() if not k.startswith("_")]
            return f"Contains sections: {', '.join(sections[:3])}{'...' if len(sections) > 3 else ''}"
        elif dep_name == "FrontPage":
            pub_data = dep_data.get("PublicationData", {})
            return f"Patent: {pub_data.get('PublicationNumber', 'Unknown')}"
        
        return f"Available data structure: {list(dep_data.keys())[:3]}"
    
    def _process_single_field(self, pdf_path: str, prompt: str, field_name: str) -> Dict[str, Any]:
        """Process single field with appropriate AI model"""
        
        try:
            if "gemini" in self.model_name.lower():
                return self._process_with_gemini(pdf_path, prompt, field_name)
            elif "gpt" in self.model_name.lower():
                return self._process_with_openai(pdf_path, prompt, field_name)
            elif "claude" in self.model_name.lower():
                return self._process_with_anthropic(pdf_path, prompt, field_name)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error processing {field_name}: {e}")
            return {field_name: {"error": str(e)}}
    
    def _process_with_gemini(self, pdf_path: str, prompt: str, field_name: str) -> Dict[str, Any]:
        """Process with Gemini"""
        model = self.client.GenerativeModel(
            model_name=self.model_name,
            system_instruction="You are an expert patent analysis system. Extract information precisely according to JSON schema requirements."
        )
        
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        response = model.generate_content(
            contents=[
                f"{prompt}\n\nReturn valid JSON for {field_name} only.",
                {"mime_type": "application/pdf", "data": pdf_data}
            ],
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        
        return self._extract_json_from_response(response.text, field_name)
    
    def _process_with_openai(self, pdf_path: str, prompt: str, field_name: str) -> Dict[str, Any]:
        """Process with OpenAI"""
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert patent analysis system. Extract information precisely according to JSON schema requirements."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n\nReturn valid JSON for {field_name} only."},
                        {
                            "type": "file",
                            "file": {
                                "filename": f"{pdf_path}",
                                "file_data": f"data:application/pdf;base64,{pdf_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return self._extract_json_from_response(response.choices[0].message.content, field_name)
    
    def _process_with_anthropic(self, pdf_path: str, prompt: str, field_name: str) -> Dict[str, Any]:
        """Process with Anthropic"""
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.messages.create(
            model=self.model_name,
            system="You are an expert patent analysis system. Extract information precisely according to JSON schema requirements.",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}\n\nReturn valid JSON for {field_name} only."
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
        
        return self._extract_json_from_response(response.content[0].text, field_name)
    
    def _extract_json_from_response(self, text: str, field_name: str) -> Dict[str, Any]:
        """Extract JSON from AI response with robust error handling"""
        try:
            # Try to find JSON blocks first
            if "```json" in text:
                json_block = text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_block)
                return {field_name: parsed} if field_name not in parsed else parsed
            
            # Find JSON-like content
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                
                # Try to parse
                try:
                    parsed = json.loads(json_text)
                    return {field_name: parsed} if field_name not in parsed else parsed
                except json.JSONDecodeError:
                    # Try to fix common issues
                    fixed_json = self._fix_json_issues(json_text)
                    parsed = json.loads(fixed_json)
                    return {field_name: parsed} if field_name not in parsed else parsed
            
            # If no valid JSON found, create error response
            return {
                field_name: {
                    "extraction_error": "No valid JSON found in response",
                    "raw_response": text[:500]
                }
            }
            
        except Exception as e:
            logger.error(f"JSON extraction failed for {field_name}: {e}")
            return {
                field_name: {
                    "extraction_error": str(e),
                    "raw_response": text[:500] if text else "No response"
                }
            }
    
    def _fix_json_issues(self, json_text: str) -> str:
        """Fix common JSON formatting issues"""
        import re
        
        # Remove trailing commas
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix unescaped quotes (basic approach)
        json_text = re.sub(r'(?<!\\)"(?=[^"]*":)', '\\"', json_text)
        
        # Remove comments
        json_text = re.sub(r'//.*?\n', '\n', json_text)
        json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
        
        return json_text
    
    def _validate_and_cleanup(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and cleanup extraction results"""
        
        # Ensure required fields are present
        if self.schema and "required" in self.schema:
            for required_field in self.schema["required"]:
                if required_field not in result or result[required_field] is None:
                    logger.warning(f"Required field {required_field} is missing")
                    result[required_field] = self._create_empty_field_structure(required_field)
        
        # Clean up error responses and empty data
        cleaned_result = {}
        for field_name, field_data in result.items():
            if field_data is None:
                continue
                
            if isinstance(field_data, dict) and "extraction_error" in field_data:
                logger.warning(f"Field {field_name} had extraction error: {field_data['extraction_error']}")
                # Keep error for debugging but try to provide empty structure
                cleaned_result[field_name] = self._create_empty_field_structure(field_name)
            else:
                cleaned_result[field_name] = field_data
        
        # Ensure publication identifier
        if "publicationIdentifier" not in cleaned_result or not cleaned_result["publicationIdentifier"]:
            cleaned_result["publicationIdentifier"] = "UNKNOWN"
        
        return cleaned_result
    
    def _create_empty_field_structure(self, field_name: str) -> Any:
        """Create empty structure for missing fields based on schema"""
        
        if not self.schema:
            return None
        
        # Get field schema
        field_schema = None
        if field_name in self.schema.get("properties", {}):
            field_schema = self.schema["properties"][field_name]
        elif field_name in self.schema.get("definitions", {}):
            field_schema = self.schema["definitions"][field_name]
        
        if not field_schema:
            return None
        
        return self._create_empty_from_schema(field_schema)
    
    def _create_empty_from_schema(self, schema: Dict) -> Any:
        """Create empty structure from schema definition"""
        
        schema_type = schema.get("type")
        
        if schema_type == "object":
            obj = {}
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            for prop_name, prop_schema in properties.items():
                if prop_name in required:
                    obj[prop_name] = self._create_empty_from_schema(prop_schema)
            
            return obj
            
        elif schema_type == "array":
            return []
            
        elif schema_type == "string":
            return ""
            
        elif schema_type in ["integer", "number"]:
            return 0
            
        elif schema_type == "boolean":
            return False
            
        else:
            return None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        return {
            "total_fields_processed": self._processing_stats["total_fields"],
            "successful_extractions": self._processing_stats["successful_fields"],
            "failed_extractions": self._processing_stats["failed_fields"],
            "success_rate": (
                self._processing_stats["successful_fields"] / 
                max(1, self._processing_stats["total_fields"])
            ) * 100,
            "field_timing": self._timing_data,
            "average_field_time": (
                sum(self._timing_data.values()) / 
                max(1, len(self._timing_data))
            ),
            "slowest_fields": sorted(
                self._timing_data.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def validate_schema_compliance(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate result against provided schema"""
        
        validation_report = {
            "is_compliant": True,
            "errors": [],
            "warnings": [],
            "field_coverage": {},
            "schema_version": self.schema.get("$schema", "unknown")
        }
        
        if not self.schema:
            validation_report["warnings"].append("No schema provided for validation")
            return validation_report
        
        # Check required fields
        required_fields = self.schema.get("required", [])
        for field in required_fields:
            if field not in result:
                validation_report["errors"].append(f"Missing required field: {field}")
                validation_report["is_compliant"] = False
                validation_report["field_coverage"][field] = "missing"
            elif result[field] is None:
                validation_report["warnings"].append(f"Required field is null: {field}")
                validation_report["field_coverage"][field] = "null"
            else:
                validation_report["field_coverage"][field] = "present"
        
        # Check optional fields
        schema_properties = self.schema.get("properties", {})
        for field in schema_properties:
            if field not in required_fields:
                if field in result and result[field] is not None:
                    validation_report["field_coverage"][field] = "present"
                else:
                    validation_report["field_coverage"][field] = "missing"
        
        # Calculate coverage statistics
        total_fields = len(schema_properties)
        present_fields = len([k for k, v in validation_report["field_coverage"].items() if v == "present"])
        validation_report["coverage_percentage"] = (present_fields / max(1, total_fields)) * 100
        
        return validation_report

    def main():
        """Universal main function for any patent type with parallel processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Patent Extractor with Parallel Processing and Schema Compliance')
    parser.add_argument('pdf_path', help='Path to the patent PDF file')
    parser.add_argument('--model', default='claude-3-sonnet-20241022', help='AI model to use')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--schema', help='Path to JSON schema file')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--temperature', type=float, default=0.1, help='Model temperature')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Maximum tokens per request')
    parser.add_argument('--validate', action='store_true', help='Validate result against schema')
    parser.add_argument('--stats', action='store_true', help='Show detailed processing statistics')
    parser.add_argument('--domain-only', action='store_true', help='Only detect patent domain')
    
    args = parser.parse_args()
    
    # Load schema if provided
    schema = None
    if args.schema:
        try:
            with open(args.schema, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema from {args.schema}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return
    
    # Initialize extractor
    extractor = PatentExtractor(
        model_name=args.model,
        api_key=args.api_key,
        json_schema=schema,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.workers
    )
    
    if args.domain_only:
        # Only run domain detection
        result = extractor.detect_domain_parallel(args.pdf_path)
        print("Domain Detection Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # Run complete parallel processing
    logger.info(f"Starting parallel processing with {args.workers} workers")
    result = extractor.process_patent_parallel(args.pdf_path)
    
    # Validation if requested
    if args.validate and schema:
        validation_report = extractor.validate_schema_compliance(result)
        print("\n" + "="*50)
        print("SCHEMA VALIDATION REPORT")
        print("="*50)
        print(f"Compliant: {validation_report['is_compliant']}")
        print(f"Coverage: {validation_report['coverage_percentage']:.1f}%")
        
        if validation_report['errors']:
            print(f"\nErrors ({len(validation_report['errors'])}):")
            for error in validation_report['errors']:
                print(f"  ❌ {error}")
        
        if validation_report['warnings']:
            print(f"\nWarnings ({len(validation_report['warnings'])}):")
            for warning in validation_report['warnings']:
                print(f"  ⚠️  {warning}")
    
    # Processing statistics if requested
    if args.stats:
        stats = extractor.get_processing_statistics()
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Average Field Time: {stats['average_field_time']:.2f}s")
        print(f"Total Fields: {stats['total_fields_processed']}")
        print(f"Successful: {stats['successful_extractions']}")
        print(f"Failed: {stats['failed_extractions']}")
        
        print(f"\nSlowest Fields:")
        for field, time_taken in stats['slowest_fields']:
            print(f"  {field}: {time_taken:.2f}s")
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Extraction completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
        if "_processing_metadata" in result:
            metadata = result["_processing_metadata"]
            print(f"🔍 Domain detected: {metadata.get('domain_detected', 'unknown')}")
            print(f"⏱️  Processing time: {metadata.get('total_time_seconds', 0):.2f} seconds")
            print(f"🚀 Workers used: {metadata.get('parallel_workers', 1)}")
            print(f"📊 Success rate: {metadata.get('processing_stats', {}).get('successful_fields', 0)}/{metadata.get('processing_stats', {}).get('total_fields', 0)} fields")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()