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

# Log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("patent-extractor")

class PatentExtractor:
    """Library for extracting structured JSON from patent PDFs (parallel processing specialized)"""
    
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
        Initialize
        
        Args:
            model_name: AI model name to use
            api_key: API authentication key
            json_schema: JSON schema (dictionary format)
            user_prompt: Custom prompt
            temperature: AI temperature setting value (0.0-1.0)
            max_tokens: Maximum number of AI tokens
            max_workers: Number of parallel processes (default increased to 8)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(self._get_env_var_name(model_name))
        self.schema = json_schema or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        
        # Default prompt
        self.prompt = user_prompt or """Extract and structure information from the attached patent PDF according to the provided JSON schema.
Include all sections specified in the schema such as front page information, claims, detailed description, etc.
Follow the schema structure accurately and extract all headings and subheadings while maintaining the hierarchical structure of the patent document.
Include identifiers and reference information for chemical formulas, figures, and tables."""
        
        # Field definitions (complete coverage based on schema)
        self.field_definitions = {
            # Wave 1: Basic required information (fully parallel)
            "publicationIdentifier": {
                "description": "Extract patent publication number",
                "wave": 1,
                "dependencies": []
            },
            "FrontPage": {
                "description": "Extract front page information (application info, inventors, applicants, classification, abstract)",
                "wave": 1,
                "dependencies": []
            },
            "Claims": {
                "description": "Extract patent claims",
                "wave": 1,
                "dependencies": []
            },
            "Description": {
                "description": "Extract technical field, background art, summary of invention, detailed description",
                "wave": 1,
                "dependencies": []
            },
            
            # Wave 2: Basic data libraries (reference basic information but execute in parallel)
            "ChemicalStructureLibrary": {
                "description": "Extract all chemical structures from the patent",
                "wave": 2,
                "dependencies": ["Claims", "Description"]
            },
            "BiologicalSequenceLibrary": {
                "description": "Extract all biological sequences from the patent",
                "wave": 2,
                "dependencies": ["Claims", "Description"]
            },
            "Figures": {
                "description": "Extract all figures from the patent",
                "wave": 2,
                "dependencies": ["Description"]
            },
            "Tables": {
                "description": "Extract all tables from the patent",
                "wave": 2,
                "dependencies": ["Description"]
            },
            "IndustrialApplicability": {
                "description": "Extract industrial applicability",
                "wave": 2,
                "dependencies": ["Description"]
            },
            
            # Wave 3: Advanced analysis and metadata (utilizing Wave 2 results)
            "InternationalSearchReport": {
                "description": "Extract international search report",
                "wave": 3,
                "dependencies": ["FrontPage"]
            },
            "PatentFamilyInformation": {
                "description": "Extract patent family information",
                "wave": 3,
                "dependencies": ["FrontPage"]
            },
            "FrontPageContinuation": {
                "description": "Extract front page continuation information (designated countries, F-Terms, etc.)",
                "wave": 3,
                "dependencies": ["FrontPage"]
            }
        }
        
        # Update field definitions dynamically from schema
        self._update_field_definitions_from_schema()
        
        # Initialize client
        self._init_client()
        
        # Shared data lock (ReadWriteLock-style implementation for acceleration)
        self._data_lock = threading.RLock()
        self._shared_data = {}
        
        # Performance measurement
        self._timing_data = {}
    
    def _update_field_definitions_from_schema(self):
        """Dynamically update field definitions from schema"""
        if not self.schema or "properties" not in self.schema:
            return
            
        schema_properties = self.schema["properties"]
        required_fields = set(self.schema.get("required", []))
        
        # Add fields that exist in schema but not in field_definitions
        for field_name in schema_properties:
            if field_name not in self.field_definitions:
                # Required fields go to Wave 1, optional fields to Wave 2
                wave = 1 if field_name in required_fields else 2
                
                # Infer dependencies based on field characteristics
                dependencies = self._infer_dependencies(field_name, schema_properties[field_name])
                
                self.field_definitions[field_name] = {
                    "description": f"Extract {field_name} section",
                    "wave": wave,
                    "dependencies": dependencies
                }
                
                logger.info(f"Added field from schema: {field_name} (Wave {wave})")
        
        # Add all type definitions from definitions section to processing targets
        definitions = self.schema.get("definitions", {})
        
        # Wave 2: Basic structure types (types directly referenced from properties)
        basic_structure_types = [
            "PersonType", "OrganizationType", "ClaimType", "SectionType", 
            "ExamplesType", "ParagraphType", "TableType", "TableRefType", 
            "ChemicalStructureType", "PatentChemicalCompoundType", "FigureRefType", 
            "DesignatedCountriesType", "FTermsType", "ProteinSequenceType", "NucleicAcidSequenceType"
        ]
        
        # Wave 3: Detailed structure types (indirectly important types)
        detailed_structure_types = [
            "MoleculeType", "TableRowType", "TableCellType", "ExampleType", 
            "ImageRefType", "BiologicalSequenceRefType"
        ]
        
        # Wave 4: Auxiliary structure types (auxiliary classification and reference types)
        auxiliary_types = [
            "RegionGroupType", "FTermType"
        ]
        
        # Add all type definitions
        all_definition_types = [
            (basic_structure_types, 2, "Basic structure type"),
            (detailed_structure_types, 3, "Detailed structure type"),
            (auxiliary_types, 4, "Auxiliary structure type")
        ]
        
        for type_list, wave, category in all_definition_types:
            for def_name in type_list:
                if def_name in definitions and def_name not in self.field_definitions:
                    dependencies = self._infer_dependencies(def_name, definitions[def_name])
                    
                    self.field_definitions[def_name] = {
                        "description": f"Extract detailed structural information of {def_name}",
                        "wave": wave,
                        "dependencies": dependencies,
                        "is_definition": True,
                        "category": category
                    }
                    
                    logger.info(f"Added {category}: {def_name} (Wave {wave})")
        
        # Check for additional undefined types
        all_definition_names = set(definitions.keys())
        processed_definitions = set(basic_structure_types + detailed_structure_types + auxiliary_types)
        missing_definitions = all_definition_names - processed_definitions
        
        if missing_definitions:
            logger.warning(f"Unprocessed type definitions found: {missing_definitions}")
            # Add unprocessed type definitions as Wave 4
            for def_name in missing_definitions:
                if def_name not in self.field_definitions:
                    dependencies = self._infer_dependencies(def_name, definitions[def_name])
                    
                    self.field_definitions[def_name] = {
                        "description": f"Extract structural information of {def_name}",
                        "wave": 4,
                        "dependencies": dependencies,
                        "is_definition": True,
                        "category": "Other type definitions"
                    }
                    
                    logger.info(f"Added missing definition: {def_name} (Wave 4)")
        
        logger.info(f"Total fields processed: {len(self.field_definitions)} (including {len(definitions)} type definitions)")
    
    def _infer_dependencies(self, field_name: str, field_schema: Dict) -> List[str]:
        """Infer field dependencies"""
        dependencies = []
        
        # Property-level dependencies
        property_dependencies = {
            # Library systems depend on basic information
            "ChemicalStructureLibrary": ["Claims", "Description"],
            "BiologicalSequenceLibrary": ["Claims", "Description"],
            # Continuation systems depend on front page
            "FrontPageContinuation": ["FrontPage"],
            "PatentFamilyInformation": ["FrontPage"],
            "InternationalSearchReport": ["FrontPage"],
            # Others
            "IndustrialApplicability": ["Description"],
            "Figures": ["Description"],
            "Tables": ["Description"]
        }
        
        # Definition-level dependencies
        definition_dependencies = {
            # Molecular and chemical related
            "MoleculeType": ["ChemicalStructureLibrary"],
            "ChemicalStructureType": ["ChemicalStructureLibrary"],
            "PatentChemicalCompoundType": ["ChemicalStructureLibrary"],
            # Biological sequence related
            "ProteinSequenceType": ["BiologicalSequenceLibrary"],
            "NucleicAcidSequenceType": ["BiologicalSequenceLibrary"],
            "BiologicalSequenceRefType": ["BiologicalSequenceLibrary"],
            # Table related
            "TableType": ["Tables"],
            "TableRowType": ["Tables"],
            "TableCellType": ["Tables"],
            "TableRefType": ["Tables"],
            # Figure related
            "FigureRefType": ["Figures"],
            "ImageRefType": ["Figures"],
            # Section and structure related
            "ExampleType": ["Description"],
            "ExamplesType": ["Description"],
            "SectionType": ["Description"],
            "ClaimType": ["Claims"],
            "ParagraphType": ["Description", "Claims"],
            # Person and organization related
            "PersonType": ["FrontPage"],
            "OrganizationType": ["FrontPage"],
            # Region and classification related
            "DesignatedCountriesType": ["FrontPageContinuation"],
            "RegionGroupType": ["FrontPageContinuation"],
            "FTermsType": ["FrontPageContinuation"],
            "FTermType": ["FrontPageContinuation"]
        }
        
        # Get direct dependencies
        if field_name in property_dependencies:
            dependencies.extend(property_dependencies[field_name])
        elif field_name in definition_dependencies:
            dependencies.extend(definition_dependencies[field_name])
        
        # Check references in schema
        field_str = json.dumps(field_schema)
        reference_mappings = {
            "ChemicalStructure": "ChemicalStructureLibrary",
            "BiologicalSequence": "BiologicalSequenceLibrary", 
            "Table": "Tables",
            "Figure": "Figures",
            "MoleculeType": "ChemicalStructureLibrary",
            "PersonType": "FrontPage",
            "OrganizationType": "FrontPage"
        }
        
        for ref_pattern, dependency in reference_mappings.items():
            if ref_pattern in field_str:
                dependencies.append(dependency)
        
        return list(set(dependencies))  # Remove duplicates
    
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
        """Get schema for specific field (for prompt use)"""
        # Search from regular properties
        if field_name in self.schema.get("properties", {}):
            field_property = self.schema["properties"][field_name]
            expanded_property = self._expand_definitions(field_property)
            
            return {
                "field_name": field_name,
                "schema": expanded_property
            }
        
        # Search from type definitions in definitions
        if field_name in self.schema.get("definitions", {}):
            field_definition = self.schema["definitions"][field_name]
            expanded_definition = self._expand_definitions(field_definition)
            
            return {
                "field_name": field_name,
                "schema": expanded_definition,
                "is_definition": True
            }
        
        return {
            "field_name": field_name,
            "schema": {"type": "object"}
        }
    
    def _expand_definitions(self, schema_part: Any) -> Any:
        """Expand $ref to inline definitions (for prompt use)"""
        if isinstance(schema_part, dict):
            if "$ref" in schema_part:
                ref_path = schema_part["$ref"]
                if ref_path.startswith("#/definitions/"):
                    def_name = ref_path.replace("#/definitions/", "")
                    definitions = self.schema.get("definitions", {})
                    if def_name in definitions:
                        # Depth limit to avoid circular references
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
        """Generate description for prompt use from schema information"""
        schema = schema_info.get("schema", {})
        
        # Generate structure description from schema
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
        
        return f"""Extract the {field_name} section and return it in the following JSON structure:

{field_name}: {schema_description}

Important:
- Return valid JSON only
- Follow the exact structure shown above
- Include all available information from the PDF
- Use null for missing values
- Ensure proper JSON formatting
"""
    
    def _create_field_prompt(self, field_name: str, dependency_context: str = "") -> str:
        """Create field-specific prompts"""
        
        # Basic property prompts
        property_prompts = {
            "publicationIdentifier": """Extract the patent publication number (e.g., WO2020162638A1, JP2020-123456A, US10123456B2, etc.).
Check the number listed at the top of the front page.""",
            "FrontPage": """Extract the following information from the first page (front page) of the PDF:
- Publication number, publication date, application number, application date
- Inventor information (name, address)
- Applicant information (name, address)
- Agent information (if any)
- International Patent Classification (IPC)
- Abstract
- Priority data if included
Focus on accuracy and extract information according to the front page layout.""",
            "Claims": """Extract all claims from the Claims section.
Include claim number and text for each claim.
Include references to chemical structures and tables.
Clearly identify the relationship between independent and dependent claims.
Include references to biological sequences if any.""",
            "Description": """Extract the following sections from the detailed description:
- Technical Field
- Background Art
- Summary of Invention
  - Problem to Solve
  - Means for Solving Problem
  - Effects of Invention
- Detailed Description of Invention
- Examples
Maintain the structure and content of each section and accurately extract the hierarchical structure.""",
            "ChemicalStructureLibrary": """Extract chemical structures, chemical formulas, and compounds from the entire patent document.
Include the following information:
- Compound numbers, SMILES, molecular formulas, chemical names
- Detailed information on atoms and bonds
- Stereochemical information
- Functional group information
- Patent-specific information (compound numbers, activity data, synthesis references, etc.)
Include references to chemical structure images.
Include uses and properties of each compound if described.""",
            "BiologicalSequenceLibrary": """Extract biological sequences (proteins, DNA, RNA) from the entire patent document.
Include the following information:
- Sequence ID (SEQ ID NO), sequence information, species, functional information
- For protein sequences: amino acid sequence, molecular weight, isoelectric point, functional domains
- For nucleic acid sequences: nucleotide sequence, sequence type (DNA/RNA), genetic elements
- Role in patent (antigen, antibody, enzyme, etc.)
Refer to sequence listing section if available.""",
            "Figures": """Extract figures from the entire patent document.
Include figure numbers, captions, and reference information.
Extract figure descriptions as much as possible.
Identify figure types (chemical structure diagrams, flowcharts, graphs, etc.).""",
            "Tables": """Extract tables from the entire patent document.
Completely extract the following information:
- Table structure, headers, data, captions
- Table numbers and position information
- Units and annotations for numerical data
- Related information for chemical compounds or biological data
- Table types (experimental data, comparative data, analytical data, etc.)
- Include statistical information if any""",
            "IndustrialApplicability": """Extract sections related to industrial applicability.
Include application fields, methods of use, and industrial impact.
Identify industrial sectors such as pharmaceuticals, chemicals, biotechnology, etc.""",
            "InternationalSearchReport": """Extract information from the international search report.
Include the following information:
- List of cited documents
- Search fields
- Content of written opinion
- Comments on patentability
Include tabular data if any.""",
            "PatentFamilyInformation": """Extract patent family information.
Include the following information:
- List of related patents
- Family structure
- Priority information
- Application status in each country
Include tabular data if any.""",
            "FrontPageContinuation": """Extract continuation information from the front page.
Include the following information:
- Designated country information
- F-Term classification
- Other classification information
- Additional inventor information
- Abstract continuation"""
        }
        
        # Type definition prompts
        definition_prompts = {
            # Basic structure types
            "PersonType": """Extract structured data for person information.
Accurately extract personal information including name and address.""",
            "OrganizationType": """Extract structured data for organization/entity information.
Accurately extract corporate information including organization name and address.""",
            "ClaimType": """Extract detailed structure of claims.
Include claim number, text, chemical structure references, biological sequence references, and table references.""",
            "SectionType": """Extract structured data for sections.
Include title, paragraphs, chemical structures, figures, and table references.""",
            "ExamplesType": """Extract structure of examples section.
Structure as an array of individual examples.""",
            "ExampleType": """Extract detailed structure of individual examples.
Include example ID, title, paragraphs, chemical structures, figures, and table references.""",
            "ParagraphType": """Extract structured data for paragraphs.
Include paragraph ID and content text.""",
            "TableType": """Extract complete structure and content of tables.
Include the following detailed information:
- Basic table information (ID, number, title, position)
- Table structure (number of rows, columns, header information, span information)
- Complete structure of row data
- Cell data (content, data type, numerical information, format)
- Chemical context (compound references, measurement conditions, etc.)
- Statistical information (error representation, sample size, etc.)
- Reference information (figure/table references, example references, etc.)""",
            "TableRowType": """Extract details of table row structure.
Include row number, row type, cell data, and row context.""",
            "TableCellType": """Extract details of table cell structure.
Include column number, content, data type, numerical information, format, and reference information.""",
            "TableRefType": """Extract reference information to tables.
Include reference ID, table ID, number, title, and context.""",
            "ChemicalStructureType": """Extract detailed information of chemical structures.
Include identifier, compound ID, image reference, structure data, and patent context.""",
            "PatentChemicalCompoundType": """Extract complete information of patent chemical compounds.
Include molecular structure information and patent-specific information (compound numbers, activity data, synthesis references, etc.).""",
            "MoleculeType": """Extract detailed structural information of chemical molecules.
Include the following detailed information:
- Molecular identifiers (name, IUPAC name, CAS number, SMILES, InChI, InChI Key, etc.)
- Molecular formulas (molecular formula, empirical formula, structural formula)
- Detailed atom information (element, atomic number, coordinates, charge, stereochemistry, etc.)
- Bond information (bond type, bond order, stereochemistry, aromaticity, etc.)
- Ring structure information (ring size, aromaticity, conformation, etc.)
- Functional group information (hydroxyl, carbonyl, amino, and other functional groups)
- Stereochemical information (chirality, optical activity, stereoisomer information, etc.)
- Molecular properties (molecular weight, exact mass, dipole moment, etc.)
Extract as detailed molecular structure data as possible.""",
            "FigureRefType": """Extract reference information to figures.
Include image reference, caption, number, and type.""",
            "ImageRefType": """Extract detailed information of image references.
Include ID, reference ID, source, alternative text, and type.""",
            "ProteinSequenceType": """Extract detailed information of protein sequences.
Include the following detailed information:
- Sequence identifiers (name, gene name, species, UniProt ID, etc.)
- Amino acid sequence (single letter code) and length
- Molecular properties (molecular weight, isoelectric point, etc.)
- Functional domains (signal peptide, catalytic domain, binding domain, etc.)
- Structural features (secondary structure, disulfide bonds, etc.)
- Post-translational modifications (phosphorylation, glycosylation, etc.)
- Role in patent and activity data""",
            "NucleicAcidSequenceType": """Extract detailed information of nucleic acid sequences.
Include the following detailed information:
- Sequence identifiers (name, gene name, species, GenBank ID, etc.)
- Nucleotide sequence (IUPAC codes) and length
- Sequence type (DNA, RNA, cDNA, etc.)
- GC content and complement sequence
- Genetic elements (promoter, enhancer, coding region, etc.)
- Variation information (substitution, insertion, deletion, etc.)
- Role in patent (vector, probe, therapeutic, etc.)""",
            "BiologicalSequenceRefType": """Extract reference information to biological sequences.
Include sequence ID, sequence type, image reference, display format, position information, and context.""",
            "DesignatedCountriesType": """Extract structure of designated country information.
Include country information by regional groups.""",
            "RegionGroupType": """Extract detailed information of regional groups.
Include country list and type.""",
            "FTermsType": """Extract structure of F-Term classification.
Structure as an array of individual F-Terms.""",
            "FTermType": """Extract detailed information of individual F-Terms.
Include code and description."""
        }
        
        # Get prompt
        if field_name in property_prompts:
            base_prompt = property_prompts[field_name]
        elif field_name in definition_prompts:
            base_prompt = definition_prompts[field_name]
        else:
            base_prompt = f"Extract structured data for {field_name}. Return as JSON according to schema."
        
        if dependency_context:
            return f"{base_prompt}\n\n{dependency_context}"
        
        return base_prompt
    
    def process_patent_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        High-speed extraction of patent PDF with parallel processing
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing structured patent information
        """
        start_time = time.time()
        logger.info(f"Starting parallel processing of PDF: {pdf_path}")
        logger.info(f"Processing {len(self.field_definitions)} fields: {list(self.field_definitions.keys())}")
        
        try:
            result = self._process_parallel_waves(pdf_path)
            
            # Set publication number from filename if not present
            if "publicationIdentifier" not in result or not result["publicationIdentifier"]:
                result["publicationIdentifier"] = Path(pdf_path).stem
            
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            
            # Add performance information
            result["_processing_info"] = {
                "total_time_seconds": total_time,
                "field_timing": self._timing_data,
                "parallel_workers": self.max_workers,
                "model_used": self.model_name,
                "fields_processed": len(self.field_definitions),
                "successful_fields": len([k for k, v in result.items() if not k.startswith("_") and v is not None])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                "error": str(e),
                "publicationIdentifier": Path(pdf_path).stem
            }
    
    def _process_parallel_waves(self, pdf_path: str) -> Dict[str, Any]:
        """Wave-based parallel processing (achieving maximum parallelism)"""
        logger.info("Starting wave-based parallel processing")
        
        # Group fields by wave
        wave_groups = {}
        for field_name, field_info in self.field_definitions.items():
            wave = field_info.get("wave", 999)
            if wave not in wave_groups:
                wave_groups[wave] = []
            wave_groups[wave].append(field_name)
        
        final_result = {}
        
        # Process in wave order (fully parallel within each wave)
        for wave in sorted(wave_groups.keys()):
            fields_in_wave = wave_groups[wave]
            wave_start_time = time.time()
            
            logger.info(f"Processing Wave {wave} with {len(fields_in_wave)} fields: {fields_in_wave}")
            
            # Process all fields in wave in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_field = {
                    executor.submit(self._extract_field_with_timing, pdf_path, field_name): field_name
                    for field_name in fields_in_wave
                }
                
                # Collect results
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
                
                # Merge wave results
                final_result.update(wave_results)
                
                # Batch update shared data (for next wave dependencies)
                with self._data_lock:
                    self._shared_data.update(wave_results)
                
                wave_time = time.time() - wave_start_time
                logger.info(f"Wave {wave} completed in {wave_time:.2f} seconds")
        
        return final_result
    
    def _extract_field_with_timing(self, pdf_path: str, field_name: str) -> Dict[str, Any]:
        """Field extraction with timing measurement"""
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
        """Extract specific field (prompt-based)"""
        schema_info = self._get_field_schema(field_name)
        field_prompt = self._create_field_prompt(field_name)
        schema_prompt = self._create_schema_prompt(field_name, schema_info)
        
        # Quick acquisition of dependency context
        dependency_context = self._get_dependency_context(field_name)
        
        full_prompt = f"{field_prompt}\n\n{schema_prompt}"
        if dependency_context:
            full_prompt += f"\n\n{dependency_context}"
        
        # Processing according to model type (all prompt-based)
        if "gemini" in self.model_name.lower():
            return self._process_field_with_gemini_prompt(pdf_path, full_prompt)
        elif "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            return self._process_field_with_openai_prompt(pdf_path, full_prompt)
        elif "claude" in self.model_name.lower():
            return self._process_field_with_anthropic_prompt(pdf_path, full_prompt)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _get_dependency_context(self, field_name: str) -> str:
        """Quick acquisition of dependency context"""
        dependencies = self.field_definitions.get(field_name, {}).get("dependencies", [])
        
        if not dependencies:
            return ""
        
        # Read lock (for acceleration)
        with self._data_lock:
            available_contexts = []
            for dep in dependencies:
                if dep in self._shared_data and self._shared_data[dep] is not None:
                    # Create summary version of context (avoid large data)
                    dep_data = self._shared_data[dep]
                    if isinstance(dep_data, dict):
                        summary = self._create_context_summary(dep_data, dep)
                        available_contexts.append(f"[{dep} Reference Information]\n{summary}")
            
            if available_contexts:
                return "\n\nPlease refer to the following related information:\n" + "\n".join(available_contexts)
        
        return ""
    
    def _create_context_summary(self, data: Dict[str, Any], field_name: str) -> str:
        """Create context summary (avoid large data issues)"""
        if field_name == "Claims":
            # Summary of claims
            claims = data.get("Claim", [])
            return f"Number of claims: {len(claims)} items"
        
        elif field_name == "Description":
            # Summary of description
            sections = []
            for section_name in ["TechnicalField", "BackgroundArt", "SummaryOfInvention", "DetailedDescriptionOfInvention"]:
                if section_name in data:
                    sections.append(section_name)
            return f"Included sections: {', '.join(sections)}"
        
        elif field_name == "ChemicalStructureLibrary":
            # Summary of chemical structure library
            compounds = data.get("Compound", [])
            return f"Number of chemical compounds: {len(compounds)}"
        
        elif field_name == "BiologicalSequenceLibrary":
            # Summary of biological sequence library
            proteins = data.get("ProteinSequence", [])
            nucleic_acids = data.get("NucleicAcidSequence", [])
            return f"Protein sequences: {len(proteins)}, Nucleic acid sequences: {len(nucleic_acids)}"
        
        elif field_name == "Tables":
            # Summary of tables
            tables = data.get("Table", [])
            return f"Number of tables: {len(tables)}"
        
        elif field_name == "Figures":
            # Summary of figures
            figures = data.get("Figure", [])
            return f"Number of figures: {len(figures)}"
        
        else:
            # General summary for others
            return f"Data structure: {list(data.keys())[:5]}"  # Only first 5 keys
    
    def _process_field_with_gemini_prompt(self, pdf_path: str, prompt: str) -> Dict[str, Any]:
        """Process field with Gemini (prompt-based)"""
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
        """Process field with OpenAI (using file type)"""
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
        
        return self._extract_json_from_text(response.choices[0].message.content)
    
    def _process_field_with_anthropic_prompt(self, pdf_path: str, prompt: str) -> Dict[str, Any]:
        """Process field with Anthropic (prompt-based)"""
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
        """Extract JSON from text (for Anthropic)"""
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

    def get_field_list(self) -> Dict[str, Dict]:
        """Get list of available fields"""
        return self.field_definitions
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate result against schema"""
        validation_report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "coverage": {}
        }
        
        if not self.schema:
            validation_report["warnings"].append("No schema provided for validation")
            return validation_report
        
        # Check required fields
        required_fields = self.schema.get("required", [])
        for field in required_fields:
            if field not in result or result[field] is None:
                validation_report["errors"].append(f"Required field '{field}' is missing or null")
                validation_report["is_valid"] = False
            else:
                validation_report["coverage"][field] = "present"
        
        # Check coverage of optional fields
        schema_properties = self.schema.get("properties", {})
        for field in schema_properties:
            if field not in required_fields:
                if field in result and result[field] is not None:
                    validation_report["coverage"][field] = "present"
                else:
                    validation_report["coverage"][field] = "missing"
        
        # Basic data type check
        for field_name, field_value in result.items():
            if field_name.startswith("_"):  # Skip metadata fields
                continue
                
            if field_name in schema_properties:
                expected_type = schema_properties[field_name].get("type")
                if expected_type and field_value is not None:
                    if expected_type == "object" and not isinstance(field_value, dict):
                        validation_report["warnings"].append(f"Field '{field_name}' should be object but got {type(field_value).__name__}")
                    elif expected_type == "array" and not isinstance(field_value, list):
                        validation_report["warnings"].append(f"Field '{field_name}' should be array but got {type(field_value).__name__}")
                    elif expected_type == "string" and not isinstance(field_value, str):
                        validation_report["warnings"].append(f"Field '{field_name}' should be string but got {type(field_value).__name__}")
        
        return validation_report

def main():
    """Main function for command line execution"""
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
    parser.add_argument('--validate', action='store_true', help='Validate result against schema')
    parser.add_argument('--list-fields', action='store_true', help='List all available fields and exit')
    
    args = parser.parse_args()
    
    # Load schema
    schema = None
    if args.schema:
        with open(args.schema, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    
    # Initialize extractor
    extractor = PatentExtractor(
        model_name=args.model,
        api_key=args.api_key,
        json_schema=schema,
        user_prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers
    )
    
    # Display field list
    if args.list_fields:
        print("Available fields:")
        field_list = extractor.get_field_list()
        for wave in sorted(set(info["wave"] for info in field_list.values())):
            print(f"\nWave {wave}:")
            for field_name, field_info in field_list.items():
                if field_info["wave"] == wave:
                    deps = ", ".join(field_info["dependencies"]) if field_info["dependencies"] else "None"
                    print(f"  {field_name}: {field_info['description']} (deps: {deps})")
        return
    
    # Process PDF
    result = extractor.process_patent_pdf(args.pdf_path)
    
    # Validation
    if args.validate and schema:
        validation_report = extractor.validate_result(result)
        print(f"\nValidation Report:")
        print(f"Valid: {validation_report['is_valid']}")
        if validation_report['errors']:
            print(f"Errors: {validation_report['errors']}")
        if validation_report['warnings']:
            print(f"Warnings: {validation_report['warnings']}")
        print(f"Field Coverage: {len([k for k, v in validation_report['coverage'].items() if v == 'present'])}/{len(validation_report['coverage'])} fields")
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Output saved to {args.output}")
        
        # Display performance information
        if "_processing_info" in result:
            info = result["_processing_info"]
            print(f"\nPerformance Summary:")
            print(f"Total Time: {info['total_time_seconds']:.2f} seconds")
            print(f"Parallel Workers: {info['parallel_workers']}")
            print(f"Model: {info['model_used']}")
            print(f"Fields Processed: {info['fields_processed']}")
            print(f"Successful Fields: {info['successful_fields']}")
            print(f"\nField Processing Times:")
            for field, time_taken in sorted(info['field_timing'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {field}: {time_taken:.2f}s")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()