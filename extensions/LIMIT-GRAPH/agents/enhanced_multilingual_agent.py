# -*- coding: utf-8 -*-
"""
Enhanced Multilingual Research Agent for LIMIT-GRAPH
Supports 15+ languages with advanced NLP capabilities and cultural awareness
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
import unicodedata
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class SupportedLanguage(Enum):
    """Supported languages with ISO codes and metadata"""
    ENGLISH = ("en", "English", "ltr", "latin")
    ARABIC = ("ar", "العربية", "rtl", "arabic")
    CHINESE_SIMPLIFIED = ("zh-cn", "中文(简体)", "ltr", "han")
    CHINESE_TRADITIONAL = ("zh-tw", "中文(繁體)", "ltr", "han")
    SPANISH = ("es", "Español", "ltr", "latin")
    FRENCH = ("fr", "Français", "ltr", "latin")
    GERMAN = ("de", "Deutsch", "ltr", "latin")
    JAPANESE = ("ja", "日本語", "ltr", "mixed")
    KOREAN = ("ko", "한국어", "ltr", "hangul")
    RUSSIAN = ("ru", "Русский", "ltr", "cyrillic")
    HINDI = ("hi", "हिन्दी", "ltr", "devanagari")
    INDONESIAN = ("id", "Bahasa Indonesia", "ltr", "latin")
    PORTUGUESE = ("pt", "Português", "ltr", "latin")
    ITALIAN = ("it", "Italiano", "ltr", "latin")
    DUTCH = ("nl", "Nederlands", "ltr", "latin")
    TURKISH = ("tr", "Türkçe", "ltr", "latin")
    PERSIAN = ("fa", "فارسی", "rtl", "arabic")
    URDU = ("ur", "اردو", "rtl", "arabic")
    BENGALI = ("bn", "বাংলা", "ltr", "bengali")
    THAI = ("th", "ไทย", "ltr", "thai")

    def __init__(self, code, name, direction, script):
        self.code = code
        self.name = name
        self.direction = direction
        self.script = script

@dataclass
class MultilingualEntity:
    """Enhanced entity with multilingual support"""
    text: str
    normalized: str
    language: SupportedLanguage
    entity_type: str
    confidence: float
    translations: Dict[str, str]
    cultural_context: Dict[str, Any]
    semantic_embeddings: Optional[List[float]] = None

@dataclass
class CrossLingualAlignment:
    """Cross-lingual entity alignment"""
    source_entity: str
    target_entity: str
    source_lang: SupportedLanguage
    target_lang: SupportedLanguage
    alignment_score: float
    alignment_method: str

class EnhancedMultilingualAgent:
    """
    Enhanced multilingual research agent with advanced NLP capabilities
    """
    
    def __init__(self, primary_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.primary_language = primary_language
        self.logger = logging.getLogger(__name__)
        
        # Initialize language-specific components
        self.language_processors = {}
        self.translation_cache = {}
        self.cultural_knowledge = self._load_cultural_knowledge()
        
        # Initialize multilingual models if available
        self.multilingual_model = None
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            self._initialize_multilingual_models()
        
        # Initialize spaCy models if available
        self.spacy_models = {}
        if SPACY_AVAILABLE:
            self._initialize_spacy_models()
        
        # Language-specific patterns and rules
        self.language_patterns = self._initialize_language_patterns()
        
        # Cross-lingual mappings
        self.cross_lingual_mappings = self._load_cross_lingual_mappings()
    
    def _initialize_multilingual_models(self):
        """Initialize multilingual transformer models"""
        try:
            # Use multilingual BERT or similar model
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.multilingual_model = AutoModel.from_pretrained(model_name)
            self.logger.info(f"Initialized multilingual model: {model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize multilingual models: {e}")
    
    def _initialize_spacy_models(self):
        """Initialize spaCy models for supported languages"""
        spacy_models = {
            SupportedLanguage.ENGLISH: "en_core_web_sm",
            SupportedLanguage.GERMAN: "de_core_news_sm",
            SupportedLanguage.SPANISH: "es_core_news_sm",
            SupportedLanguage.FRENCH: "fr_core_news_sm",
            SupportedLanguage.ITALIAN: "it_core_news_sm",
            SupportedLanguage.PORTUGUESE: "pt_core_news_sm",
            SupportedLanguage.DUTCH: "nl_core_news_sm",
            SupportedLanguage.CHINESE_SIMPLIFIED: "zh_core_web_sm",
            SupportedLanguage.JAPANESE: "ja_core_news_sm"
        }
        
        for lang, model_name in spacy_models.items():
            try:
                self.spacy_models[lang] = spacy.load(model_name)
                self.logger.info(f"Loaded spaCy model for {lang.name}")
            except OSError:
                self.logger.warning(f"spaCy model {model_name} not available for {lang.name}")
    
    def _load_cultural_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural knowledge base for different languages/regions"""
        return {
            "ar": {
                "greeting_patterns": ["السلام عليكم", "أهلا وسهلا", "مرحبا"],
                "formal_titles": ["الدكتور", "الأستاذ", "المهندس", "الشيخ"],
                "cultural_concepts": ["الحج", "رمضان", "العيد", "الصلاة"],
                "naming_conventions": {
                    "pattern": "اسم + اسم الأب + اسم الجد + اللقب",
                    "prefixes": ["أبو", "أم", "ابن", "بنت"]
                }
            },
            "zh-cn": {
                "greeting_patterns": ["你好", "您好", "早上好", "晚上好"],
                "formal_titles": ["先生", "女士", "老师", "教授", "博士"],
                "cultural_concepts": ["春节", "中秋节", "端午节", "清明节"],
                "naming_conventions": {
                    "pattern": "姓 + 名",
                    "common_surnames": ["王", "李", "张", "刘", "陈"]
                }
            },
            "ja": {
                "greeting_patterns": ["こんにちは", "おはようございます", "こんばんは"],
                "formal_titles": ["さん", "先生", "博士", "教授"],
                "cultural_concepts": ["正月", "桜", "茶道", "武道"],
                "naming_conventions": {
                    "pattern": "姓 + 名",
                    "honorifics": ["さん", "様", "君", "ちゃん"]
                }
            },
            "hi": {
                "greeting_patterns": ["नमस्ते", "नमस्कार", "आदाब"],
                "formal_titles": ["जी", "साहब", "डॉक्टर", "प्रोफेसर"],
                "cultural_concepts": ["दिवाली", "होली", "दशहरा", "करवा चौथ"],
                "naming_conventions": {
                    "pattern": "नाम + पिता का नाम + उपनाम",
                    "prefixes": ["श्री", "श्रीमती", "कुमार", "कुमारी"]
                }
            }
        }
    
    def _initialize_language_patterns(self) -> Dict[SupportedLanguage, Dict[str, Any]]:
        """Initialize language-specific patterns for entity extraction"""
        patterns = {}
        
        # English patterns
        patterns[SupportedLanguage.ENGLISH] = {
            "person_patterns": [
                r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
                r"\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+\b"
            ],
            "location_patterns": [
                r"\b[A-Z][a-z]+ (?:City|State|Country|University|Hospital)\b",
                r"\bin [A-Z][a-z]+(?:, [A-Z][a-z]+)*\b"
            ],
            "organization_patterns": [
                r"\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|University|Hospital)\b",
                r"\bthe [A-Z][a-z]+ (?:Company|Corporation|Institute)\b"
            ]
        }
        
        # Arabic patterns
        patterns[SupportedLanguage.ARABIC] = {
            "person_patterns": [
                r"(?:الدكتور|الأستاذ|المهندس)\s+[\u0600-\u06FF]+",
                r"[\u0600-\u06FF]+\s+(?:بن|ابن)\s+[\u0600-\u06FF]+",
                r"(?:أبو|أم)\s+[\u0600-\u06FF]+"
            ],
            "location_patterns": [
                r"(?:مدينة|قرية|محافظة)\s+[\u0600-\u06FF]+",
                r"[\u0600-\u06FF]+\s+(?:الشريف|المقدسة|المنورة)"
            ],
            "organization_patterns": [
                r"(?:شركة|مؤسسة|جامعة|وزارة)\s+[\u0600-\u06FF]+",
                r"[\u0600-\u06FF]+\s+(?:المحدودة|والشركاه)"
            ]
        }
        
        # Chinese patterns
        patterns[SupportedLanguage.CHINESE_SIMPLIFIED] = {
            "person_patterns": [
                r"[\u4e00-\u9fff]{1,2}(?:先生|女士|老师|教授|博士)",
                r"[\u4e00-\u9fff]{2,4}(?=\s|$|，|。)"
            ],
            "location_patterns": [
                r"[\u4e00-\u9fff]+(?:市|省|县|区|镇|村)",
                r"[\u4e00-\u9fff]+(?:大学|医院|公司|学校)"
            ],
            "organization_patterns": [
                r"[\u4e00-\u9fff]+(?:公司|集团|企业|机构|组织)",
                r"[\u4e00-\u9fff]+(?:大学|学院|研究所)"
            ]
        }
        
        return patterns
    
    def _load_cross_lingual_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load cross-lingual entity mappings"""
        return {
            # Common entities with translations
            "person_titles": {
                "en": {"doctor": "Dr.", "professor": "Prof.", "mister": "Mr."},
                "ar": {"doctor": "الدكتور", "professor": "الأستاذ", "mister": "السيد"},
                "zh-cn": {"doctor": "博士", "professor": "教授", "mister": "先生"},
                "es": {"doctor": "Dr.", "professor": "Prof.", "mister": "Sr."},
                "fr": {"doctor": "Dr.", "professor": "Prof.", "mister": "M."}
            },
            "locations": {
                "en": {"university": "University", "hospital": "Hospital", "school": "School"},
                "ar": {"university": "جامعة", "hospital": "مستشفى", "school": "مدرسة"},
                "zh-cn": {"university": "大学", "hospital": "医院", "school": "学校"},
                "es": {"university": "Universidad", "hospital": "Hospital", "school": "Escuela"},
                "fr": {"university": "Université", "hospital": "Hôpital", "school": "École"}
            }
        }
    
    def detect_language(self, text: str) -> Tuple[SupportedLanguage, float]:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        # Simple character-based detection
        script_counts = defaultdict(int)
        
        for char in text:
            if '\u0600' <= char <= '\u06FF':  # Arabic
                script_counts['arabic'] += 1
            elif '\u4e00' <= char <= '\u9fff':  # Chinese
                script_counts['han'] += 1
            elif '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':  # Japanese
                script_counts['japanese'] += 1
            elif '\uac00' <= char <= '\ud7af':  # Korean
                script_counts['korean'] += 1
            elif '\u0900' <= char <= '\u097f':  # Devanagari (Hindi)
                script_counts['devanagari'] += 1
            elif '\u0400' <= char <= '\u04ff':  # Cyrillic (Russian)
                script_counts['cyrillic'] += 1
            elif 'a' <= char.lower() <= 'z':  # Latin
                script_counts['latin'] += 1
        
        total_chars = sum(script_counts.values())
        if total_chars == 0:
            return SupportedLanguage.ENGLISH, 0.5
        
        # Determine primary script
        primary_script = max(script_counts.items(), key=lambda x: x[1])
        confidence = primary_script[1] / total_chars
        
        # Map script to language
        script_to_language = {
            'arabic': SupportedLanguage.ARABIC,
            'han': SupportedLanguage.CHINESE_SIMPLIFIED,
            'japanese': SupportedLanguage.JAPANESE,
            'korean': SupportedLanguage.KOREAN,
            'devanagari': SupportedLanguage.HINDI,
            'cyrillic': SupportedLanguage.RUSSIAN,
            'latin': self._detect_latin_language(text)
        }
        
        detected_lang = script_to_language.get(primary_script[0], SupportedLanguage.ENGLISH)
        return detected_lang, confidence
    
    def _detect_latin_language(self, text: str) -> SupportedLanguage:
        """Detect specific Latin-script language"""
        # Simple keyword-based detection for Latin scripts
        language_keywords = {
            SupportedLanguage.SPANISH: ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
            SupportedLanguage.FRENCH: ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
            SupportedLanguage.GERMAN: ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            SupportedLanguage.ITALIAN: ['il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'non'],
            SupportedLanguage.PORTUGUESE: ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para'],
            SupportedLanguage.DUTCH: ['de', 'van', 'het', 'een', 'en', 'in', 'op', 'dat', 'met', 'voor'],
            SupportedLanguage.INDONESIAN: ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah', 'ini']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for lang, keywords in language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[lang] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return SupportedLanguage.ENGLISH  # Default fallback
    
    def normalize_text(self, text: str, language: SupportedLanguage) -> str:
        """
        Normalize text based on language-specific rules
        
        Args:
            text: Input text to normalize
            language: Target language
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKC', text)
        
        # Language-specific normalization
        if language in [SupportedLanguage.ARABIC, SupportedLanguage.PERSIAN, SupportedLanguage.URDU]:
            normalized = self._normalize_arabic_script(normalized)
        elif language in [SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.CHINESE_TRADITIONAL]:
            normalized = self._normalize_chinese(normalized)
        elif language == SupportedLanguage.JAPANESE:
            normalized = self._normalize_japanese(normalized)
        
        # General cleanup
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _normalize_arabic_script(self, text: str) -> str:
        """Normalize Arabic script text"""
        # Remove diacritics
        diacritics = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F\u0670'
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
        
        # Normalize Arabic letters
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')
        text = text.replace('ى', 'ي')
        
        return text
    
    def _normalize_chinese(self, text: str) -> str:
        """Normalize Chinese text"""
        # Convert traditional to simplified (basic mapping)
        traditional_to_simplified = {
            '繁': '繁', '體': '体', '語': '语', '學': '学', '國': '国'
        }
        
        for trad, simp in traditional_to_simplified.items():
            text = text.replace(trad, simp)
        
        return text
    
    def _normalize_japanese(self, text: str) -> str:
        """Normalize Japanese text"""
        # Convert full-width to half-width for ASCII characters
        normalized = ""
        for char in text:
            if '\uff01' <= char <= '\uff5e':
                normalized += chr(ord(char) - 0xfee0)
            else:
                normalized += char
        
        return normalized
    
    def extract_multilingual_entities(self, text: str, language: SupportedLanguage = None) -> List[MultilingualEntity]:
        """
        Extract entities from multilingual text
        
        Args:
            text: Input text
            language: Target language (auto-detect if None)
            
        Returns:
            List of extracted multilingual entities
        """
        if language is None:
            language, _ = self.detect_language(text)
        
        entities = []
        
        # Use spaCy if available
        if language in self.spacy_models:
            entities.extend(self._extract_entities_spacy(text, language))
        
        # Use pattern-based extraction
        entities.extend(self._extract_entities_patterns(text, language))
        
        # Enhance with multilingual embeddings
        if self.multilingual_model and TRANSFORMERS_AVAILABLE:
            entities = self._enhance_with_embeddings(entities, text)
        
        return entities
    
    def _extract_entities_spacy(self, text: str, language: SupportedLanguage) -> List[MultilingualEntity]:
        """Extract entities using spaCy"""
        entities = []
        nlp = self.spacy_models[language]
        doc = nlp(text)
        
        for ent in doc.ents:
            entity = MultilingualEntity(
                text=ent.text,
                normalized=self.normalize_text(ent.text, language),
                language=language,
                entity_type=ent.label_,
                confidence=0.8,  # spaCy doesn't provide confidence scores
                translations={},
                cultural_context=self._get_cultural_context(ent.text, language)
            )
            entities.append(entity)
        
        return entities
    
    def _extract_entities_patterns(self, text: str, language: SupportedLanguage) -> List[MultilingualEntity]:
        """Extract entities using pattern matching"""
        entities = []
        
        if language not in self.language_patterns:
            return entities
        
        patterns = self.language_patterns[language]
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_text = match.group()
                    entity = MultilingualEntity(
                        text=entity_text,
                        normalized=self.normalize_text(entity_text, language),
                        language=language,
                        entity_type=entity_type.replace('_patterns', ''),
                        confidence=0.7,
                        translations={},
                        cultural_context=self._get_cultural_context(entity_text, language)
                    )
                    entities.append(entity)
        
        return entities
    
    def _get_cultural_context(self, entity_text: str, language: SupportedLanguage) -> Dict[str, Any]:
        """Get cultural context for entity"""
        lang_code = language.code
        if lang_code in self.cultural_knowledge:
            cultural_data = self.cultural_knowledge[lang_code]
            
            context = {}
            
            # Check for cultural concepts
            if 'cultural_concepts' in cultural_data:
                for concept in cultural_data['cultural_concepts']:
                    if concept in entity_text:
                        context['cultural_significance'] = concept
                        break
            
            # Check for formal titles
            if 'formal_titles' in cultural_data:
                for title in cultural_data['formal_titles']:
                    if title in entity_text:
                        context['formal_title'] = title
                        break
            
            return context
        
        return {}
    
    def _enhance_with_embeddings(self, entities: List[MultilingualEntity], text: str) -> List[MultilingualEntity]:
        """Enhance entities with semantic embeddings"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.multilingual_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            
            for entity in entities:
                entity.semantic_embeddings = embeddings
        
        except Exception as e:
            self.logger.warning(f"Failed to generate embeddings: {e}")
        
        return entities
    
    def align_cross_lingual_entities(self, entities1: List[MultilingualEntity], 
                                   entities2: List[MultilingualEntity]) -> List[CrossLingualAlignment]:
        """
        Align entities across languages
        
        Args:
            entities1: Entities from first language
            entities2: Entities from second language
            
        Returns:
            List of cross-lingual alignments
        """
        alignments = []
        
        for ent1 in entities1:
            for ent2 in entities2:
                if ent1.entity_type == ent2.entity_type:
                    # Calculate alignment score
                    score = self._calculate_alignment_score(ent1, ent2)
                    
                    if score > 0.5:  # Threshold for alignment
                        alignment = CrossLingualAlignment(
                            source_entity=ent1.text,
                            target_entity=ent2.text,
                            source_lang=ent1.language,
                            target_lang=ent2.language,
                            alignment_score=score,
                            alignment_method="semantic_similarity"
                        )
                        alignments.append(alignment)
        
        return sorted(alignments, key=lambda x: x.alignment_score, reverse=True)
    
    def _calculate_alignment_score(self, ent1: MultilingualEntity, ent2: MultilingualEntity) -> float:
        """Calculate alignment score between two entities"""
        score = 0.0
        
        # Semantic embedding similarity
        if ent1.semantic_embeddings and ent2.semantic_embeddings:
            score += self._cosine_similarity(ent1.semantic_embeddings, ent2.semantic_embeddings) * 0.6
        
        # Normalized text similarity
        norm_sim = self._string_similarity(ent1.normalized, ent2.normalized)
        score += norm_sim * 0.3
        
        # Cultural context similarity
        if ent1.cultural_context and ent2.cultural_context:
            context_sim = len(set(ent1.cultural_context.keys()) & set(ent2.cultural_context.keys())) / \
                         max(len(ent1.cultural_context), len(ent2.cultural_context), 1)
            score += context_sim * 0.1
        
        return min(score, 1.0)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple Levenshtein distance implementation
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        max_len = max(m, n)
        return 1.0 - (dp[m][n] / max_len) if max_len > 0 else 0.0
    
    def process_multilingual_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Process a multilingual query with comprehensive analysis
        
        Args:
            query: Input query in any supported language
            context: Optional context information
            
        Returns:
            Comprehensive analysis results
        """
        # Detect language
        detected_lang, lang_confidence = self.detect_language(query)
        
        # Extract entities
        entities = self.extract_multilingual_entities(query, detected_lang)
        
        # Process context if provided
        context_entities = []
        if context:
            context_lang, _ = self.detect_language(context)
            context_entities = self.extract_multilingual_entities(context, context_lang)
        
        # Cross-lingual alignment if different languages
        alignments = []
        if context_entities and detected_lang != context_entities[0].language:
            alignments = self.align_cross_lingual_entities(entities, context_entities)
        
        return {
            'query': query,
            'detected_language': {
                'language': detected_lang.name,
                'code': detected_lang.code,
                'confidence': lang_confidence,
                'direction': detected_lang.direction,
                'script': detected_lang.script
            },
            'entities': [
                {
                    'text': ent.text,
                    'normalized': ent.normalized,
                    'type': ent.entity_type,
                    'confidence': ent.confidence,
                    'cultural_context': ent.cultural_context
                } for ent in entities
            ],
            'context_entities': [
                {
                    'text': ent.text,
                    'normalized': ent.normalized,
                    'type': ent.entity_type,
                    'confidence': ent.confidence
                } for ent in context_entities
            ],
            'cross_lingual_alignments': [
                {
                    'source': align.source_entity,
                    'target': align.target_entity,
                    'source_lang': align.source_lang.code,
                    'target_lang': align.target_lang.code,
                    'score': align.alignment_score
                } for align in alignments
            ],
            'supported_languages': [lang.code for lang in SupportedLanguage],
            'processing_metadata': {
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'spacy_available': SPACY_AVAILABLE,
                'loaded_spacy_models': list(self.spacy_models.keys())
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced multilingual agent
    agent = EnhancedMultilingualAgent()
    
    # Test queries in different languages
    test_queries = [
        "Hello, my name is John Smith and I work at Microsoft.",
        "مرحبا، اسمي أحمد محمد وأعمل في جامعة الملك سعود.",
        "你好，我叫李明，在北京大学工作。",
        "こんにちは、私の名前は田中太郎です。東京大学で働いています。",
        "Hola, me llamo María García y trabajo en la Universidad de Barcelona.",
        "Bonjour, je m'appelle Pierre Dupont et je travaille à l'Université de Paris."
    ]
    
    for query in test_queries:
        print(f"\nProcessing: {query}")
        result = agent.process_multilingual_query(query)
        print(f"Language: {result['detected_language']['language']} ({result['detected_language']['code']})")
        print(f"Entities: {len(result['entities'])}")
        for entity in result['entities']:
            print(f"  - {entity['text']} ({entity['type']}, {entity['confidence']:.2f})")