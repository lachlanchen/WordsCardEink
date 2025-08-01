"""
Eink Words GPT Project - Updated with Structured OpenAI Outputs
----------------------------------------------------------------

Project Name: Eink Words GPT
Author: Lachlan CHEN
Website: https://lazying.art
GitHub: https://github.com/lachlanchen/

Description:
Updated version using structured OpenAI outputs for more reliable JSON parsing.
This version imports from the legacy module and creates new enhanced classes.
"""

import json
import traceback
import os
import csv
from typing import List, Dict, Any
from pprint import pprint
from datetime import datetime

# Import specific classes and functions from the legacy module
from words_data_utils import (
    OpenAiChooser, 
    EmojiWordChooser,
    split_word, 
    split_word_with_color, 
    count_syllables,
    clean_word_details,
    clean_and_transcribe,
    JSONParsingError,
    NotEnoughUniqueWordsError
)

from words_database import WordsDatabase
from openai_request_json import OpenAIRequestJSONBase


class AdvancedWordFetcher(OpenAIRequestJSONBase):
    """
    Enhanced AdvancedWordFetcher that uses structured OpenAI outputs with improved two-step processing
    """
    
    def __init__(self, max_retries=3, use_cache=True):
        super().__init__(use_cache, max_retries)
        self.examples = self.load_examples()
        self.model_name = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-0125-preview", "gpt-4-1106-preview"]
        
    def load_examples(self):
        """Load examples from CSV or use defaults"""
        examples_file_path = 'data/word_examples.csv'
        if os.path.exists(examples_file_path):
            try:
                with open(examples_file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    example_list = list(reader)
                    if len(example_list) > 1:
                       return example_list
            except Exception as e:
                print(f"Error loading examples: {e}")
                traceback.print_exc()

        return [
            {"word": "abstraction", "syllable_word": "ab·strac·tion", "phonetic": "ˈæb·stræk·ʃən", "japanese_synonym": "抽象（ちゅうしょう）"},
            {"word": "paradox", "syllable_word": "par·a·dox", "phonetic": "ˈpær·ə·dɒks", "japanese_synonym": "逆説（ぎゃくせつ）"}
        ]
        
    def save_examples(self):
        """Save examples to CSV"""
        try:
            if not self.examples or len(self.examples) == 0:
                print("No examples to save")
                return
                
            examples_file_path = 'data/word_examples.csv'
            os.makedirs(os.path.dirname(examples_file_path), exist_ok=True)
            
            with open(examples_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.examples[0].keys())
                writer.writeheader()
                writer.writerows(self.examples)
        except Exception as e:
            print(f"Error saving examples: {e}")
            traceback.print_exc()
        
    def load_propensities(self):
        """Load word propensities from file"""
        propensities_file_path = 'data/words_propensity.txt'
        propensities = []
        if os.path.exists(propensities_file_path):
            try:
                with open(propensities_file_path, 'r') as file:
                    propensities = [line.strip() for line in file if line.strip() and not line.startswith('#')]
            except Exception as e:
                print(f"Error loading propensities: {e}")
                traceback.print_exc()
        return propensities
        
    def fetch_words_local(self, num_words, word_database):
        """Fetch words from NLTK corpus"""
        import nltk
        from nltk.corpus import words
        from words_data_utils import random_shuffle
        
        try:
            nltk_word_list = words.words()
            unique_words = [word for word in nltk_word_list if not word_database.word_exists(word)]
            unique_words = random_shuffle(unique_words)
            
            if len(unique_words) < num_words:
                raise ValueError(f"Not enough unique words. Only {len(unique_words)} unique words found.")
            
            return unique_words[:num_words]
        except Exception as e:
            print(f"Error fetching local words: {e}")
            traceback.print_exc()
            raise
        
    def save_unused_words(self, words, random_words, file_path='data/unused_words.csv'):
        """Save unused words to CSV"""
        try:
            unused_words = set(words) - set(random_words)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for word in unused_words:
                    writer.writerow([word])
        except Exception as e:
            print(f"Error saving unused words: {e}")
            traceback.print_exc()
        
    def map_syllables_phonetics(self, syllables, phonetics):
        """Map syllables to phonetics for display"""
        try:
            max_length = max(len(syllables), len(phonetics))
            syllables.extend([''] * (max_length - len(syllables)))
            phonetics.extend([''] * (max_length - len(phonetics)))
            mapping = ', '.join([f"{syl} ↔ {phon}" for syl, phon in zip(syllables, phonetics)])
            return mapping
        except Exception as e:
            print(f"Error mapping syllables to phonetics: {e}")
            traceback.print_exc()
            return ""
    
    # ===== SCHEMAS =====
    
    def get_word_list_schema(self):
        """Schema for word list responses"""
        return {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["words"],
            "additionalProperties": False
        }
    
    def get_basic_phonetic_schema(self):
        """Schema for basic word phonetics (step 1)"""
        return {
            "type": "object",
            "properties": {
                "word_phonetics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "phonetic": {"type": "string"}
                        },
                        "required": ["word", "phonetic"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["word_phonetics"],
            "additionalProperties": False
        }
    
    def get_syllable_pair_schema(self):
        """Schema for syllable-phonetic pairs (step 2)"""
        return {
            "type": "object",
            "properties": {
                "syllable_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "pairs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "word_syllable": {"type": "string"},
                                        "phonetic_syllable": {"type": "string"}
                                    },
                                    "required": ["word_syllable", "phonetic_syllable"],
                                    "additionalProperties": False
                                },
                                "minItems": 1
                            }
                        },
                        "required": ["word", "pairs"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["syllable_pairs"],
            "additionalProperties": False
        }
    
    def get_basic_japanese_schema(self):
        """Schema for basic Japanese synonyms WITHOUT furigana (step 1)"""
        return {
            "type": "object",
            "properties": {
                "japanese_synonyms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "japanese_synonym": {"type": "string"}
                        },
                        "required": ["word", "japanese_synonym"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["japanese_synonyms"],
            "additionalProperties": False
        }
    
    def get_japanese_furigana_pairs_schema(self):
        """Schema for Japanese furigana pairs (step 2)"""
        return {
            "type": "object",
            "properties": {
                "furigana_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "character_pairs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "characters": {"type": "string"},
                                        "furigana": {"type": "string"},
                                        "needs_furigana": {"type": "boolean"}
                                    },
                                    "required": ["characters", "furigana", "needs_furigana"],
                                    "additionalProperties": False
                                },
                                "minItems": 1
                            }
                        },
                        "required": ["word", "character_pairs"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["furigana_pairs"],
            "additionalProperties": False
        }
    
    def get_arabic_transliteration_schema(self):
        """Schema for Arabic with transliteration"""
        return {
            "type": "object",
            "properties": {
                "arabic_words": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "arabic_synonym": {"type": "string"},
                            "transliteration": {"type": "string"}
                        },
                        "required": ["word", "arabic_synonym", "transliteration"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["arabic_words"],
            "additionalProperties": False
        }
    
    def get_french_phonetic_schema(self):
        """Schema for French with phonetics"""
        return {
            "type": "object",
            "properties": {
                "french_words": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "french_synonym": {"type": "string"},
                            "french_phonetic": {"type": "string"}
                        },
                        "required": ["word", "french_synonym", "french_phonetic"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["french_words"],
            "additionalProperties": False
        }
    
    # ===== ENGLISH PHONETICS (TWO-STEP PROCESS) =====
    
    def fetch_english_phonetics_step1(self, word):
        """Step 1: Get basic phonetics for a single word"""
        try:
            prompt = (
                f"Provide the IPA phonetic transcription for this English word: {word}\n"
                "Use standard IPA notation with stress markers (ˈ for primary stress, ˌ for secondary stress)."
            )
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_basic_phonetic_schema(),
                system_content="You are a phonetics expert. Provide accurate IPA transcriptions.",
                filename=f"phonetics_step1_{word}.json"
            )
            
            return response["word_phonetics"][0]  # Return first (and only) result
        except Exception as e:
            print(f"Error in fetch_english_phonetics_step1 for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    def fetch_english_phonetics_step2(self, word_phonetic):
        """Step 2: Separate into syllable-phonetic pairs for a single word"""
        try:
            word = word_phonetic["word"]
            phonetic = word_phonetic["phonetic"]
            
            prompt = (
                f"For the word '{word}' with phonetic transcription '{phonetic}', "
                "break them down into corresponding syllable pairs. "
                "Each word syllable must correspond exactly to a phonetic syllable. "
                "The concatenation of word syllables must equal the original word exactly. "
                "The concatenation of phonetic syllables must equal the original phonetic transcription exactly. "
                "Treat stress markers (ˈ ˌ) as part of the phonetic syllable they precede."
            )
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_syllable_pair_schema(),
                system_content="You are an expert in syllable division and phonetic transcription.",
                filename=f"phonetics_step2_{word}.json"
            )
            
            return self.validate_and_convert_syllable_pairs(response["syllable_pairs"])[0]
        except Exception as e:
            print(f"Error in fetch_english_phonetics_step2 for word '{word_phonetic.get('word', 'unknown')}': {e}")
            traceback.print_exc()
            raise
    
    def validate_and_convert_syllable_pairs(self, syllable_pairs):
        """Validate syllable pairs and convert to dot-separated format"""
        try:
            result = []
            
            for item in syllable_pairs:
                word = item["word"]
                pairs = item["pairs"]
                
                # Reconstruct word and phonetic from pairs
                word_syllables = [pair["word_syllable"] for pair in pairs]
                phonetic_syllables = [pair["phonetic_syllable"] for pair in pairs]
                
                reconstructed_word = ''.join(word_syllables)
                reconstructed_phonetic = ''.join(phonetic_syllables)
                
                # Validate reconstruction
                if reconstructed_word.lower() != word.lower():
                    raise ValueError(f"Word syllables don't match original: {reconstructed_word} != {word}")
                
                # Create dot-separated versions
                syllable_word = '·'.join(word_syllables)
                phonetic = '·'.join(phonetic_syllables)
                
                result.append({
                    "word": word,
                    "syllable_word": syllable_word,
                    "phonetic": phonetic
                })
            
            return result
        except Exception as e:
            print(f"Error validating syllable pairs: {e}")
            traceback.print_exc()
            raise
    
    def fetch_english_phonetics_complete(self, word, word_database=None):
        """Complete English phonetics processing for a single word (both steps)"""
        try:
            print(f"Processing English phonetics for: {word}")
            
            # Step 1: Get basic phonetics
            word_phonetic = self.fetch_english_phonetics_step1(word)
            
            # Step 2: Get syllable pairs and validate
            completed_phonetic = self.fetch_english_phonetics_step2(word_phonetic)
            
            # Save to database
            if word_database:
                word_database.update_word_details(completed_phonetic)
            
            return completed_phonetic
            
        except Exception as e:
            print(f"Error in English phonetics processing for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    # ===== JAPANESE WITH FURIGANA (TWO-STEP PROCESS) =====
    
    def fetch_japanese_synonym_step1(self, word):
        """Step 1: Get basic Japanese synonym WITHOUT furigana"""
        try:
            prompt = (
                f"Provide a Japanese synonym for this English word: {word}\n"
                "Use appropriate Japanese characters (kanji, hiragana, katakana) but DO NOT include furigana readings in parentheses. "
                "Just provide the bare Japanese text without any pronunciation guides."
            )
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_basic_japanese_schema(),
                system_content="You are a Japanese language expert. Provide accurate synonyms using appropriate scripts without furigana.",
                filename=f"japanese_step1_{word}.json"
            )
            
            return response["japanese_synonyms"][0]  # Return first (and only) result
        except Exception as e:
            print(f"Error in fetch_japanese_synonym_step1 for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    def fetch_japanese_synonym_step2(self, word, japanese_text):
        """Step 2: Add furigana to kanji/katakana in Japanese text"""
        try:
            prompt = (
                f"For the Japanese text '{japanese_text}' (synonym of English word '{word}'), "
                "break it down into character groups and provide furigana pairs. "
                "Group consecutive kanji or katakana characters together. "
                "For each group that contains kanji or katakana, provide the hiragana reading. "
                "For hiragana characters, mark needs_furigana as false. "
                "The result should allow reconstruction of the text with proper furigana format like: 特徴（とくちょう）"
            )
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_japanese_furigana_pairs_schema(),
                system_content="You are an expert in Japanese writing systems and furigana annotation.",
                filename=f"japanese_step2_{word}.json"
            )
            
            return self.convert_japanese_furigana_pairs(response["furigana_pairs"][0])
        except Exception as e:
            print(f"Error in fetch_japanese_synonym_step2 for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    def convert_japanese_furigana_pairs(self, furigana_data):
        """Convert Japanese furigana pairs to standard format"""
        try:
            word = furigana_data["word"]
            character_pairs = furigana_data["character_pairs"]
            
            japanese_with_furigana = ""
            
            for pair in character_pairs:
                characters = pair["characters"]
                furigana = pair["furigana"]
                needs_furigana = pair["needs_furigana"]
                
                if needs_furigana and furigana.strip():
                    # Add furigana in parentheses
                    japanese_with_furigana += f"{characters}（{furigana}）"
                else:
                    # Just add the characters without furigana
                    japanese_with_furigana += characters
            
            return {
                "word": word,
                "japanese_synonym": japanese_with_furigana
            }
        except Exception as e:
            print(f"Error converting Japanese furigana pairs: {e}")
            traceback.print_exc()
            raise
    
    def fetch_japanese_synonym_complete(self, word, word_database=None):
        """Complete Japanese synonym processing for a single word (both steps)"""
        try:
            print(f"Processing Japanese synonym for: {word}")
            
            # Step 1: Get basic Japanese synonym
            japanese_basic = self.fetch_japanese_synonym_step1(word)
            japanese_text = japanese_basic["japanese_synonym"]
            
            # Step 2: Add furigana if needed
            completed_japanese = self.fetch_japanese_synonym_step2(word, japanese_text)
            
            # Save to database
            if word_database:
                word_database.update_word_details(completed_japanese)
            
            return completed_japanese
            
        except Exception as e:
            print(f"Error in Japanese synonym processing for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    # ===== ARABIC WITH TRANSLITERATION =====
    
    def fetch_arabic_with_transliteration(self, word, word_database=None):
        """Fetch Arabic synonym with transliteration for a single word"""
        try:
            print(f"Processing Arabic synonym for: {word}")
            
            prompt = (
                f"For this English word: {word}\n"
                "Provide:\n"
                "1. Arabic synonym in Arabic script\n"
                "2. Romanized transliteration (Latin script phonetic representation)\n"
                "Use standard Arabic transliteration conventions."
            )
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_arabic_transliteration_schema(),
                system_content="You are an Arabic language expert. Provide accurate translations and transliterations.",
                filename=f"arabic_{word}.json"
            )
            
            arabic_detail = response["arabic_words"][0]
            
            # Save to database
            if word_database:
                word_database.update_word_details({
                    "word": arabic_detail["word"],
                    "arabic_synonym": arabic_detail["arabic_synonym"],
                    "arabic_transliteration": arabic_detail["transliteration"]
                })
            
            return arabic_detail
        except Exception as e:
            print(f"Error in Arabic processing for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    # ===== FRENCH WITH PHONETICS =====
    
    def fetch_french_with_phonetics(self, word, word_database=None):
        """Fetch French synonym with phonetics for a single word"""
        try:
            print(f"Processing French synonym for: {word}")
            
            prompt = (
                f"For this English word: {word}\n"
                "Provide:\n"
                "1. French synonym\n"
                "2. IPA phonetic transcription for the French word\n"
                "Use standard French IPA notation."
            )
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_french_phonetic_schema(),
                system_content="You are a French language and phonetics expert. Provide accurate translations and IPA transcriptions.",
                filename=f"french_{word}.json"
            )
            
            french_detail = response["french_words"][0]
            
            # Save to database
            if word_database:
                word_database.update_word_details({
                    "word": french_detail["word"],
                    "french_synonym": french_detail["french_synonym"],
                    "french_phonetic": french_detail["french_phonetic"]
                })
            
            return french_detail
        except Exception as e:
            print(f"Error in French processing for word '{word}': {e}")
            traceback.print_exc()
            raise
    
    # ===== GENERAL TRANSLITERATION METHOD =====
    
    def transliterate_text(self, text, source_lang, target_script="latin"):
        """
        General transliteration method (currently used for Arabic)
        Can be extended for other languages in the future
        """
        try:
            if source_lang.lower() == "arabic" and target_script.lower() == "latin":
                return self._arabic_to_latin_transliteration(text)
            else:
                raise NotImplementedError(f"Transliteration from {source_lang} to {target_script} not implemented")
        except Exception as e:
            print(f"Error in transliterate_text: {e}")
            traceback.print_exc()
            raise
    
    def _arabic_to_latin_transliteration(self, arabic_text):
        """
        Arabic to Latin transliteration
        This is a simplified implementation - you might want to use libraries like 'transliterate'
        """
        try:
            # Simplified mapping - in practice, you'd use a proper transliteration library
            arabic_to_latin = {
                'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h',
                'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's',
                'ش': 'sh', 'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a',
                'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
                'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y'
            }
            
            result = ""
            for char in arabic_text:
                result += arabic_to_latin.get(char, char)
            
            return result
        except Exception as e:
            print(f"Error in Arabic to Latin transliteration: {e}")
            traceback.print_exc()
            return arabic_text  # Return original if conversion fails
    
    # ===== MAIN INTERFACE METHODS (KEEPING COMPATIBILITY) =====
    
    def fetch_words(self, num_words, word_database, include_existing=True):
        """Main method to fetch words - delegates to openai or local with fallback"""
        try:
            words = self.fetch_words_openai(num_words, word_database, include_existing)
            if words and len(words) > 0:
                return words
            else:
                print("OpenAI fetch returned no words, falling back to local words")
                return self.fetch_words_local(num_words, word_database)
        except Exception as e:
            print(f"Error in fetch_words_openai: {e}, falling back to local words")
            traceback.print_exc()
            return self.fetch_words_local(num_words, word_database)

    def fetch_words_openai(self, num_words, word_database, include_existing=True):
        """Fetch words using structured OpenAI outputs with enhanced strategies"""
        try:
            propensities = self.load_propensities()
            local_words = self.fetch_words_local(num_words, word_database)
            
            words_number_scale_factor = 2
            num_words_scaled = num_words * words_number_scale_factor
            
            # Try multiple strategies to get unique words
            strategies = [
                ("common_daily", "common, everyday, and frequently used"),
                ("intermediate", "intermediate level, commonly found in newspapers and magazines"),
                ("business_professional", "business, professional, and workplace"),
                ("academic_vocabulary", "academic, scholarly, or formal"),
                ("technical_terms", "technical, scientific, or specialized"),
                ("advanced_literary", "advanced, sophisticated, or literary")
            ]
            
            partial_words = []
            
            for strategy_name, word_type in strategies:
                print(f"Trying strategy: {strategy_name}")
                
                try:
                    if propensities and strategy_name == "common_daily":
                        criteria_list = "\n".join([f"{i+1}) {propensity}" for i, propensity in enumerate(propensities)])
                        user_message = (
                            f"Generate a list of {num_words_scaled} unique {word_type} words that meet one or more of the following criteria:\n"
                            f"{criteria_list}\n"
                            "Focus on words that are commonly used in daily conversation and basic writing."
                        )
                    else:
                        user_message = f"Generate a list of {num_words_scaled} unique {word_type} English words suitable for language learning."
                    
                    response = self.send_request_with_json_schema(
                        prompt=user_message,
                        json_schema=self.get_word_list_schema(),
                        system_content=f"You are a vocabulary expert creating {word_type} English words.",
                        filename=f"fetch_words_{strategy_name}_{num_words_scaled}.json"
                    )
                    
                    words_list = response["words"]
                    
                    if include_existing:
                        unique_words = words_list
                        print(f"Strategy '{strategy_name}': {len(words_list)} total words (including existing)")
                    else:
                        unique_words = [word for word in words_list if not word_database.word_exists(word)]
                        print(f"Strategy '{strategy_name}': {len(words_list)} total words, {len(unique_words)} unique")
                    
                    partial_words.extend(unique_words)
                    partial_words = list(set(partial_words))
                    
                    print(f"Collected so far: {len(partial_words)} unique words")
                    
                    if len(partial_words) >= num_words:
                        print(f"Found enough words ({len(partial_words)}) from strategy '{strategy_name}', stopping early")
                        return partial_words[:num_words_scaled]
                        
                except Exception as e:
                    print(f"Strategy '{strategy_name}' failed: {e}")
                    traceback.print_exc()
                    continue
            
            if len(partial_words) > 0:
                print(f"Using collected results: {len(partial_words)} words")
                return partial_words[:num_words_scaled] if len(partial_words) >= num_words else partial_words
                
            print("No OpenAI words found, using local dictionary words as fallback...")
            if local_words and len(local_words) > 0:
                return local_words[:num_words]
            else:
                print("No local words available either, returning empty list")
                return []
        except Exception as e:
            print(f"Error in fetch_words_openai: {e}")
            traceback.print_exc()
            raise

    def fetch_word_details(self, words, word_database=None, num_words_phonetic=10, include_existing=True):
        """Fetch comprehensive word details using the new two-step processes - ONE WORD AT A TIME"""
        try:
            # Separate new words from existing words
            new_words = []
            existing_words = []
            
            for word in words:
                if word_database and word_database.word_exists(word):
                    existing_words.append(word)
                else:
                    new_words.append(word)
            
            print(f"Processing: {len(new_words)} new words, {len(existing_words)} existing words")
            
            if new_words:
                self.save_unused_words(words, new_words)

                # Process each word individually
                all_word_details = []
                
                for word in new_words:
                    try:
                        print(f"\n=== Processing word: {word} ===")
                        word_detail = {"word": word}
                        
                        # Process English phonetics (two-step)
                        try:
                            english_detail = self.fetch_english_phonetics_complete(word, word_database)
                            word_detail.update(english_detail)
                        except Exception as e:
                            print(f"Failed to process English phonetics for '{word}': {e}")
                            traceback.print_exc()
                        
                        # Process Japanese synonyms (two-step)
                        try:
                            japanese_detail = self.fetch_japanese_synonym_complete(word, word_database)
                            word_detail.update(japanese_detail)
                        except Exception as e:
                            print(f"Failed to process Japanese synonym for '{word}': {e}")
                            traceback.print_exc()
                        
                        # Process Arabic with transliteration
                        try:
                            arabic_detail = self.fetch_arabic_with_transliteration(word, word_database)
                            word_detail["arabic_synonym"] = arabic_detail["arabic_synonym"]
                            word_detail["arabic_transliteration"] = arabic_detail["transliteration"]
                        except Exception as e:
                            print(f"Failed to process Arabic synonym for '{word}': {e}")
                            traceback.print_exc()
                        
                        # Process French with phonetics
                        try:
                            french_detail = self.fetch_french_with_phonetics(word, word_database)
                            word_detail["french_synonym"] = french_detail["french_synonym"]
                            word_detail["french_phonetic"] = french_detail["french_phonetic"]
                        except Exception as e:
                            print(f"Failed to process French synonym for '{word}': {e}")
                            traceback.print_exc()
                        
                        # Update database with combined details
                        if word_database:
                            word_database.update_word_details(word_detail)
                        
                        all_word_details.append(word_detail)
                        
                    except Exception as e:
                        print(f"Failed to process word '{word}' completely: {e}")
                        traceback.print_exc()
                        # Continue with next word instead of failing completely
                        continue
                
                # Get updated word details from database for all processed words
                processed_words = [detail["word"] for detail in all_word_details]
                all_processed_words = processed_words + existing_words
                
                if all_processed_words:
                    final_word_details = []
                    for word in all_processed_words:
                        try:
                            word_detail = word_database.find_word_details(word)
                            if word_detail:
                                final_word_details.append(word_detail)
                        except Exception as e:
                            print(f"Error fetching final details for word '{word}': {e}")
                            traceback.print_exc()

                    # Update examples if we have results
                    if final_word_details and len(final_word_details) > 0:
                        self.examples = final_word_details[0:2] if len(final_word_details) >= 2 else final_word_details
                        self.save_examples()
                    
                    return final_word_details
                else:
                    print("No words were successfully processed")
                    return []
                
            else:
                # If only existing words, just return their details from database
                if existing_words and word_database:
                    print("Only existing words, fetching from database...")
                    word_details = []
                    for word in existing_words:
                        try:
                            word_detail = word_database.find_word_details(word)
                            if word_detail:
                                word_details.append(word_detail)
                        except Exception as e:
                            print(f"Error fetching details for existing word '{word}': {e}")
                            traceback.print_exc()
                    return word_details
                else:
                    return []
                    
        except Exception as e:
            print(f"Error in fetch_word_details: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to fetch word details.")

    # ===== LEGACY COMPATIBILITY METHODS =====
    
    def recheck_syllable_and_phonetic(self, word_details, word_database=None, messages=""):
        """Legacy method - now uses the new two-step process"""
        try:
            results = []
            for detail in word_details:
                word = detail["word"]
                result = self.fetch_english_phonetics_complete(word, word_database)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in recheck_syllable_and_phonetic: {e}")
            traceback.print_exc()
            raise
    
    def recheck_japanese_synonym(self, word_details, word_database=None, messages=''):
        """Legacy method - now uses the new two-step process"""
        try:
            results = []
            for detail in word_details:
                word = detail["word"]
                result = self.fetch_japanese_synonym_complete(word, word_database)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in recheck_japanese_synonym: {e}")
            traceback.print_exc()
            raise
    
    def recheck_japanese_synonym_with_conditions(self, word_details, word_database=None):
        """Legacy method - now uses the new two-step process"""
        try:
            results = []
            for detail in word_details:
                word = detail["word"]
                result = self.fetch_japanese_synonym_complete(word, word_database)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in recheck_japanese_synonym_with_conditions: {e}")
            traceback.print_exc()
            raise

    def recheck_pure_kanji_synonym(self, word_details, word_database=None):
        """Legacy method - delegates to new implementation"""
        try:
            words = [detail["word"] for detail in word_details]
            return self.fetch_pure_kanji_synonyms(words, word_database)
        except Exception as e:
            print(f"Error in recheck_pure_kanji_synonym: {e}")
            traceback.print_exc()
            raise

    def recheck_arabic_synonym(self, word_details, word_database=None):
        """Legacy method - now uses Arabic with transliteration"""
        try:
            results = []
            for detail in word_details:
                word = detail["word"]
                result = self.fetch_arabic_with_transliteration(word, word_database)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in recheck_arabic_synonym: {e}")
            traceback.print_exc()
            raise

    def fetch_pure_kanji_synonyms(self, words, word_database=None, messages=None):
        """Fetch pure kanji synonyms - keeping existing implementation for now"""
        try:
            prompt = (
                "Provide pure Japanese kanji synonyms for these English words. "
                "Use as few Japanese characters as possible. "
                f"Words: {', '.join(words)}"
            )
            
            schema = {
                "type": "object",
                "properties": {
                    "kanji_synonyms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "kanji_synonym": {"type": "string"},
                                "chinese_synonym": {"type": "string"},
                                "simplified_chinese_synonym": {"type": "string"}
                            },
                            "required": ["word", "kanji_synonym", "chinese_synonym", "simplified_chinese_synonym"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["kanji_synonyms"],
                "additionalProperties": False
            }
            
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=schema,
                system_content="You are an expert in Japanese kanji and Chinese characters.",
                filename=f"kanji_synonyms_{'_'.join(words[:3])}.json"
            )
            
            synonyms = response["kanji_synonyms"]
            
            # Save to database
            for detail in synonyms:
                if word_database:
                    word_database.update_word_details(detail)
            
            return synonyms
        except Exception as e:
            print(f"Error in fetch_pure_kanji_synonyms: {e}")
            traceback.print_exc()
            raise

    def fetch_arabic_synonyms(self, words, word_database, messages=None):
        """Legacy method - now uses Arabic with transliteration"""
        try:
            results = []
            for word in words:
                result = self.fetch_arabic_with_transliteration(word, word_database)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in fetch_arabic_synonyms: {e}")
            traceback.print_exc()
            raise

    def fetch_french_synonyms(self, words, word_database=None, model_name=None):
        """Legacy method - now uses French with phonetics"""
        try:
            results = []
            for word in words:
                result = self.fetch_french_with_phonetics(word, word_database)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in fetch_french_synonyms: {e}")
            traceback.print_exc()
            raise


class PhoneticRechecker(OpenAIRequestJSONBase):
    """
    Enhanced PhoneticRechecker that uses structured OpenAI outputs
    """
    
    def __init__(self, max_retries=3, use_cache=True):
        super().__init__(use_cache, max_retries)
        self.processed_csv = "word_phonetics_processed.csv"
        self.log_folder = "logs-word-phonetics"
    
    def word_exists(self, word):
        """Check if word exists in processed CSV"""
        if not os.path.exists(self.processed_csv):
            return False
        try:
            with open(self.processed_csv, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['word'] == word:
                        return True
            return False
        except Exception as e:
            print(f"Error checking word existence: {e}")
            traceback.print_exc()
            return False
        
    def save_to_csv(self, data):
        """Save data to CSV"""
        try:
            filename = self.processed_csv
            file_exists = os.path.isfile(filename)
            with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['word', 'syllable_word', 'phonetic']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for item in data:
                    writer.writerow(item)
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            traceback.print_exc()
        
    def save_log(self, words, prompt, response_content):
        """Save log to file"""
        try:
            if isinstance(words, list):
                word_list = [word_dict['word'] if isinstance(word_dict, dict) else str(word_dict) for word_dict in words]
                word_str = ",".join(word_list)
            else:
                word_str = str(words)
            
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
            
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.log_folder, f"{word_str}-{timestamp}.json")
            
            data_to_save = {
                "prompt": prompt,
                "response": response_content
            }
            
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data_to_save, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log: {e}")
            traceback.print_exc()
        
    def sanitize_input_and_normalize_syllables(self, data):
        """Sanitize input data"""
        try:
            for item in data:
                item['word'] = item['word'].replace('.', '').replace('/', '')
                sanitized_syllables = []
                for syl in item['syllables']:
                    sanitized = {
                        'word_syllable': syl['word_syllable'].replace('.', '').replace('/', ''),
                        'phonetic_syllable': syl['phonetic_syllable'].replace('.', '').replace('/', '')
                    }
                    sanitized_syllables.append(sanitized)
                item['syllables'] = sanitized_syllables
                
                joined_syllables = ''.join([syl['word_syllable'] for syl in sanitized_syllables])
                if joined_syllables == item['word']:
                    item['match'] = True
                else:
                    # Handle repetition correction
                    corrected_syllables = []
                    for syl in sanitized_syllables:
                        if corrected_syllables:
                            if corrected_syllables[-1]['word_syllable'][-1] == syl['word_syllable'][0]:
                                corrected_syllable = syl['word_syllable'][1:]
                            else:
                                corrected_syllable = syl['word_syllable']
                            corrected_syllables.append({'word_syllable': corrected_syllable, 'phonetic_syllable': syl['phonetic_syllable']})
                        else:
                            corrected_syllables.append(syl)
                    
                    corrected_word = ''.join([syl['word_syllable'] for syl in corrected_syllables])
                    item['corrected_word'] = corrected_word
                    item['match'] = corrected_word == item['word']
            
            return data
        except Exception as e:
            print(f"Error sanitizing input: {e}")
            traceback.print_exc()
            raise
        
    def convert_to_dot_separated_json(self, data):
        """Convert to dot-separated format"""
        try:
            result = []
            for item in data:
                syllable_word = '·'.join([syl['word_syllable'] for syl in item['syllables']]).replace("· ·", " ")
                phonetic = '·'.join([syl['phonetic_syllable'] for syl in item['syllables']]).replace("· ·", " ")
                
                result.append({
                    'word': item['word'],
                    'syllable_word': syllable_word,
                    'phonetic': phonetic
                })
            return result
        except Exception as e:
            print(f"Error converting to dot-separated format: {e}")
            traceback.print_exc()
            raise

    def get_phonetic_pairs_schema(self):
        """Schema for phonetic pairs responses"""
        return {
            "type": "object",
            "properties": {
                "phonetic_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "phonetics": {"type": "string"},
                            "syllables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "word_syllable": {"type": "string"},
                                        "phonetic_syllable": {"type": "string"}
                                    },
                                    "required": ["word_syllable", "phonetic_syllable"],
                                    "additionalProperties": False
                                },
                                "minItems": 1
                            }
                        },
                        "required": ["word", "phonetics", "syllables"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["phonetic_data"],
            "additionalProperties": False
        }

    def recheck_word_phonetics_with_paired_tuple(self, words, word_database, force=False):
        """Recheck word phonetics using structured OpenAI outputs"""
        try:
            words_filtered = []
            for word in words:
                if self.word_exists(word) and not force:
                    print(f"Skipping '{word}' as it exists in {self.processed_csv}.")
                    continue
                else:
                    words_filtered.append(word)

            if len(words_filtered) == 0:
                return []

            prompt = (
                "Provide the correct and standard phonetics of the word, "
                "and the syllable of this word with its phonetics in pair. "
                "Make sure the join of syllables will be exact the original word and phonetics. "
                "Treat space as a syllable. "
                f"The words: {words_filtered} "
            )

            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_phonetic_pairs_schema(),
                system_content="You are an expert linguist specializing in phonetic transcription and syllable division.",
                filename=f"phonetic_pairs_{'_'.join(words_filtered[:3])}.json"
            )
            
            word_phonetics = response["phonetic_data"]
            print("Phonetic data received:")
            pprint(word_phonetics)

            # Process the response through the provided functions
            data_sanitized = self.sanitize_input_and_normalize_syllables(word_phonetics)
            converted_data = self.convert_to_dot_separated_json(data_sanitized)

            self.save_to_csv(converted_data)
            self.save_log(word_phonetics, prompt, str(response))

            print("OpenAI: ")
            pprint(converted_data)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            # Save word details to database
            for detail in converted_data:
                if word_database:
                    print(f"Updating syllable and phonetic of word with", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                    word_database.update_word_details(detail)
                    
            return converted_data
            
        except Exception as e:
            print(f"Error in recheck_word_phonetics_with_paired_tuple: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to recheck phonetics.")