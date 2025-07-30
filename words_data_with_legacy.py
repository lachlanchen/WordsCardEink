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
from typing import List, Dict, Any
from pprint import pprint

# Import specific classes and functions from the legacy module

from words_data_legacy import (
    # WordsDatabase, 
    # OpenAiChooser, 
    # EmojiWordChooser,
    AdvancedWordFetcher as LegacyAdvancedWordFetcher,
    PhoneticRechecker as LegacyPhoneticRechecker,
    # split_word, 
    # split_word_with_color, 
    # count_syllables,
    # clean_word_details,
    # clean_and_transcribe,
    # JSONParsingError,
    # NotEnoughUniqueWordsError
)

from words_data_utils import (
    # WordsDatabase, 
    OpenAiChooser, 
    EmojiWordChooser,
    # AdvancedWordFetcher as LegacyAdvancedWordFetcher,
    # PhoneticRechecker as LegacyPhoneticRechecker,
    split_word, 
    split_word_with_color, 
    count_syllables,
    clean_word_details,
    clean_and_transcribe,
    JSONParsingError,
    NotEnoughUniqueWordsError
)

from words_database import (
    WordsDatabase, 
    # OpenAiChooser, 
    # EmojiWordChooser,
    # AdvancedWordFetcher as LegacyAdvancedWordFetcher,
    # PhoneticRechecker as LegacyPhoneticRechecker,
    # split_word, 
    # split_word_with_color, 
    # count_syllables,
    # clean_word_details,
    # clean_and_transcribe,
    # JSONParsingError,
    # NotEnoughUniqueWordsError
)

from openai_request_json import OpenAIRequestJSONBase


class AdvancedWordFetcher(OpenAIRequestJSONBase):
    """
    Enhanced AdvancedWordFetcher that uses structured OpenAI outputs
    """
    
    def __init__(self, max_retries=3, use_cache=True):
        super().__init__(use_cache, max_retries)
        self.legacy_fetcher = LegacyAdvancedWordFetcher(max_retries)
        # Copy attributes from legacy fetcher
        self.examples = self.legacy_fetcher.examples
        self.model_name = self.legacy_fetcher.model_name
        
    def load_examples(self):
        return self.legacy_fetcher.load_examples()
        
    def save_examples(self):
        return self.legacy_fetcher.save_examples()
        
    def load_propensities(self):
        return self.legacy_fetcher.load_propensities()
        
    def fetch_words_local(self, num_words, word_database):
        return self.legacy_fetcher.fetch_words_local(num_words, word_database)
        
    def save_unused_words(self, words, random_words, file_path='data/unused_words.csv'):
        return self.legacy_fetcher.save_unused_words(words, random_words, file_path)
        
    def map_syllables_phonetics(self, syllables, phonetics):
        return self.legacy_fetcher.map_syllables_phonetics(syllables, phonetics)
    
    def get_word_list_schema(self):
        """Schema for word list responses"""
        return {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 1
                }
            },
            "required": ["words"],
            "additionalProperties": False
        }
    
    def get_word_details_schema(self):
        """Schema for detailed word information - simplified to essential fields"""
        return {
            "type": "object",
            "properties": {
                "word_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "syllable_word": {"type": "string"},
                            "phonetic": {"type": "string"},
                            "japanese_synonym": {"type": "string"}
                        },
                        "required": ["word", "syllable_word", "phonetic", "japanese_synonym"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["word_details"],
            "additionalProperties": False
        }
    
    def get_syllable_phonetic_schema(self):
        """Schema for syllable and phonetic information"""
        return {
            "type": "object",
            "properties": {
                "word_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "syllable_word": {"type": "string"},
                            "phonetic": {"type": "string"}
                        },
                        "required": ["word", "syllable_word", "phonetic"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["word_details"],
            "additionalProperties": False
        }
    
    def get_japanese_synonym_schema(self):
        """Schema for Japanese synonym information"""
        return {
            "type": "object",
            "properties": {
                "word_details": {
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
            "required": ["word_details"],
            "additionalProperties": False
        }
    
    def get_kanji_chinese_schema(self):
        """Schema for kanji and Chinese synonyms"""
        return {
            "type": "object",
            "properties": {
                "word_details": {
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
                    },
                    "minItems": 1
                }
            },
            "required": ["word_details"],
            "additionalProperties": False
        }
    
    def get_arabic_synonym_schema(self):
        """Schema for Arabic synonym information"""
        return {
            "type": "object",
            "properties": {
                "word_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "arabic_synonym": {"type": "string"}
                        },
                        "required": ["word", "arabic_synonym"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["word_details"],
            "additionalProperties": False
        }
    
    def get_french_synonym_schema(self):
        """Schema for French synonym information"""
        return {
            "type": "object",
            "properties": {
                "word_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "french_synonym": {"type": "string"}
                        },
                        "required": ["word", "french_synonym"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["word_details"],
            "additionalProperties": False
        }

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
            return self.fetch_words_local(num_words, word_database)

    def fetch_words_openai(self, num_words, word_database, include_existing=True):
        """
        Fetch words using structured OpenAI outputs with enhanced strategies
        """
        propensities = self.load_propensities()
        local_words = self.fetch_words_local(num_words, word_database)
        
        words_number_scale_factor = 2
        num_words_scaled = num_words * words_number_scale_factor
        
        # Try multiple strategies to get unique words, starting with common words
        strategies = [
            ("common_daily", "common, everyday, and frequently used"),
            ("intermediate", "intermediate level, commonly found in newspapers and magazines"),
            ("business_professional", "business, professional, and workplace"),
            ("academic_vocabulary", "academic, scholarly, or formal"),
            ("technical_terms", "technical, scientific, or specialized"),
            ("advanced_literary", "advanced, sophisticated, or literary")
        ]
        
        partial_words = []  # Collect partial results
        
        for strategy_name, word_type in strategies:
            print(f"Trying strategy: {strategy_name}")
            
            # Choose the appropriate prompt based on strategy
            if propensities and strategy_name == "common_daily":
                criteria_list = "\n".join([f"{i+1}) {propensity}" for i, propensity in enumerate(propensities)])
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} words that meet one or more of the following criteria:\n"
                    f"{criteria_list}\n"
                    "Focus on words that are commonly used in daily conversation and basic writing. "
                    "Return the words in a JSON object with a 'words' array containing the list of words."
                )
            elif strategy_name == "common_daily":
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} English words. "
                    "Focus on vocabulary that people use in daily conversations, basic reading, and everyday situations. "
                    "Include words from: daily activities, emotions, common objects, basic verbs and adjectives, family and relationships, food, weather, etc. "
                    "These should be words that intermediate English learners would encounter regularly. "
                    f"Return the words in a JSON object with a 'words' array containing the list of words."
                )
            elif strategy_name == "intermediate":
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} English words. "
                    "Focus on vocabulary commonly found in newspapers, magazines, and general media. "
                    "Include words from: current events, general science, basic business, travel, culture, health, education, etc. "
                    "These should be words that upper-intermediate English learners encounter in mainstream media. "
                    f"Return the words in a JSON object with a 'words' array containing the list of words."
                )
            elif strategy_name == "business_professional":
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} English words. "
                    "Focus on vocabulary used in professional and business contexts. "
                    "Include words from: meetings, presentations, negotiations, management, finance, marketing, project management, etc. "
                    "These should be words that professionals encounter in workplace communication. "
                    f"Return the words in a JSON object with a 'words' array containing the list of words."
                )
            elif strategy_name == "academic_vocabulary":
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} English words. "
                    "Focus on vocabulary found in academic texts, research papers, and scholarly writing. "
                    "Include words from: research methodology, analysis, theories, formal writing, academic discussions, etc. "
                    "These should be words that university students and researchers encounter in academic contexts. "
                    f"Return the words in a JSON object with a 'words' array containing the list of words."
                )
            elif strategy_name == "technical_terms":
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} English words. "
                    "Focus on specialized vocabulary from technical and scientific fields. "
                    "Include words from: science, technology, medicine, engineering, computer science, biology, physics, etc. "
                    "These should be technical terms that specialists in various fields would use. "
                    f"Return the words in a JSON object with a 'words' array containing the list of words."
                )
            else:  # advanced_literary
                user_message = (
                    f"Generate a list of {num_words_scaled} unique {word_type} English words. "
                    "Focus on sophisticated vocabulary found in literature, advanced texts, and formal writing. "
                    "Include words from: classical literature, philosophy, advanced humanities, rare but meaningful words, etc. "
                    "These should be words that challenge even advanced English speakers and readers. "
                    f"Return the words in a JSON object with a 'words' array containing the list of words."
                )

            if strategy_name in ["common_daily", "intermediate"]:
                system_content = (
                    "You are an assistant with a vast vocabulary knowledge. "
                    "You specialize in practical, useful English vocabulary. "
                    f"Focus on {word_type} words that are genuinely useful for English learners."
                )
            elif strategy_name in ["business_professional", "academic_vocabulary"]:
                system_content = (
                    "You are an assistant with extensive knowledge of professional and academic English. "
                    "You specialize in formal and professional vocabulary. "
                    f"Focus on {word_type} words that are essential for professional and academic success."
                )
            else:  # technical_terms, advanced_literary
                system_content = (
                    "You are an assistant with a vast vocabulary and creativity. "
                    "You specialize in advanced and sophisticated English vocabulary. "
                    f"Focus on {word_type} words that would challenge an advanced English learner."
                )
            
            try:
                # Use different cache filenames for different strategies
                cache_filename = f"fetch_words_{strategy_name}_{num_words_scaled}.json"
                
                # Use structured outputs to get the response
                response = self.send_request_with_json_schema(
                    prompt=user_message,
                    json_schema=self.get_word_list_schema(),
                    system_content=system_content,
                    filename=cache_filename
                )
                
                words_list = response["words"]
                
                # Filter out words that already exist in the database (unless include_existing=True)
                if include_existing:
                    unique_words = words_list  # Include all words, even existing ones
                    print(f"Strategy '{strategy_name}': {len(words_list)} total words (including existing)")
                else:
                    unique_words = [word for word in words_list if not word_database.word_exists(word)]
                    print(f"Strategy '{strategy_name}': {len(words_list)} total words, {len(unique_words)} unique")
                
                # Add unique words to our collection
                partial_words.extend(unique_words)
                partial_words = list(set(partial_words))  # Remove duplicates
                
                print(f"Collected so far: {len(partial_words)} unique words")
                
                # If we got enough unique words, return them early
                if len(partial_words) >= num_words:
                    print(f"Found enough words ({len(partial_words)}) from strategy '{strategy_name}', stopping early")
                    return partial_words[:num_words_scaled]  # Return up to the requested amount
                    
            except Exception as e:
                print(f"Strategy '{strategy_name}' failed: {e}")
                continue
        
        # If all strategies failed to get enough words, try a fallback with random generation
        print("All strategies provided insufficient words, trying fallback with creative approach...")
        try:
            fallback_message = (
                f"Create {num_words_scaled} unique, creative, and sophisticated English words. "
                "Mix real advanced vocabulary with technical terms from various fields: "
                "neuroscience, quantum physics, philosophy, linguistics, biotechnology, "
                "cybersecurity, environmental science, psychology, literature, etc. "
                "Focus on words that would appear in graduate-level academic papers or professional journals. "
                "Return the words in a JSON object with a 'words' array."
            )
            
            response = self.send_request_with_json_schema(
                prompt=fallback_message,
                json_schema=self.get_word_list_schema(),
                system_content="You are a vocabulary expert creating challenging advanced English words.",
                filename=f"fetch_words_fallback_{num_words_scaled}.json"
            )
            
            words_list = response["words"]
            unique_words = words_list if include_existing else [word for word in words_list if not word_database.word_exists(word)]
            
            print(f"Fallback strategy: {len(words_list)} total words, {len(unique_words)} {'(including existing)' if include_existing else 'unique'}")
            
            if len(unique_words) > 0:
                partial_words.extend(unique_words)
                partial_words = list(set(partial_words))  # Remove duplicates
                
        except Exception as e:
            print(f"Fallback strategy failed: {e}")
        
        # Use what we have collected
        if len(partial_words) > 0:
            print(f"Using collected results: {len(partial_words)} words")
            return partial_words[:num_words_scaled] if len(partial_words) >= num_words else partial_words
            
        # Final fallback: use local dictionary words
        print("No OpenAI words found, using local dictionary words as fallback...")
        if local_words and len(local_words) > 0:
            if include_existing:
                print(f"Found {len(local_words)} words from local dictionary (including existing)")
                return local_words[:num_words]
            else:
                local_unique = [word for word in local_words if not word_database.word_exists(word)]
                if local_unique:
                    print(f"Found {len(local_unique)} unique words from local dictionary")
                    return local_unique[:num_words]
                else:
                    print("Even local words are mostly duplicates, returning available local words anyway")
                    return local_words[:num_words]
        else:
            print("No local words available either, returning empty list")
            return []

    def fetch_word_details(self, words, word_database=None, num_words_phonetic=10, include_existing=True):
        """
        Fetch word details using structured OpenAI outputs
        """
        # Initialize phonetic_checker if it doesn't exist
        if not hasattr(self, 'phonetic_checker'):
            self.phonetic_checker = PhoneticRechecker()
        
        # Separate new words from existing words
        new_words = []
        existing_words = []
        
        for word in words:
            if word_database and word_database.word_exists(word):
                existing_words.append(word)
            else:
                new_words.append(word)
        
        print(f"Processing: {len(new_words)} new words, {len(existing_words)} existing words")
        
        # Only process new words with OpenAI
        if new_words:
            random_words = new_words
            self.save_unused_words(words, random_words)

            # Directly create the example string
            example_word = self.examples[0].get("word", "")
            syllables = split_word(self.examples[0].get("syllable_word", ""))
            phonetics = split_word(self.examples[0].get("phonetic", ""))
            mappings = self.map_syllables_phonetics(syllables, phonetics)

            words_string = ', '.join(random_words).lower()
            
            detailed_list_message = (
                "Could you provide a detailed syllable (using ·) and phonetic separation (also using ·) "
                "ensuring a one-to-one correspondence between syllable_word and its IPA phonetic? "
                "Please adjust the syllable or phonetic divisions if necessary "
                "to ensure each syllable directly matches/aligns with its corresponding phonetic element, "
                "even if this means altering the conventional syllable breakdown."
                f"For example, the separation of {example_word} should reflect the correspondence: \n {mappings}. \n"
                "For japanese_synonym: "
                "- If kanji exists for the concept, use kanji with hiragana reading in parentheses: 特徴（とくちょう）, 定義（ていぎ） "
                "- If no kanji exists or hiragana is more natural, use hiragana only: つかむ, する "
                "- Prefer kanji when commonly used in written Japanese "
                f"Could you provide me the linguistic details for words [ {words_string} ] ?"
                "Return the data in a JSON object with a 'word_details' array containing: word, syllable_word, phonetic, and japanese_synonym for each word."
            )

            system_content = "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words. For Japanese synonyms, always use kanji characters with hiragana readings in parentheses when possible (e.g., 特徴（とくちょう）, 定義（ていぎ）). Avoid providing purely hiragana words unless no kanji equivalent exists."
            
            try:
                # Use structured outputs to get the response
                response = self.send_request_with_json_schema(
                    prompt=detailed_list_message,
                    json_schema=self.get_word_details_schema(),
                    system_content=system_content,
                    filename=f"word_details_{'_'.join(random_words[:3])}.json"
                )
                
                word_phonetics = response["word_details"]

                # Save word details to database
                for detail in word_phonetics:
                    if word_database:
                        try:
                            word_database.insert_word_details(detail)
                        except Exception as e:
                            if "UNIQUE constraint failed" in str(e):
                                print(f"Word '{detail.get('word', 'unknown')}' already exists in database, updating instead...")
                                word_database.update_word_details(detail)
                            else:
                                print(f"Error inserting word details: {e}")
                                continue
                
                # Post-processing steps remain the same
                if word_database:
                    words_list = [word["word"] for word in word_phonetics]
                    
                    print("Starting comparing separation...")
                    self.phonetic_checker.recheck_word_phonetics_with_paired_tuple(words_list, word_database)
                    print("Starting check Japanese...")
                    self.recheck_japanese_synonym_with_conditions(word_phonetics.copy(), word_database)
                    print("Generating kanji...")
                    word_database.update_kanji_for_all_words()
                    print("Starting check pure kanji...")
                    self.recheck_pure_kanji_synonym(word_phonetics.copy(), word_database)
                    print("Starting check Arabic...")
                    word_database.convert_and_update_chinese_synonyms()
                    self.recheck_arabic_synonym(word_phonetics.copy(), word_database)

                    # Get updated word details from database for all words (new + existing)
                    all_processed_words = words_list + existing_words
                    word_phonetics = [word_database.find_word_details(word) for word in all_processed_words]
                    word_phonetics = [w for w in word_phonetics if w is not None]  # Filter out None results

                    self.examples = word_phonetics[0:2] if len(word_phonetics) >= 2 else word_phonetics
                    self.save_examples()
                    
                return word_phonetics
                
            except Exception as e:
                print(f"Error in fetch_word_details: {e}")
                traceback.print_exc()
                raise RuntimeError("Failed to fetch word details.")
        
        else:
            # If only existing words, just return their details from database
            if existing_words and word_database:
                print("Only existing words, fetching from database...")
                word_phonetics = [word_database.find_word_details(word) for word in existing_words]
                word_phonetics = [w for w in word_phonetics if w is not None]  # Filter out None results
                return word_phonetics
            else:
                return []

    def recheck_syllable_and_phonetic(self, word_details, word_database=None, messages=""):
        """
        Recheck syllable and phonetic information using structured outputs
        """
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Rechecking syllable and phonetics: ")
        pprint([word["word"] for word in word_details])
        print("\n")

        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'syllable_word', 'phonetic']} for word in word_details]
        word_details = clean_word_details(word_details)

        detailed_list_message = (
            "That ˈ and ˌ are always treated as separator regardless if '·' exists or not. "
            "Please ensure the syllable separation in the 'phonetic' transcription aligns with the 'syllable_word' separation. "
            "Each syllable should be marked with a central dot (·) in both fields. "
            "Adjust the 'phonetic' field or the syllable divisions in 'syllable_word' to be matched. "
            "Return the corrected data in a JSON object with a 'word_details' array."
        )

        system_content = (
            "You are an assistant skilled in linguistics, "
            "capable of providing accurate and detailed phonetic and linguistic attributes for given words. "
            "You are excellent in separate words and their phonetics into consistent and accurate separations with '·'."
        )

        try:
            # Use structured outputs to get the response
            response = self.send_request_with_json_schema(
                prompt=detailed_list_message,
                json_schema=self.get_syllable_phonetic_schema(),
                system_content=system_content,
                filename=f"syllable_phonetic_{'_'.join([w['word'] for w in word_details[:3]])}.json"
            )
            
            word_phonetics = response["word_details"]
            word_phonetics = clean_word_details(word_phonetics)

            print("OpenAI: ")
            pprint(word_phonetics)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            # Save word details to database
            for detail in word_phonetics:
                if word_database:
                    print(f"Updating syllable and phonetic of word with", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                    print("\n")
                    word_database.update_word_details(detail)
                    
            return word_phonetics
            
        except Exception as e:
            print(f"Error in recheck_syllable_and_phonetic: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to recheck syllable and phonetic.")

    def recheck_japanese_synonym(self, word_details, word_database=None, messages=''):
        """
        Recheck Japanese synonym using structured outputs
        """
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Rechecking Japanese synonym: ")
        pprint([word["word"] for word in word_details])
        print("\n")

        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'japanese_synonym']} for word in word_details]
        word_details = clean_and_transcribe(word_details)

        detailed_list_message = (
            "Could you correct the furigana/hiragana of kanji/katakana inside the parentheses as needed? "
            "Return the corrected data in a JSON object with a 'word_details' array."
        )

        system_content = (
            "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words. "
            "You are excellent in providing hiragana (furigana) for consecutive kanji/katakana."
        )

        try:
            # Use structured outputs to get the response
            response = self.send_request_with_json_schema(
                prompt=detailed_list_message,
                json_schema=self.get_japanese_synonym_schema(),
                system_content=system_content,
                filename=f"japanese_synonym_{'_'.join([w['word'] for w in word_details[:3]])}.json"
            )
            
            word_phonetics = response["word_details"]
            word_phonetics = clean_word_details(word_phonetics)

            print("OpenAI: ")
            print(word_phonetics)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            # Save word details to database
            for detail in word_phonetics:
                if word_database:
                    print("Updating japanese synonym of word with ", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                    print("\n")
                    word_database.update_word_details(detail)

            return word_phonetics
            
        except Exception as e:
            print(f"Error in recheck_japanese_synonym: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to recheck Japanese synonym.")

    def recheck_japanese_synonym_with_conditions(self, word_details, word_database=None):
        """
        Use legacy method for complex Japanese synonym checking with conditions
        """
        return self.legacy_fetcher.recheck_japanese_synonym_with_conditions(word_details, word_database)

    def recheck_pure_kanji_synonym(self, word_details, word_database=None):
        """
        Use legacy method for kanji synonym checking
        """
        return self.legacy_fetcher.recheck_pure_kanji_synonym(word_details, word_database)

    def recheck_arabic_synonym(self, word_details, word_database=None):
        """
        Use legacy method for Arabic synonym checking
        """
        return self.legacy_fetcher.recheck_arabic_synonym(word_details, word_database)

    def fetch_pure_kanji_synonyms(self, words, word_database=None, messages=None):
        """
        Fetch pure kanji synonyms using structured outputs
        """
        detailed_list_message = (
            "Based on the following words, provide the pure Japanese kanji synonym with as least Japanese letters as possible for each word in the list. "
            "If it's hard just give some loosely related or use traditional Chinese as kanji_synonym field. "
            "Each dictionary contains 'word', 'kanji_synonym', (traditional) 'chinese_synonym' and 'simplified_chinese_synonym'. "
            "Return the data in a JSON object with a 'word_details' array."
            f"The words to process are: {', '.join(words)}."
        )

        system_content = (
            "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. "
            "You are excellent in providing pure kanji synonyms."
        )

        try:
            # Use structured outputs to get the response
            response = self.send_request_with_json_schema(
                prompt=detailed_list_message,
                json_schema=self.get_kanji_chinese_schema(),
                system_content=system_content,
                filename=f"kanji_synonyms_{'_'.join(words[:3])}.json"
            )
            
            synonyms = response["word_details"]

            # Save synonyms to database
            for detail in synonyms:
                if word_database:
                    word_database.update_word_details(detail)

            return synonyms
            
        except Exception as e:
            print(f"Error in fetch_pure_kanji_synonyms: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to fetch pure kanji synonyms.")

    def fetch_arabic_synonyms(self, words, word_database, messages=None):
        """
        Fetch Arabic synonyms using structured outputs
        """
        print("Words to fetch arabic: ", words)
        
        detailed_list_message = (
            "Based on the following words, provide the Arabic synonym for each word in the list. "
            "Each dictionary contains 'word' and 'arabic_synonym'. "
            "Return the data in a JSON object with a 'word_details' array."
            f"The words to process are: {', '.join(words)}."
        )

        system_content = (
            "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. "
            "You are excellent in providing Arabic synonyms."
        )

        try:
            # Use structured outputs to get the response
            response = self.send_request_with_json_schema(
                prompt=detailed_list_message,
                json_schema=self.get_arabic_synonym_schema(),
                system_content=system_content,
                filename=f"arabic_synonyms_{'_'.join(words[:3])}.json"
            )
            
            synonyms = response["word_details"]

            # Save synonyms to database
            for detail in synonyms:
                if word_database:
                    word_database.update_word_details(detail)

            return synonyms
            
        except Exception as e:
            print(f"Error in fetch_arabic_synonyms: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to fetch Arabic synonyms.")

    def fetch_french_synonyms(self, words, word_database=None, model_name=None):
        """
        Fetch French synonyms using structured outputs
        """
        detailed_list_message = (
            "Based on the following words, provide the French synonym for each word in the list. "
            "Return the data in a JSON object with a 'word_details' array, "
            "where each dictionary contains 'word' and 'french_synonym'. "
            f"The words to process are: {', '.join(words)}."
        )

        system_content = (
            "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. "
            "You are excellent in providing French synonyms."
        )

        try:
            # Use structured outputs to get the response
            response = self.send_request_with_json_schema(
                prompt=detailed_list_message,
                json_schema=self.get_french_synonym_schema(),
                system_content=system_content,
                filename=f"french_synonyms_{'_'.join(words[:3])}.json"
            )
            
            synonyms = response["word_details"]

            # Save synonyms to database
            for detail in synonyms:
                if word_database:
                    word_database.update_word_details(detail)

            return synonyms
            
        except Exception as e:
            print(f"Error in fetch_french_synonyms: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to fetch French synonyms.")


class PhoneticRechecker(OpenAIRequestJSONBase):
    """
    Enhanced PhoneticRechecker that uses structured OpenAI outputs
    """
    
    def __init__(self, max_retries=3, use_cache=True):
        super().__init__(use_cache, max_retries)
        self.legacy_checker = LegacyPhoneticRechecker(max_retries)
        # Copy attributes from legacy checker
        self.processed_csv = self.legacy_checker.processed_csv
        self.log_folder = self.legacy_checker.log_folder
    
    def word_exists(self, word):
        return self.legacy_checker.word_exists(word)
        
    def save_to_csv(self, data):
        return self.legacy_checker.save_to_csv(data)
        
    def save_log(self, words, prompt, response_content):
        return self.legacy_checker.save_log(words, prompt, response_content)
        
    def sanitize_input_and_normalize_syllables(self, data):
        return self.legacy_checker.sanitize_input_and_normalize_syllables(data)
        
    def convert_to_dot_separated_json(self, data):
        return self.legacy_checker.convert_to_dot_separated_json(data)
    
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
        """
        Recheck word phonetics using structured OpenAI outputs
        """
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
            "Return the data in a JSON object with a 'phonetic_data' array."
        )

        system_content = "You are an expert linguist specializing in phonetic transcription and syllable division."

        try:
            # Use structured outputs to get the response
            response = self.send_request_with_json_schema(
                prompt=prompt,
                json_schema=self.get_phonetic_pairs_schema(),
                system_content=system_content,
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