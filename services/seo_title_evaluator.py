import re
import logging
from typing import Optional, Dict, List
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache, cached_property


class SEOTitleEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', weights: Optional[Dict[str, float]] = None):
        self.model = SentenceTransformer(model_name)
        self.weights = weights or {
            "keyword_start": 1.5,
            "keyword_contains": 2.5,
            "length_optimal": 2.5,
            "capitalized": 0.8,
            "punctuation": 0.8,
            "voice_search": 1.2,
            "repeated_keyword_penalty": -0.6,
            "unique_word_ratio_good": 0.7,
            "unique_word_ratio_bad": -0.3,
            "semantic": 3.0,
            "clickbait_penalty": -1.0,
            "gibberish_penalty": -2.0,
            "weak_grammar_penalty": -1.0,
        }

    @cached_property
    def hazm_tools(self):
        try:
            from hazm import Normalizer, word_tokenize, Lemmatizer
            return {
                "available": True,
                "normalizer": Normalizer(),
                "tokenize": word_tokenize,
                "lemmatizer": Lemmatizer()
            }
        except ImportError:
            return {"available": False}

    @cached_property
    def spacy_en(self):
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except:
            return None

    def update_weights_from_feedback(self, feedback: Dict[str, float]):
        for k, v in feedback.items():
            if k in self.weights:
                self.weights[k] = (self.weights[k] + v) / 2

    def _length_score(self, length: int) -> float:
        if length < 15:
            return -0.7
        if 15 <= length <= 60:
            return self.weights["length_optimal"] * (1 - abs(37 - length) / 37)
        return -1.0

    def _lemmatize(self, text: str) -> List[str]:
        text = text.lower()
        if self.hazm_tools["available"]:
            text = self.hazm_tools["normalizer"].normalize(text)
            tokens = self.hazm_tools["tokenize"](text)
            return [self.hazm_tools["lemmatizer"].lemmatize(t) for t in tokens if t.isalpha()]
        elif self.spacy_en:
            doc = self.spacy_en(text)
            return [token.lemma_ for token in doc if token.is_alpha]
        else:
            return re.findall(r'\b\w+\b', text)

    def _count_keyword_occurrences(self, text: str, keyword: str) -> int:
        words = self._lemmatize(text)
        key_parts = self._lemmatize(keyword)
        return sum(1 for kw in key_parts if kw in words)

    def _repeated_keyword_penalty(self, count: int, total_words: int) -> float:
        ratio = count / max(total_words, 1)
        if ratio > 0.3:
            return self.weights["repeated_keyword_penalty"] * (ratio * 3)
        return 0.0

    def _is_clickbait(self, title: str) -> float:
        patterns = [r"\b(you|won't|never|must|shocking|secret|revealed)\b",
                    r"همه چیز درباره", r"چیست\??", r"راهنمای کامل", r"بهترین .* چیست"]
        return self.weights["clickbait_penalty"] if any(re.search(p, title.lower()) for p in patterns) else 0.0

    def _is_gibberish(self, title: str) -> float:
        words = title.split()
        bad_words = [w for w in words if len(w) > 20 or not re.search(r'[a-zA-Z\u0600-\u06FF0-9]', w)]
        if len(bad_words) / max(len(words), 1) > 0.3:
            return self.weights["gibberish_penalty"]
        return 0.0

    def _grammar_weakness_penalty(self, title: str) -> float:
        if self.spacy_en:
            doc = self.spacy_en(title)
            roots = [t for t in doc if t.dep_ in ("ROOT", "nsubj")]
            if not roots:
                return self.weights["weak_grammar_penalty"]
        return 0.0

    @lru_cache(maxsize=512)
    def _encode(self, text: str):
        return self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

    def evaluate_title(self, title: str, focus_keyword: str) -> float:
        score = 0.0
        title_lower = title.lower()
        focus_lower = focus_keyword.lower()

        # Keyword placement
        if title_lower.startswith(focus_lower):
            score += self.weights["keyword_start"]
        elif focus_lower in title_lower:
            score += self.weights["keyword_contains"] * (1 - title_lower.find(focus_lower) / len(title_lower))

        # Length
        score += self._length_score(len(title.strip()))

        # Formatting cues
        if title and title[0].isupper():
            score += self.weights["capitalized"]
        if any(p in title for p in [":", "-", "?"]):
            score += self.weights["punctuation"]

        # Voice search friendly
        voice_keywords = ["how to", "what is", "why is", "where can", "who is", "چگونه", "چیست", "چرا", "کجا"]
        if any(p in title_lower for p in voice_keywords):
            score += self.weights["voice_search"]

        # Keyword repetition
        lemmatized_words = self._lemmatize(title)
        kcount = self._count_keyword_occurrences(title, focus_keyword)
        score += self._repeated_keyword_penalty(kcount, len(lemmatized_words))

        # Unique word ratio
        unique_ratio = len(set(lemmatized_words)) / max(len(lemmatized_words), 1)
        score += self.weights["unique_word_ratio_good"] * unique_ratio if unique_ratio > 0.5 else self.weights["unique_word_ratio_bad"] * (1 - unique_ratio)

        # Semantic similarity
        try:
            score += util.cos_sim(self._encode(title), self._encode(focus_keyword)).item() * self.weights["semantic"]
        except Exception as e:
            logging.warning(f"Semantic similarity failed: {e}")
            score += 0.5 * self.weights["semantic"]

        # Penalties
        score += self._is_clickbait(title)
        score += self._is_gibberish(title)
        score += self._grammar_weakness_penalty(title)

        return round(min(max(score, 0), 10), 3)

    def quick_score(self, title: str, keyword: str) -> float:
        base = 0.0
        title_lower = title.lower()
        keyword_lower = keyword.lower()
        if title_lower.startswith(keyword_lower):
            base += 1.0
        if keyword_lower in title_lower:
            base += 1.0
        if 15 <= len(title) <= 60:
            base += 1.5
        if title and title[0].isupper():
            base += 0.5
        return round(min(base, 4.0), 2)
