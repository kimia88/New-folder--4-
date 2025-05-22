import re
import logging
from typing import Optional, Dict, List
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache

try:
    import hazm
    from hazm import Normalizer, word_tokenize, Lemmatizer
    hazm_available = True
except ImportError:
    hazm_available = False

import spacy
try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    nlp_en = None


class SEOTitleEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', weights: Optional[Dict[str, float]] = None):
        self.model = SentenceTransformer(model_name)

        self.use_farsi = hazm_available
        if hazm_available:
            self.lemmatizer_fa = Lemmatizer()
            self.normalizer_fa = Normalizer()
        self.nlp_en = nlp_en

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
        if self.use_farsi:
            text = self.normalizer_fa.normalize(text)
            tokens = word_tokenize(text)
            return [self.lemmatizer_fa.lemmatize(t) for t in tokens if t.isalpha()]
        elif self.nlp_en:
            doc = self.nlp_en(text)
            return [token.lemma_ for token in doc if token.is_alpha]
        else:
            return re.findall(r'\b\w+\b', text.lower())

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
        patterns = [
            r"همه چیز درباره",
            r"چیست\??",
            r"راهنمای کامل",
            r"بهترین .* چیست",
            r"(you|won't|never|must|shocking|secret|revealed)",
        ]
        if any(re.search(p, title.lower()) for p in patterns):
            return self.weights["clickbait_penalty"]
        return 0.0

    def _is_gibberish(self, title: str) -> float:
        words = title.split()
        long_words = [w for w in words if len(w) > 20 or not re.search(r'[a-zA-Z\u0600-\u06FF0-9]', w)]
        if len(long_words) / max(len(words), 1) > 0.3:
            return self.weights["gibberish_penalty"]
        return 0.0

    def _grammar_weakness_penalty(self, title: str) -> float:
        if self.nlp_en:
            doc = self.nlp_en(title)
            root_tokens = [t for t in doc if t.dep_ in ("ROOT", "nsubj")]
            if len(root_tokens) < 1:
                return self.weights["weak_grammar_penalty"]
        return 0.0

    @lru_cache(maxsize=512)
    def _encode(self, text: str):
        return self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

    def evaluate_title(self, title: str, focus_keyword: str) -> float:
        score = 0.0
        title_lower = title.lower()
        focus_lower = focus_keyword.lower()

        if title_lower.startswith(focus_lower):
            score += self.weights["keyword_start"]
        elif focus_lower in title_lower:
            pos = title_lower.find(focus_lower)
            score += self.weights["keyword_contains"] * (1 - pos / len(title_lower))

        score += self._length_score(len(title.strip()))

        if title and title[0].isupper():
            score += self.weights["capitalized"]

        if any(ch in title for ch in [":", "-", "?"]):
            score += self.weights["punctuation"]

        voice_search_phrases = ["how to", "what is", "why is", "where can", "who is", "چگونه", "چیست", "چرا", "کجا"]
        if any(p in title_lower for p in voice_search_phrases):
            score += self.weights["voice_search"]

        lemmatized_words = self._lemmatize(title)
        count_focus = self._count_keyword_occurrences(title, focus_keyword)
        score += self._repeated_keyword_penalty(count_focus, len(lemmatized_words))

        unique_words = len(set(lemmatized_words))
        ratio = unique_words / max(len(lemmatized_words), 1)
        if ratio > 0.5:
            score += self.weights["unique_word_ratio_good"] * ratio
        else:
            score += self.weights["unique_word_ratio_bad"] * (1 - ratio)

        try:
            title_emb = self._encode(title)
            keyword_emb = self._encode(focus_keyword)
            sim = util.cos_sim(title_emb, keyword_emb).item()
            score += sim * self.weights["semantic"]
        except Exception as e:
            logging.warning(f"Semantic embedding failed: {e}")
            score += 0.5 * self.weights["semantic"]

        score += self._is_clickbait(title)
        score += self._is_gibberish(title)
        score += self._grammar_weakness_penalty(title)

        return round(min(max(score, 0), 10), 3)

    def quick_score(self, title: str, focus_keyword: str) -> float:
        title_lower = title.lower()
        focus_lower = focus_keyword.lower()
        base = 0.0
        if title_lower.startswith(focus_lower):
            base += 1.0
        if focus_lower in title_lower:
            base += 1.0
        if 15 <= len(title) <= 60:
            base += 1.5
        if title and title[0].isupper():
            base += 0.5
        return round(min(base, 4.0), 2)
