import re
import logging
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer, util


class SEOTitleEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', weights: Optional[Dict[str, float]] = None):
        """
        ساختاردهنده کلاس ارزیابی عنوان برای سئو
        :param model_name: نام مدل sentence-transformers برای embedding
        :param weights: دیکشنری وزن‌های پارامترهای مختلف (اختیاری)
        """
        self.model = SentenceTransformer(model_name)

        # وزن‌های پیش‌فرض
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
            "semantic": 3.0
        }

    def update_weights_from_feedback(self, feedback: Dict[str, float]):
        """
        به‌روزرسانی وزن‌ها بر اساس بازخورد (یادگیری ساده)
        :param feedback: دیکشنری وزن‌های جدید
        """
        for k, v in feedback.items():
            if k in self.weights:
                self.weights[k] = (self.weights[k] + v) / 2

    def _length_score(self, length: int) -> float:
        """
        امتیاز طول عنوان (اوج بین 15 تا 60 کاراکتر)
        """
        if length < 15:
            return -0.7
        elif 15 <= length <= 60:
            # نمره خطی: بیشترین در 37، کاهش تدریجی اطراف آن
            return self.weights["length_optimal"] * (1 - abs(37 - length) / 37)
        else:
            return -1.0

    def _count_phrase_occurrences(self, text: str, phrase: str) -> int:
        """
        شمارش تعداد تکرار دقیق یک عبارت چندکلمه‌ای در متن
        :param text: متن به صورت کلمه‌های جدا شده
        :param phrase: عبارت چندکلمه‌ای
        :return: تعداد تکرار
        """
        text_words = text.split()
        phrase_words = phrase.split()
        count = 0
        for i in range(len(text_words) - len(phrase_words) + 1):
            if text_words[i:i + len(phrase_words)] == phrase_words:
                count += 1
        return count

    def _repeated_keyword_penalty(self, count: int, total_words: int) -> float:
        """
        جریمه بر اساس نسبت تکرار کلیدواژه
        """
        ratio = count / max(total_words, 1)
        if ratio > 0.3:
            return self.weights["repeated_keyword_penalty"] * (ratio * 3)
        return 0.0

    def evaluate_title(self, title: str, focus_keyword: str) -> float:
        """
        ارزیابی کلی عنوان بر اساس پارامترهای مختلف
        :param title: عنوان مورد ارزیابی
        :param focus_keyword: کلیدواژه اصلی
        :return: امتیاز بین 0 تا 10
        """
        score = 0.0
        title_lower = title.lower()
        focus_lower = focus_keyword.lower()

        # کلیدواژه در ابتدا
        if title_lower.startswith(focus_lower):
            score += self.weights["keyword_start"]
        elif focus_lower in title_lower:
            score += self.weights["keyword_contains"]

        # طول عنوان
        length = len(title.strip())
        score += self._length_score(length)

        # شروع با حرف بزرگ
        if title and title[0].isupper():
            score += self.weights["capitalized"]

        # علائم نگارشی مفید
        if any(ch in title for ch in [":", "-", "?"]):
            score += self.weights["punctuation"]

        # عبارات جستجوی صوتی
        voice_search_phrases = ["how to", "what is", "why is", "where can", "who is"]
        if any(phrase in title_lower for phrase in voice_search_phrases):
            score += self.weights["voice_search"]

        # شمارش دقیق تکرار کلیدواژه (برای چندکلمه‌ای‌ها دقیق‌تر)
        words = re.findall(r'\w+', title_lower)
        count_focus = self._count_phrase_occurrences(" ".join(words), focus_lower)
        score += self._repeated_keyword_penalty(count_focus, len(words))

        # نسبت کلمات یکتا
        unique_words = len(set(words))
        ratio = unique_words / max(len(words), 1)
        if ratio > 0.5:
            score += self.weights["unique_word_ratio_good"] * ratio
        else:
            score += self.weights["unique_word_ratio_bad"] * (1 - ratio)

        # ارزیابی معنایی با embedding (با fallback)
        try:
            title_emb = self.model.encode(title, convert_to_tensor=True, normalize_embeddings=True)
            keyword_emb = self.model.encode(focus_keyword, convert_to_tensor=True, normalize_embeddings=True)
            semantic_sim = util.cos_sim(title_emb, keyword_emb).item()
            score += semantic_sim * self.weights["semantic"]
        except Exception as e:
            logging.warning(f"Semantic evaluation failed for '{title}': {e}")
            # fallback: نمره متوسط
            score += 0.5 * self.weights["semantic"]

        # محدود کردن نمره به 0 تا 10
        final_score = round(min(max(score, 0), 10), 3)
        return final_score
