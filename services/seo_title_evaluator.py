import re
import logging
from sentence_transformers import SentenceTransformer, util

class SEOTitleEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

        # وزن‌های قابل تنظیم
        self.weight_keyword_start = 1.5
        self.weight_keyword_contains = 2.5
        self.weight_length_optimal = 2.5
        self.weight_capitalized = 0.8
        self.weight_punctuation = 0.8
        self.weight_voice_search = 1.2
        self.weight_repeated_keyword_penalty = -0.6
        self.weight_unique_word_ratio_good = 0.7
        self.weight_unique_word_ratio_bad = -0.3
        self.weight_semantic = 3.0

    def evaluate_title(self, title: str, focus_keyword: str) -> float:
        score = 0.0
        title_lower = title.lower()
        focus_lower = focus_keyword.lower()

        # تشویق به وجود کلیدواژه در ابتدا یا متن عنوان
        if title_lower.startswith(focus_lower):
            score += self.weight_keyword_start
        elif focus_lower in title_lower:
            score += self.weight_keyword_contains

        # امتیاز طول عنوان
        length = len(title.strip())
        if length < 20:
            score -= 0.5
        elif 20 <= length <= 60:
            score += 1.5 + ((length - 20) / 40)  # تا 2.5 امتیاز
        else:
            score -= 1.0

        # شروع با حرف بزرگ
        if title and title[0].isupper():
            score += self.weight_capitalized

        # وجود علائم نگارشی مفید
        if any(ch in title for ch in [":", "-", "?"]):
            score += self.weight_punctuation

        # عبارات پرسشی متداول جستجوی صوتی
        voice_search_phrases = ["how to", "what is", "why is", "where can", "who is"]
        if any(phrase in title_lower for phrase in voice_search_phrases):
            score += self.weight_voice_search

        # تعداد زیاد کلیدواژه تمرکز در عنوان
        count_focus = title_lower.count(focus_lower)
        if count_focus > 2:
            score += self.weight_repeated_keyword_penalty

        # نسبت کلمات یکتا به کل کلمات (تنوع واژگان)
        words = re.findall(r'\w+', title_lower)
        unique_words = len(set(words))
        ratio = unique_words / max(len(words), 1)
        if ratio > 0.5:
            score += self.weight_unique_word_ratio_good
        else:
            score += self.weight_unique_word_ratio_bad

        # ارزیابی معنایی با embedding
        try:
            title_emb = self.model.encode(title, convert_to_tensor=True, normalize_embeddings=True)
            keyword_emb = self.model.encode(focus_keyword, convert_to_tensor=True, normalize_embeddings=True)
            semantic_sim = util.cos_sim(title_emb, keyword_emb).item()
            score += semantic_sim * self.weight_semantic
        except Exception as e:
            logging.warning(f"Semantic evaluation failed for title='{title}' & keyword='{focus_keyword}': {e}")

        # محدود کردن امتیاز به بازه ۰ تا ۱۰
        final_score = round(min(max(score, 0), 10), 3)
        return final_score
