import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from sentence_transformers import util
from services.seo_title_evaluator import SEOTitleEvaluator


class SEOServiceAdvanced:
    def __init__(self, db, q_service, min_score: float = 8.5, retries: int = 8, delay: float = 4.0, max_backoff: float = 60.0):
        self.db = db
        self.q_service = q_service
        self.base_min_score = min_score
        self.retries = retries
        self.delay = delay
        self.max_backoff = max_backoff
        self.evaluator = SEOTitleEvaluator()
        self.past_scores: List[float] = []

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("seo_service_advanced.log", encoding="utf-8")
            ]
        )

    def extract_focus_keyword(self, title: str) -> str:
        words = re.findall(r'\b\w+\b', title.lower())
        stopwords = set(["برای", "از", "در", "با", "به", "و", "یا", "این", "که", "را", "یک", "تا", "آن", "می"])  # فارسی
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return " ".join(sorted(keywords, key=len, reverse=True)[:5])

    def _semantic_similarity(self, a: str, b: str) -> float:
        try:
            emb_a = self.evaluator.model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
            emb_b = self.evaluator.model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
            return util.cos_sim(emb_a, emb_b).item()
        except Exception as e:
            logging.warning("Semantic similarity failed: %s", e)
            return 0.0

    def _looks_gibberish(self, title: str) -> bool:
        words = re.findall(r'\w+', title)
        gibberish_count = sum(1 for w in words if len(w) > 20 or not re.search(r'[a-zA-Z0-9؀-ۿ]', w))
        return gibberish_count / len(words) > 0.3 if words else False

    def analyze_weaknesses(self, title: str, keyword: str) -> List[str]:
        weaknesses = []
        title_lower = title.lower()
        keyword_lower = keyword.lower()

        if not title_lower.startswith(keyword_lower):
            weaknesses.append("کلمه کلیدی باید ابتدای عنوان باشد")

        if len(title) < 15:
            weaknesses.append("عنوان خیلی کوتاه است")
        elif len(title) > 60:
            weaknesses.append("عنوان خیلی بلند است")

        count_focus = title_lower.count(keyword_lower)
        if count_focus / max(len(title.split()), 1) > 0.3:
            weaknesses.append("کلمه کلیدی بیش از حد تکرار شده است")

        if title and not title[0].isupper():
            weaknesses.append("حرف اول عنوان باید بزرگ باشد")

        if not any(p in title for p in [":", "-", "?"]):
            weaknesses.append("بهتر است از نشانه نگاری استفاده شود")

        if self._looks_gibberish(title):
            weaknesses.append("عنوان به نظر نامفهوم یا gibberish است")

        return weaknesses

    def improve_title(self, title: str, keyword: str, weaknesses: List[str]) -> Tuple[str, float, float]:
     original_score = self.evaluator.evaluate_title(title, keyword)
     keyword_lower = keyword.lower()

    # بهبودها بر اساس ضعف‌های مشخص‌شده
     if "کلمه کلیدی باید ابتدای عنوان باشد" in weaknesses:
        if not title.lower().startswith(keyword_lower):
            # حذف نسخه‌های تکراری کلمه کلیدی از جای دیگه
            title_no_kw = re.sub(re.escape(keyword), '', title, flags=re.IGNORECASE).strip()
            title = f"{keyword} {title_no_kw}"

     if "عنوان خیلی کوتاه است" in weaknesses:
        if len(title) < 20:
            title += " - راهنمای کاربردی"

     if "عنوان خیلی بلند است" in weaknesses:
        if len(title) > 60:
            title = title[:57].rstrip() + "..."

     if "کلمه کلیدی بیش از حد تکرار شده است" in weaknesses:
        words = title.split()
        count_kw = sum(1 for w in words if w.lower() == keyword_lower)
        if count_kw / max(len(words), 1) > 0.3:
            words = [w for w in words if w.lower() != keyword_lower]
            title = f"{keyword} {' '.join(words)}"

     if "حرف اول عنوان باید بزرگ باشد" in weaknesses:
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]

     if "بهتر است از نشانه نگاری استفاده شود" in weaknesses:
        if not any(p in title for p in [":", "-", "؟", "؛"]):
            title += " - بررسی کامل"

     if "عنوان به نظر نامفهوم یا gibberish است" in weaknesses:
        title = f"{keyword} - ساده و قابل‌فهم برای همه"

    # نمره نهایی
     new_score = self.evaluator.evaluate_title(title, keyword)
     return title, original_score, new_score


    def _parse_response(self, raw: str) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*?"original_title".*?"score"\s*:\s*\d+.*?\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("JSON block not found.")
        except Exception as e:
            logging.debug("Failed to parse response: %s | Raw: %s", e, raw)
            raise

    def _build_prompt(self, original: str, lang_id: int, emphasize_change: bool, previous: Optional[str] = None) -> str:
        examples_fa = (
            'مثال:\n'
            '{\n'
            '  "original_title": "چرا خواب کافی مهم است؟",\n'
            '  "optimized_title": "خواب کافی؛ کلید طلایی برای سلامت ذهن و بدن",\n'
            '  "score": 9.2\n'
            '}\n\n'
            '{\n'
            '  "original_title": "ترفندهای عکاسی با موبایل",\n'
            '  "optimized_title": "۱۰ ترفند طلایی عکاسی حرفه‌ای با موبایل",\n'
            '  "score": 9.0\n'
            '}\n\n'
        )

        prompt = (
            "عنوان زیر را برای بهینه‌سازی SEO بازنویسی کن. فقط خروجی را در قالب JSON زیر تولید کن:\n"
            '{\n  "original_title": "...",\n  "optimized_title": "...",\n  "score": عددی بین ۰ تا ۱۰\n}\n\n'
            + examples_fa +
            f"عنوان:\n{original.strip()}\n"
        )

        if previous:
            prompt += f"\nعنوان پیشنهادی قبلی:\n{previous}"

        if emphasize_change:
            prompt += (
                "\n❗ لطفاً نسخه‌ای بسیار خلاقانه و تاثیرگذار با تمرکز بر کلمات کلیدی ارائه بده."
                "\n🔍 از عباراتی با قدرت جستجوی بالا استفاده کن و جذابیت انسانی و سئو را همزمان در نظر بگیر."
            )
        else:
            prompt += "\n✅ فقط بازنویسی ملایم با حفظ معنای اصلی و بهبود کلمات کلیدی."

        return prompt

    def _attempt_optimization(self, title: str, keyword: str, lang_id: int, original_score: float) -> List[Dict[str, Any]]:
        results = []
        backoff = self.delay
        previous_title = None

        focus_areas = {
            "keyword": False,
            "creativity": False,
            "grammar": False,
            "semantic": False,
        }

        for attempt in range(1, self.retries + 1):
            emphasize_parts = []

            # تحلیل ضعف‌ها با تابع analyze_weaknesses برای تاکید پویا
            weaknesses = self.analyze_weaknesses(title, keyword)
            if original_score < self.base_min_score:
                if "کلمه کلیدی باید ابتدای عنوان باشد" in weaknesses:
                    focus_areas["keyword"] = True
                if "عنوان به نظر نامفهوم یا gibberish است" in weaknesses:
                    focus_areas["creativity"] = True
                # اینجا می‌تونی گرامر رو اضافه کنی مثلاً:
                # if some_grammar_check(title) == False:
                #    focus_areas["grammar"] = True
                if original_score < self.base_min_score / 2:
                    focus_areas = {k: True for k in focus_areas}

            emphasize_change = any(focus_areas.values())

            prompt = self._build_prompt(title, lang_id, emphasize_change, previous=previous_title)

            if emphasize_change:
                if focus_areas["keyword"]:
                    emphasize_parts.append("تمرکز روی استفاده بهتر و دقیق‌تر از کلمات کلیدی باشد.")
                if focus_areas["creativity"]:
                    emphasize_parts.append("خلاقیت بیشتر و دوری از عبارات نامفهوم و بی‌ربط.")
                if focus_areas["grammar"]:
                    emphasize_parts.append("اصلاح گرامر و نگارش صحیح.")
                if focus_areas["semantic"]:
                    emphasize_parts.append("ارتباط معنایی قوی‌تر با موضوع.")

                prompt += "\n🔎 لطفاً به نکات زیر توجه ویژه داشته باش:\n" + "\n".join(f"- {p}" for p in emphasize_parts)

            logging.info(f"🔄 Attempt {attempt} | Prompt sent with emphasis on: {emphasize_parts}")

            try:
                self.q_service.send_request(prompt)
                time.sleep(1.0)
                raw = self.q_service.get_response()
                if not raw or "⚠️" in raw:
                    raise ValueError("Invalid response")

                data = self._parse_response(raw)
                candidate = data["optimized_title"].strip()
                if not candidate or self._looks_gibberish(candidate):
                    raise ValueError("Gibberish title")

                score = self.evaluator.evaluate_title(candidate, keyword)
                similarity = self._semantic_similarity(title, candidate)

                results.append({
                    "title": candidate,
                    "score": score,
                    "similarity": similarity
                })

                logging.info(f"✅ Attempt {attempt} | Score: {score:.2f} | Similarity: {similarity:.2f}")
                previous_title = candidate

                # بهبود خودکار ساده با تابع improve_title
                improved_title, _, improved_score = self.improve_title(candidate, keyword, weaknesses)
                if improved_score > score:
                    candidate = improved_title
                    score = improved_score

                if score > original_score:
                    original_score = score
                    title = candidate
                    # مجدد تحلیل ضعف برای دور بعد
                    weaknesses = self.analyze_weaknesses(title, keyword)
                    focus_areas = {
                        "keyword": "کلمه کلیدی باید ابتدای عنوان باشد" in weaknesses,
                        "creativity": "عنوان به نظر نامفهوم یا gibberish است" in weaknesses,
                        "grammar": False,
                        "semantic": False,
                    }

            except Exception as e:
                logging.warning(f"⚠️ Attempt {attempt} failed: {e}")

            time.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)

        return results

    def _select_best_attempt(self, original_score: float, original_title: str, attempts: List[Dict[str, Any]]) -> Tuple[str, float]:
        if not attempts:
            return original_title, original_score
        best = max(attempts, key=lambda x: x["score"])
        if best["score"] > original_score:
            return best["title"], best["score"]
        return original_title, original_score

    def _process_single_content(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
     content_id = content.get("content_id") or content.get("ContentID")
     title = (content.get("title") or content.get("Title") or "").strip()

     if not title:
        logging.info("❌ Skipped empty title for content_id %s", content_id)
        return None

     logging.info("▶️ Processing content_id %s | Title: %s", content_id, title)

     keyword = self.extract_focus_keyword(title)
     original_score = self.evaluator.evaluate_title(title, keyword)
     self.past_scores.append(original_score)

     attempts = self._attempt_optimization(title, keyword, lang_id=0, original_score=original_score)
     best_title, best_score = self._select_best_attempt(original_score, title, attempts)

     logging.info(f"✔️ Best optimized title score: {best_score:.2f} (original was {original_score:.2f})")

     return {
        "content_id": content_id,
        "original_title": title,
        "optimized_title": best_title,
        "score": best_score
    }


class SEOTitleEvaluator:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def evaluate_title(self, title: str, keyword: str) -> float:
        score = 0.0
        # امتیاز به نسبت طول عنوان
        score += min(len(title) / 15.0, 2.0) * 2.0
        # امتیاز کلمه کلیدی: چند بار کلمه کلیدی داخل عنوان است
        keyword_count = title.lower().count(keyword.lower())
        score += min(keyword_count, 3) * 2.0
        # امتیاز بر اساس تعداد کلمات معنادار
        meaningful_words = len([w for w in re.findall(r'\b\w+\b', title) if len(w) > 2])
        score += min(meaningful_words / 7.0, 2.0) * 2.0

        # اضافه کردن فاکتور معنایی (مثلاً تشابه معنایی با کلمه کلیدی)
        try:
            emb_title = self.model.encode(title, convert_to_tensor=True, normalize_embeddings=True)
            emb_keyword = self.model.encode(keyword, convert_to_tensor=True, normalize_embeddings=True)
            semantic_score = util.cos_sim(emb_title, emb_keyword).item()
            score += semantic_score * 4.0  # وزن معنایی بیشتر
        except Exception as e:
            logging.warning("Semantic evaluation failed: %s", e)

        # نرمال‌سازی نهایی
        return min(score, 10.0)
