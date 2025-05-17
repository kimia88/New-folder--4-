import json
import logging
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any
from difflib import SequenceMatcher

from services.seo_title_evaluator import SEOTitleEvaluator
from services.llm_service import QService  # فرض می‌کنیم همین کلاس QService است


class SEOService:
    def __init__(self, db, q_service: QService, min_score=8.5, retries=8, delay=4.0, max_backoff=60.0):
        self.db = db
        self.q_service = q_service
        self.min_score = min_score
        self.retries = retries
        self.delay = delay
        self.max_backoff = max_backoff
        self.evaluator = SEOTitleEvaluator()

        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    def extract_focus_keyword(self, title: str) -> str:
        stopwords = {"the", "of", "and", "a", "an", "to", "in", "on", "for", "with", "at", "by",
                     "را", "و", "که", "از", "این", "با", "به"}
        words = [w.lower() for w in re.findall(r'\w+', title) if w.lower() not in stopwords]
        return " ".join(words[:3]) if words else title.strip().lower()

    def generate_title_for_all(self):
        logging.info("🚀 شروع فرآیند بهینه‌سازی عناوین برای سئو...")
        contents = self.db.get_all_purecontents()
        results = []

        for content_id, title, *_rest, lang_id in contents:
            if not title or not title.strip():
                logging.info(f"❌ Skipped empty title for content_id {content_id}")
                continue

            keyword = self.extract_focus_keyword(title)
            original_score = self.evaluator.evaluate_title(title, keyword)
            best_title, best_score = title, original_score

            candidates = [{"title": title, "score": original_score}]
            backoff = self.delay

            for attempt in range(1, self.retries + 1):
                prompt = self._build_prompt(best_title, lang_id, best_score, attempt > 2 or best_score < self.min_score / 2)

                try:
                    response_initial = self.q_service.send_request(prompt)
                    if "error" in response_initial:
                        raise ValueError("Error in sending request")

                    raw_response = self.q_service.get_response()
                    if not raw_response or "⚠️" in raw_response:
                        raise ValueError("No valid response from model")

                    data = self._parse_response(raw_response)
                    candidate = data.get("optimized_title", "").strip()

                    if not candidate:
                        raise ValueError("Empty optimized_title")

                    if self._looks_gibberish(candidate):
                        raise ValueError("Gibberish detected")

                    score = self.evaluator.evaluate_title(candidate, keyword)
                    similarity = self._similarity_ratio(best_title, candidate)

                    candidates.append({"title": candidate, "score": score})

                    if score > best_score and similarity < 0.8:
                        best_title = candidate
                        best_score = score
                        backoff = self.delay  # reset backoff after improvement

                except (ValueError, json.JSONDecodeError):
                    pass  # خطاها رو نادیده می‌گیریم و تلاش بعدی رو می‌زنیم
                except Exception:
                    pass

                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

            if best_score > original_score and best_title != title:
                self._update_title_in_database(content_id, best_title, best_score)
                logging.info(
                    f"✅ content_id: {content_id} | original_score: {original_score:.2f} | optimized_score: {best_score:.2f}\n"
                    f"    original_title: {title}\n"
                    f"    optimized_title: {best_title}"
                )
            else:
                logging.info(f"ℹ️ content_id: {content_id} | No meaningful improvement")

            results.append({
                "content_id": content_id,
                "original_title": title,
                "optimized_title": best_title,
                "seo_score": best_score,
                "candidates": candidates
            })

        self._save_results(results)
        logging.info("🎉 فرآیند بهینه‌سازی عناوین به پایان رسید")

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        json_matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
        if not json_matches:
            raise ValueError("JSON not found in response.")
        try:
            return json.loads(json_matches[0])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

    def _build_prompt(self, title: str, lang_id: int, last_score: float, emphasize_change: bool) -> str:
        if lang_id == 1:  # Persian
            prompt = (
                "عنوان زیر را به صورت بهینه برای سئو بازنویسی کن. عنوان باید معنی‌دار، جذاب، خلاصه، و شامل کلیدواژه‌های مهم باشد. "
                "فقط خروجی JSON مانند زیر برگردان:\n"
                "{\n"
                "  \"original_title\": \"...\",\n"
                "  \"optimized_title\": \"...\",\n"
                "  \"score\": عدد بین ۰ تا ۱۰\n"
                "}\n\n"
                f"عنوان:\n{title}"
            )
            if emphasize_change:
                prompt += "\n❗ لطفاً نسخه‌ای کاملاً جدید و با سئو قوی‌تر پیشنهاد بده."
        else:  # English
            prompt = (
                "Rewrite the following title to be SEO optimized. Make it clear, attractive, and keyword-rich. "
                "Respond ONLY with a JSON like:\n"
                "{\n"
                "  \"original_title\": \"...\",\n"
                "  \"optimized_title\": \"...\",\n"
                "  \"score\": number between 0 and 10\n"
                "}\n\n"
                f"Title:\n{title}"
            )
            if emphasize_change:
                prompt += "\n❗ Please provide a much better SEO version."

        return prompt

    def _save_results(self, results: List[Dict[str, Any]]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"seo_output/seo_results_{ts}.json"
        os.makedirs("seo_output", exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"📁 Results saved: {path}")
        except Exception as e:
            logging.error(f"❌ Failed to save results: {e}")

    def _update_title_in_database(self, content_id: int, optimized_title: str, seo_score: float):
        try:
            self.db.update_pure_content(content_id, optimized_title)
            logging.info(f"✅ Updated content_id {content_id} with: {optimized_title} (Score: {seo_score:.2f})")
        except Exception as e:
            logging.error(f"❌ DB update error for content_id {content_id}: {e}")

    def _similarity_ratio(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _looks_gibberish(self, title: str) -> bool:
        words = re.findall(r'\w+', title)
        if not words:
            return True
        gibberish_count = 0
        for w in words:
            if len(w) > 20 or not re.search(r'[a-zA-Z0-9\u0600-\u06FF]', w):
                gibberish_count += 1
        return gibberish_count / len(words) > 0.3
