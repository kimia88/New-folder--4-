import json
import logging
import os
import re
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import spacy  
from sentence_transformers import SentenceTransformer, util
from services.seo_title_evaluator import SEOTitleEvaluator  # فرض بر این است که این کلاس موجود است و درست کار می‌کند

# بارگذاری مدل NLP برای کلیدواژه
nlp = spacy.load("en_core_web_sm")  # برای فارسی هم می‌توانید مدل مناسب را جایگزین کنید

class SEOServiceAdvanced:
    def __init__(
        self,
        db,
        q_service,
        min_score: float = 8.5,
        retries: int = 8,
        delay: float = 4.0,
        max_backoff: float = 60.0,
        max_workers: int = 5,
    ):
        self.db = db
        self.q_service = q_service
        self.base_min_score = min_score
        self.retries = retries
        self.delay = delay
        self.max_backoff = max_backoff
        self.evaluator = SEOTitleEvaluator()
        self.past_scores: List[float] = []
        self.max_workers = max_workers

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("seo_service_advanced.log", encoding="utf-8")
            ]
        )

    def extract_focus_keyword(self, title: str) -> str:
        doc = nlp(title.lower())
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        keywords = sorted(keywords, key=lambda w: len(w), reverse=True)[:5]
        return " ".join(keywords) if keywords else title.strip().lower()

    def _dynamic_min_score(self) -> float:
        if len(self.past_scores) >= 5:
            mean = statistics.mean(self.past_scores)
            stdev = statistics.stdev(self.past_scores)
            dynamic_threshold = min(10.0, max(self.base_min_score, mean + stdev / 2))
            logging.debug(f"Dynamic min_score adjusted to {dynamic_threshold:.2f}")
            return dynamic_threshold
        return self.base_min_score

    def _semantic_similarity(self, a: str, b: str) -> float:
        try:
            emb_a = self.evaluator.model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
            emb_b = self.evaluator.model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
            return util.cos_sim(emb_a, emb_b).item()
        except Exception as e:
            logging.warning(f"Semantic similarity failed: {e}")
            return 0.0

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
        ratio = gibberish_count / len(words)
        if ratio > 0.3:
            logging.debug(f"Gibberish detected ({ratio:.2f}) in title: {title}")
        return ratio > 0.3

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        json_matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
        if not json_matches:
            logging.debug(f"Raw response:\n{raw}")
            raise ValueError("JSON not found in response.")
        try:
            return json.loads(json_matches[0])
        except json.JSONDecodeError as e:
            logging.debug(f"JSON decode error: {e}\nRaw JSON: {json_matches[0]}")
            raise

    def _build_prompt(self, title: str, lang_id: int, last_score: float, emphasize_change: bool) -> str:
        if lang_id == 1:  # Persian
            prompt = (
                "عنوان زیر را به صورت بهینه و کاملاً جدید برای سئو بازنویسی کن. عنوان باید معنی‌دار، جذاب، خلاصه، "
                "و شامل کلیدواژه‌های مهم باشد. از کلمات تکراری و اضافی پرهیز کن.\n"
                "خروجی فقط یک JSON مانند زیر باشد:\n"
                "{\n  \"original_title\": \"...\",\n  \"optimized_title\": \"...\",\n  \"score\": عدد بین ۰ تا ۱۰\n}\n\n"
                f"عنوان:\n{title}"
            )
            if emphasize_change:
                prompt += "\n❗ لطفاً نسخه‌ای کاملاً جدید، قوی‌تر و متفاوت از نظر سئو تولید کن."
        else:
            prompt = (
                "Rewrite the following title to be completely new and SEO optimized. It should be meaningful, attractive, concise, "
                "and keyword-rich. Avoid repetitive or redundant words.\n"
                "Return ONLY a JSON like this:\n"
                "{\n  \"original_title\": \"...\",\n  \"optimized_title\": \"...\",\n  \"score\": number between 0 and 10\n}\n\n"
                f"Title:\n{title}"
            )
            if emphasize_change:
                prompt += "\n❗ Please generate a completely new, stronger, and distinct SEO version."

        return prompt

    def _update_title_in_database(self, content_id: int, optimized_title: str, seo_score: float):
        try:
            self.db.update_pure_content(content_id, optimized_title)
            logging.info(f"✅ Updated content_id {content_id} with: {optimized_title} (Score: {seo_score:.2f})")
        except Exception as e:
            logging.error(f"❌ DB update error for content_id {content_id}: {e}")

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

    def _combine_scores(self, seo_score: float, engagement_score: Optional[float]) -> float:
        # در اینجا فقط نمره سئو را استفاده می‌کنیم
        return seo_score

    def _process_single_content(self, content):
        content_id, title, lang_id = content
        if not title or not title.strip():
            logging.info(f"❌ Skipped empty title for content_id {content_id}")
            return None

        keyword = self.extract_focus_keyword(title)
        original_score = self.evaluator.evaluate_title(title, keyword)
        self.past_scores.append(original_score)

        best_title = title
        best_score = original_score
        min_score_dynamic = self._dynamic_min_score()

        candidates = [{"title": title, "score": original_score}]
        backoff = self.delay

        for attempt in range(1, self.retries + 1):
            emphasize_change = attempt > 2 or best_score < min_score_dynamic / 2
            prompt = self._build_prompt(best_title, lang_id, best_score, emphasize_change=emphasize_change)

            try:
                self.q_service.send_request(prompt)
                raw_response = self.q_service.get_response()
                if not raw_response or "⚠️" in raw_response:
                    raise ValueError("Invalid response from model")

                data = self._parse_response(raw_response)
                candidate = data.get("optimized_title", "").strip()
                if not candidate or self._looks_gibberish(candidate):
                    raise ValueError("Gibberish or empty optimized title")

                semantic_sim = self._semantic_similarity(best_title, candidate)
                seq_sim = self._similarity_ratio(best_title, candidate)
                similarity = (semantic_sim + seq_sim) / 2

                score = self.evaluator.evaluate_title(candidate, keyword)
                combined_score = self._combine_scores(score, None)

                candidates.append({"title": candidate, "score": combined_score})

                # فقط نمره رو معیار قرار بده، تا پیشرفت واقعی در عنوان صورت بگیره
                if combined_score > best_score:
                    best_title = candidate
                    best_score = combined_score
                    backoff = self.delay
                    logging.info(f"✨ Improved title on attempt {attempt} with combined score {combined_score:.2f}")
                else:
                    logging.debug(f"No improvement on attempt {attempt} (score: {combined_score:.2f})")

            except Exception as e:
                logging.warning(f"Attempt {attempt} failed for content_id {content_id}: {e}")

            time.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)

        if best_score > original_score and best_title != title and best_score >= min_score_dynamic:
            self._update_title_in_database(content_id, best_title, best_score)
        else:
            logging.info(f"ℹ️ content_id {content_id} | No meaningful improvement")

        return {
            "content_id": content_id,
            "original_title": title,
            "optimized_title": best_title,
            "seo_score": best_score,
            "candidates": candidates
        }

    def optimize_titles(self, limit: int = 50):
        contents = self.db.select(
            f"""
            SELECT Id, Title, ContentLanguageId 
            FROM dbo.TblPureContent 
            WHERE Title IS NOT NULL AND LEN(Title) > 0 
            ORDER BY NEWID() 
            OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY
            """
        )

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_content = {executor.submit(self._process_single_content, c): c for c in contents}
            for future in as_completed(future_to_content):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    logging.error(f"Error processing content: {e}")

        self._save_results(results)
        return results
