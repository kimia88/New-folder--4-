import json
import logging
import re
import time
from datetime import datetime
from typing import List, Dict, Any
from difflib import SequenceMatcher

import spacy
from sentence_transformers import util
from services.seo_title_evaluator import SEOTitleEvaluator

nlp = spacy.load("en_core_web_sm")

class SEOServiceAdvanced:
    def __init__(
        self,
        db,
        q_service,
        min_score=8.5,
        retries=8,
        delay=4.0,
        max_backoff=60.0,
    ):
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
                logging.FileHandler("seo_service_advanced.log", encoding="utf-8"),
            ],
        )

    def extract_focus_keyword(self, title: str) -> str:
        doc = nlp(title.lower())
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        keywords = sorted(keywords, key=lambda w: len(w), reverse=True)[:5]
        return " ".join(keywords) if keywords else title.strip().lower()

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
        gibberish_count = sum(1 for w in words if len(w) > 20 or not re.search(r'[a-zA-Z0-9؀-ۿ]', w))
        return gibberish_count / len(words) > 0.3

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*?\"original_title\".*?\"score\"\s*:\s*\d+.*?\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("JSON block not found.")
        except Exception as e:
            logging.debug(f"Failed to parse response: {e} | Raw: {raw}")
            raise

    def _build_prompt(self, title: str, lang_id: int, emphasize_change: bool) -> str:
        if lang_id == 1:
            prompt = (
                "عنوان زیر را به صورت جدید، سئو شده و فقط با خروجی JSON بازنویسی کن."
                " زبان را حفظ کن و از ترجمه خودداری کن. فقط این فرمت را تولید کن:\n"
                '{\n  "original_title": "...",\n  "optimized_title": "...",\n  "score": عدد بین ۰ تا ۱۰\n}\n'
                f"\nعنوان:\n{title}"
            )
        else:
            prompt = (
                "Rewrite the following title for SEO, preserving the original language."
                " Return ONLY a JSON in this format:\n"
                '{\n  "original_title": "...",\n  "optimized_title": "...",\n  "score": number between 0 and 10\n}\n'
                f"\nTitle:\n{title}"
            )
        if emphasize_change:
            prompt += "\n❗ Output must be a strong, new version of the title."
        return prompt

    def _update_title_in_database(self, content_id: int, optimized_title: str, seo_score: float):
        try:
            self.db.update_pure_content(content_id, optimized_title)
            logging.info(f"✅ Updated content_id {content_id} with: {optimized_title} (Score: {seo_score:.2f})")
        except Exception as e:
            logging.error(f"❌ DB update error for content_id {content_id}: {e}")

    def _process_single_content(self, content: Dict[str, Any]):
        content_id = content.get("content_id") or content.get("ContentID")
        title = (content.get("title") or content.get("Title") or "").strip().replace("  ", " ")

        logging.info(f"▶️ Start processing content_id {content_id} | Title: {title}")

        if not title:
            logging.info(f"❌ Skipped empty title for content_id {content_id}")
            return None

        keyword = self.extract_focus_keyword(title)
        original_score = self.evaluator.evaluate_title(title, keyword)
        self.past_scores.append(original_score)

        best_title = title
        best_score = original_score

        lang_id = 1 if any('\u0600' <= c <= '\u06FF' for c in title) else 0
        backoff = self.delay

        for attempt in range(1, self.retries + 1):
            emphasize_change = attempt > 2 or best_score < self.base_min_score / 2
            prompt = self._build_prompt(best_title, lang_id, emphasize_change)

            logging.info(f"🔄 Attempt {attempt} | Sending prompt for content_id {content_id}")

            try:
                self.q_service.send_request(prompt)
                time.sleep(1.0)
                raw_response = self.q_service.get_response()
                logging.info(f"Received raw response for content_id {content_id}: {raw_response[:200]}")

                if not raw_response or "⚠️" in raw_response:
                    raise ValueError("Invalid or empty response from model")

                data = self._parse_response(raw_response)
                candidate = data["optimized_title"].strip()

                if not candidate or self._looks_gibberish(candidate):
                    raise ValueError("Gibberish or empty optimized title")

                score = self.evaluator.evaluate_title(candidate, keyword)

                if score > best_score and self._semantic_similarity(best_title, candidate) > 0.6:
                    best_title = candidate
                    best_score = score
                    backoff = self.delay
                    logging.info(f"✅ Improved title on attempt {attempt} with score {score:.2f}")
                else:
                    logging.info(f"ℹ️ No improvement on attempt {attempt} (Best score: {best_score:.2f})")

            except Exception as e:
                logging.warning(f"⚠️ Attempt {attempt} failed for content_id {content_id} | Error: {e}")

            logging.info(f"⏳ Sleeping for {backoff:.1f} seconds before next attempt")
            time.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)

        if best_score > original_score:
            self._update_title_in_database(content_id, best_title, best_score)
        else:
            logging.info(f"ℹ️ content_id {content_id} | No better SEO score found. Keeping original title.")

        logging.info(f"🏁 Finished processing content_id {content_id} | Final title: \"{best_title}\" | Final score: {best_score:.2f}")

        return {
            "content_id": content_id,
            "original_title": title,
            "optimized_title": best_title,
            "seo_score": best_score,
            "keyword": keyword,
        }

    def optimize_titles(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [res for content in contents if (res := self._process_single_content(content))]
