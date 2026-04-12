import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


class JobService:
    def __init__(self):
        self.app_id = os.getenv("ADZUNA_APP_ID", "")
        self.app_key = os.getenv("ADZUNA_APP_KEY", "")
        self.country = os.getenv("ADZUNA_COUNTRY", "in")
        self.base_url = f"https://api.adzuna.com/v1/api/jobs/{self.country}/search/1"

    def _fetch_jobs(self, query_params, safe_limit):
        base_params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": safe_limit,
            "content-type": "application/json",
        }
        params = {**base_params, **query_params}
        request_url = f"{self.base_url}?{urlencode(params)}"

        try:
            with urlopen(request_url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError):
            return []

        jobs = []
        for item in payload.get("results", []):
            company_name = (item.get("company") or {}).get("display_name") or "Unknown Company"
            location = (item.get("location") or {}).get("display_name") or "Remote/Not specified"
            jobs.append(
                {
                    "company": company_name,
                    "role": item.get("title") or "Role not specified",
                    "location": location,
                    "apply_link": item.get("redirect_url") or "#",
                }
            )

        return jobs[:safe_limit]

    def get_recommendations(self, skills_query, limit=8):
        if not skills_query or not self.app_id or not self.app_key:
            return []

        normalized_query = " ".join(str(skills_query).replace(",", " ").split())
        if not normalized_query:
            return []

        safe_limit = max(5, min(int(limit), 10))
        words = [w for w in normalized_query.split() if len(w) > 1]
        top_words = words[:8] or normalized_query.split()[:8]
        broad_keywords = " ".join(top_words)
        focused_keywords = " ".join(top_words[:3])

        search_attempts = [
            {"what_or": broad_keywords},
            {"what": focused_keywords},
            {"what_or": focused_keywords},
            {"what_or": "software developer data analyst python sql"},
        ]

        for attempt in search_attempts:
            jobs = self._fetch_jobs(attempt, safe_limit)
            if jobs:
                return jobs

        return []
