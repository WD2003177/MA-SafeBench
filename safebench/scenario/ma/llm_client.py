from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OpenAICompatibleClient:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.environ.get("MA_LLM_API_KEY", "")
        self.base_url = os.environ.get("MA_LLM_BASE_URL") or config.get("ma_llm_base_url") or "https://api.openai.com/v1"
        self.model = os.environ.get("MA_LLM_MODEL") or config.get("ma_llm_model")
        self.timeout_s = float(os.environ.get("MA_LLM_TIMEOUT_S") or config.get("ma_llm_timeout_s", 10))
        self.temperature = float(config.get("ma_llm_temperature", 0.0))
        self.max_retries = int(config.get("ma_llm_max_retries", 1))

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def complete_json(self, scene_summary: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, schema_name: str = "ma_decision") -> Optional[Dict[str, Any]]:
        if not self.available():
            return None
        scene_payload = dict(scene_summary) if isinstance(scene_summary, dict) else {}
        prompt_prefix = scene_payload.pop("_ma_prompt", None)
        if not prompt_prefix:
            prompt_prefix = (
                "You control adversarial scenario actors in CARLA. Return only JSON with keys "
                "phase and commands. Commands may use behaviors cut_in_and_brake, block_ego_lane, recover. "
                "Do not output waypoints, throttle, steer, brake, or free-form code."
            )
        prompt = str(prompt_prefix) + "\n\n" + json.dumps(scene_payload, sort_keys=True)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return valid compact JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": False,
                    "schema": schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        url = self.base_url.rstrip("/") + "/chat/completions"
        last_error = None
        for _ in range(max(1, self.max_retries)):
            try:
                body, content = self._post_chat(url, payload)
                proposal = self._extract_json(content)
                if isinstance(proposal, dict):
                    proposal["_ma_raw_response"] = content
                    proposal["_ma_raw_body"] = body
                    proposal["_ma_llm_model"] = self.model
                return proposal
            except urllib.error.HTTPError as exc:
                last_error = self._http_error_text(exc)
                if "response_format" in payload:
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    try:
                        body, content = self._post_chat(url, fallback_payload)
                        proposal = self._extract_json(content)
                        if isinstance(proposal, dict):
                            proposal["_ma_raw_response"] = content
                            proposal["_ma_raw_body"] = body
                            proposal["_ma_llm_model"] = self.model
                            proposal["_ma_response_format_fallback"] = True
                        return proposal
                    except (KeyError, ValueError, urllib.error.URLError, TimeoutError) as retry_exc:
                        last_error = str(retry_exc)
            except (KeyError, ValueError, urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
        return {"phase": "recover", "commands": [], "_ma_llm_error": last_error or "llm_failed"}

    def _post_chat(self, url: str, payload: Dict[str, Any]):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return body, str(content).strip()

    def _extract_json(self, content: str) -> Dict[str, Any]:
        text = str(content or "").strip()
        try:
            return json.loads(text)
        except ValueError:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1))
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("no_json_object_found")

    def _http_error_text(self, exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8")
        except Exception:
            return str(exc)
