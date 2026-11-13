"""Constraint satisfaction and instruction following compliance checker."""

import re
from typing import Any, Dict, List, Optional
from src.scorers.base import BaseScorer, EvaluationResult


class ConstraintCheckScorer(BaseScorer):
    """Evaluates multi-constraint prompt following requirements deterministically.

    Supports:
    - sentence_count
    - all_lowercase
    - forbidden_chars
    - line_count
    - each_line_regex
    - line_regex
    - word_count
    - each_line_word_count
    - line_word_counts
    - contains_all
    - contains_none
    - ends_with
    - acrostic
    - sentence_word_counts
    - sentence_starts_with
    - sentence_ends_with
    - max_sentence_words
    - paragraph_count
    - paragraph_sentence_counts
    - paragraph_first_letters
    """

    def _split_sentences(self, text: str) -> List[str]:
        # Clean text
        trimmed = text.strip()
        # Split on sentence terminals .!? followed by space or end of string
        parts = re.split(r'(?<=[.!?])\s+', trimmed)
        return [p.strip() for p in parts if p.strip()]

    def _split_lines(self, text: str) -> List[str]:
        return text.splitlines()

    def _split_paragraphs(self, text: str) -> List[str]:
        paras = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paras if p.strip()]

    def _check_constraint(self, c_type: str, params: Dict[str, Any], text: str) -> tuple[bool, str]:
        trimmed = text.strip()

        if c_type == "sentence_count":
            target = params["n"]
            sents = self._split_sentences(trimmed)
            passed = len(sents) == target
            return passed, f"sentence_count: expected {target}, got {len(sents)}"

        if c_type == "all_lowercase":
            # letters must be lowercase (no uppercase)
            has_upper = any(ch.isupper() for ch in trimmed)
            return not has_upper, "all_lowercase: contains uppercase letters" if has_upper else "all_lowercase: satisfied"

        if c_type == "forbidden_chars":
            chars = params.get("chars", [])
            found = [ch for ch in chars if ch in trimmed]
            passed = len(found) == 0
            return passed, f"forbidden_chars: found forbidden {found}" if not passed else "forbidden_chars: satisfied"

        if c_type == "line_count":
            target = params["n"]
            lines = self._split_lines(trimmed)
            passed = len(lines) == target
            return passed, f"line_count: expected {target}, got {len(lines)}"

        if c_type == "each_line_regex":
            pattern = re.compile(params["pattern"])
            lines = self._split_lines(trimmed)
            for idx, line in enumerate(lines):
                if not pattern.search(line):
                    return False, f"each_line_regex: line {idx} '{line}' did not match pattern '{params['pattern']}'"
            return True, "each_line_regex: all lines matched"

        if c_type == "line_regex":
            idx = params["index"]
            pattern = re.compile(params["pattern"])
            lines = self._split_lines(trimmed)
            if idx < 0:
                idx = len(lines) + idx
            if idx < 0 or idx >= len(lines):
                return False, f"line_regex: index {idx} out of range (line count {len(lines)})"
            if not pattern.search(lines[idx]):
                return False, f"line_regex: line {idx} '{lines[idx]}' failed pattern '{params['pattern']}'"
            return True, f"line_regex: line {idx} matched"

        if c_type == "word_count":
            words = trimmed.split()
            wlen = len(words)
            min_w = params.get("min", 0)
            max_w = params.get("max", 1000000)
            passed = min_w <= wlen <= max_w
            return passed, f"word_count: expected {min_w}-{max_w}, got {wlen}"

        if c_type == "each_line_word_count":
            min_w = params.get("min", 0)
            max_w = params.get("max", 1000000)
            lines = self._split_lines(trimmed)
            for idx, line in enumerate(lines):
                wlen = len(line.split())
                if not (min_w <= wlen <= max_w):
                    return False, f"each_line_word_count: line {idx} had {wlen} words (expected {min_w}-{max_w})"
            return True, "each_line_word_count: satisfied"

        if c_type == "line_word_counts":
            expected_counts = params["counts"]
            lines = self._split_lines(trimmed)
            actual_counts = [len(l.split()) for l in lines]
            passed = actual_counts == expected_counts
            return passed, f"line_word_counts: expected {expected_counts}, got {actual_counts}"

        if c_type == "contains_all":
            items = params.get("items", [])
            missing = [item for item in items if item.lower() not in trimmed.lower()]
            passed = len(missing) == 0
            return passed, f"contains_all: missing {missing}" if not passed else "contains_all: satisfied"

        if c_type == "contains_none":
            items = params.get("items", [])
            found = []
            for item in items:
                # Use word boundary check for word tokens
                if re.search(r'\b' + re.escape(item) + r'\b', trimmed, re.IGNORECASE):
                    found.append(item)
            passed = len(found) == 0
            return passed, f"contains_none: found forbidden {found}" if not passed else "contains_none: satisfied"

        if c_type == "ends_with":
            target = params["text"]
            passed = trimmed.endswith(target)
            return passed, f"ends_with: expected ending '{target}'" if not passed else "ends_with: satisfied"

        if c_type == "acrostic":
            word = params["word"].upper()
            lines = self._split_lines(trimmed)
            if len(lines) != len(word):
                return False, f"acrostic: line count {len(lines)} != acrostic length {len(word)}"
            first_letters = "".join(l.strip()[0].upper() if l.strip() else "" for l in lines)
            passed = first_letters == word
            return passed, f"acrostic: expected {word}, got {first_letters}"

        if c_type == "sentence_word_counts":
            expected_counts = params["counts"]
            sents = self._split_sentences(trimmed)
            actual_counts = [len(s.split()) for s in sents]
            passed = actual_counts == expected_counts
            return passed, f"sentence_word_counts: expected {expected_counts}, got {actual_counts}"

        if c_type == "sentence_starts_with":
            idx = params.get("index", 0)
            text_target = params["text"]
            sents = self._split_sentences(trimmed)
            if idx < 0:
                idx = len(sents) + idx
            if idx < 0 or idx >= len(sents):
                return False, f"sentence_starts_with: sentence index {idx} out of range"
            first_word = sents[idx].split()[0] if sents[idx].split() else ""
            passed = first_word.lower() == text_target.lower()
            return passed, f"sentence_starts_with: sentence {idx} starts with '{first_word}', expected '{text_target}'"

        if c_type == "sentence_ends_with":
            idx = params.get("index", -1)
            char_target = params["char"]
            sents = self._split_sentences(trimmed)
            if idx < 0:
                idx = len(sents) + idx
            if idx < 0 or idx >= len(sents):
                return False, f"sentence_ends_with: sentence index {idx} out of range"
            sent = sents[idx].strip()
            passed = sent.endswith(char_target)
            return passed, f"sentence_ends_with: sentence {idx} ends with '{sent[-1] if sent else ''}', expected '{char_target}'"

        if c_type == "max_sentence_words":
            max_n = params["n"]
            sents = self._split_sentences(trimmed)
            for idx, s in enumerate(sents):
                wlen = len(s.split())
                if wlen > max_n:
                    return False, f"max_sentence_words: sentence {idx} has {wlen} words (exceeds {max_n})"
            return True, "max_sentence_words: satisfied"

        if c_type == "paragraph_count":
            target = params["n"]
            paras = self._split_paragraphs(trimmed)
            passed = len(paras) == target
            return passed, f"paragraph_count: expected {target}, got {len(paras)}"

        if c_type == "paragraph_sentence_counts":
            expected_counts = params["counts"]
            paras = self._split_paragraphs(trimmed)
            actual_counts = [len(self._split_sentences(p)) for p in paras]
            passed = actual_counts == expected_counts
            return passed, f"paragraph_sentence_counts: expected {expected_counts}, got {actual_counts}"

        if c_type == "paragraph_first_letters":
            expected_letters = [l.upper() for l in params["letters"]]
            paras = self._split_paragraphs(trimmed)
            actual_letters = []
            for p in paras:
                words = p.split()
                if words and words[0]:
                    actual_letters.append(words[0][0].upper())
                else:
                    actual_letters.append("")
            passed = actual_letters == expected_letters
            return passed, f"paragraph_first_letters: expected {expected_letters}, got {actual_letters}"

        return False, f"Unknown constraint type: {c_type}"

    def score(self, completion: str, expected: Any, metadata: Optional[dict] = None) -> EvaluationResult:
        metadata = metadata or {}
        constraints = metadata.get("constraints", [])
        if not constraints and isinstance(expected, dict) and "constraints" in expected:
            constraints = expected["constraints"]

        if not constraints:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value=expected,
                details="No constraints specified in metadata or expected value.",
            )

        results = []
        passed_count = 0
        details_list = []

        for c in constraints:
            c_type = c.get("type", "")
            params = c.get("params", {})
            c_id = c.get("id", c_type)
            ok, msg = self._check_constraint(c_type, params, completion)
            results.append({"id": c_id, "type": c_type, "passed": ok, "details": msg})
            if ok:
                passed_count += 1
            else:
                details_list.append(f"[{c_id}] {msg}")

        all_passed = passed_count == len(constraints)
        score = passed_count / len(constraints) if constraints else 0.0
        details = "All constraints satisfied." if all_passed else "; ".join(details_list)

        return EvaluationResult(
            passed=all_passed,
            score=score,
            extracted_value={"passed_constraints": passed_count, "total_constraints": len(constraints)},
            expected_value="all_constraints_satisfied",
            details=details,
            metrics={"constraint_results": results, "passed_ratio": score},
        )
