"""Unit test execution scorer for generated Python functions with sandboxing."""

import re
import multiprocessing
from typing import Any, Dict, List, Optional
from src.scorers.base import BaseScorer, EvaluationResult


def _worker_exec_tests(code: str, fn_name: str, test_cases: List[Dict[str, Any]], return_dict: dict):
    """Worker function executed in separate process to enforce timeout and isolation."""
    try:
        # Provide standard safe builtins (including __import__ for standard stdlib imports like collections)
        safe_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else dict(__builtins__.__dict__)
        # Block dangerous functions
        for dangerous in ("eval", "exec", "open", "input", "compile", "breakpoint", "exit", "quit"):
            safe_builtins.pop(dangerous, None)

        namespace = {"__builtins__": safe_builtins}
        # Pre-import common stdlib modules
        import collections, math, itertools, functools, heapq, bisect, re
        namespace.update({
            "collections": collections,
            "OrderedDict": collections.OrderedDict,
            "defaultdict": collections.defaultdict,
            "deque": collections.deque,
            "Counter": collections.Counter,
            "math": math,
            "itertools": itertools,
            "functools": functools,
            "heapq": heapq,
            "bisect": bisect,
            "re": re,
        })

        exec(code, namespace, namespace)
        if fn_name not in namespace:
            return_dict["error"] = f"Function '{fn_name}' was not defined in generated code."
            return_dict["passed"] = False
            return_dict["passed_count"] = 0
            return_dict["total_count"] = len(test_cases)
            return

        target_fn = namespace[fn_name]
        passed = 0
        details = []

        for idx, tc in enumerate(test_cases):
            args = tc.get("args", [])
            expected = tc.get("expected")
            try:
                # Deepcopy args if mutable to prevent test pollution
                import copy
                call_args = copy.deepcopy(args)
                result = target_fn(*call_args)
                if result == expected:
                    passed += 1
                else:
                    details.append(f"Test {idx} failed: {fn_name}({args}) returned {result!r}, expected {expected!r}")
            except Exception as test_err:
                details.append(f"Test {idx} raised error: {type(test_err).__name__}: {test_err}")

        return_dict["passed"] = passed == len(test_cases)
        return_dict["passed_count"] = passed
        return_dict["total_count"] = len(test_cases)
        return_dict["details"] = details
        return_dict["error"] = None
    except Exception as err:
        return_dict["error"] = f"Code execution failed: {type(err).__name__}: {err}"
        return_dict["passed"] = False
        return_dict["passed_count"] = 0
        return_dict["total_count"] = len(test_cases)


class UnitTestScorer(BaseScorer):
    """Executes unit tests against Python function implementations in completions."""

    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout_seconds = timeout_seconds

    def _extract_code(self, text: str) -> str:
        trimmed = text.strip()
        # Look for markdown code fence
        fences = re.findall(r"```(?:python)?\s*\n?(.*?)\n?```", trimmed, re.DOTALL)
        if fences:
            return fences[-1].strip()
        return trimmed

    def score(self, completion: str, expected: Any, metadata: Optional[dict] = None) -> EvaluationResult:
        metadata = metadata or {}
        fn_name = metadata.get("function_name", "")
        test_cases = metadata.get("test_cases", [])

        if not fn_name or not test_cases:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value=expected,
                details=f"Missing function_name or test_cases in metadata (got fn={fn_name}, tests={len(test_cases)})",
            )

        code = self._extract_code(completion)
        if not code:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value="",
                expected_value="pass_all_unit_tests",
                details="No executable code block or content found in completion.",
            )

        # Run with timeout using multiprocessing
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        proc = multiprocessing.Process(
            target=_worker_exec_tests,
            args=(code, fn_name, test_cases, return_dict),
        )
        proc.start()
        proc.join(timeout=self.timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(0.5)
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value="pass_all_unit_tests",
                details=f"Execution timed out (> {self.timeout_seconds}s). Possible infinite loop or recursion.",
                metrics={"timed_out": True},
            )

        err = return_dict.get("error")
        if err:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value="pass_all_unit_tests",
                details=err,
                metrics={"syntax_or_runtime_error": True},
            )

        passed = return_dict.get("passed", False)
        passed_count = return_dict.get("passed_count", 0)
        total_count = return_dict.get("total_count", len(test_cases))
        details_list = return_dict.get("details", [])
        score = passed_count / total_count if total_count > 0 else 0.0

        details = f"Passed {passed_count}/{total_count} unit tests."
        if details_list:
            details += " Failures: " + "; ".join(details_list[:3])

        return EvaluationResult(
            passed=passed,
            score=score,
            extracted_value={"passed_tests": passed_count, "total_tests": total_count},
            expected_value="pass_all_unit_tests",
            details=details,
            metrics={"passed_count": passed_count, "total_count": total_count, "score": score},
        )
