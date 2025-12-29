#!/usr/bin/env python3
"""
Prompt Optimizer - Automatic prompt improvement and A/B testing.

Features:
- Prompt template management
- A/B testing framework
- Automatic optimization
- Performance tracking

Usage:
    from prompt_optimizer import PromptOptimizer

    optimizer = PromptOptimizer(llm_client)
    best_prompt = optimizer.optimize(base_prompt, test_cases)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import json
import hashlib
import random
from datetime import datetime
import statistics


class OptimizationStrategy(Enum):
    """Strategies for prompt optimization."""
    CLARITY = "clarity"
    CONCISENESS = "conciseness"
    SPECIFICITY = "specificity"
    EXAMPLES = "examples"
    STRUCTURE = "structure"


@dataclass
class PromptVariant:
    """A prompt variant for testing."""
    id: str
    template: str
    strategy: OptimizationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


@dataclass
class TestCase:
    """Test case for prompt evaluation."""
    input: str
    expected_output: Optional[str] = None
    criteria: Optional[Dict[str, float]] = None  # {criterion: weight}


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt variant."""
    variant_id: str
    scores: Dict[str, float]
    latency_ms: float
    token_count: int
    output: str


@dataclass
class ABTestResult:
    """Result of A/B test between prompt variants."""
    winner: str
    confidence: float
    variant_stats: Dict[str, Dict[str, float]]
    sample_size: int


class PromptTemplate:
    """Dynamic prompt template with variable substitution."""

    def __init__(self, template: str):
        self.template = template
        self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        import re
        return re.findall(r'\{(\w+)\}', self.template)

    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f'{{{var}}}', str(kwargs[var]))
        return result

    def validate(self, **kwargs) -> bool:
        """Check if all required variables are provided."""
        return all(var in kwargs for var in self.variables)


class PromptEvaluator:
    """Evaluate prompt quality using various metrics."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.criteria = {
            "relevance": self._score_relevance,
            "clarity": self._score_clarity,
            "completeness": self._score_completeness,
            "accuracy": self._score_accuracy
        }

    def evaluate(
        self,
        prompt: str,
        test_case: TestCase,
        criteria: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Evaluate a prompt on a test case."""
        import time

        start = time.time()
        response = self.llm.generate(prompt.replace("{input}", test_case.input))
        latency = (time.time() - start) * 1000

        scores = {}
        active_criteria = criteria or list(self.criteria.keys())

        for criterion in active_criteria:
            if criterion in self.criteria:
                scores[criterion] = self.criteria[criterion](
                    prompt, test_case, response.text
                )

        return EvaluationResult(
            variant_id=hashlib.md5(prompt.encode()).hexdigest()[:8],
            scores=scores,
            latency_ms=latency,
            token_count=response.usage.get("total_tokens", 0),
            output=response.text
        )

    def _score_relevance(self, prompt: str, test_case: TestCase, output: str) -> float:
        """Score how relevant the output is to the input."""
        eval_prompt = f"""Rate the relevance of this response to the input on a scale of 0-1.

Input: {test_case.input}
Response: {output}

Just respond with a number between 0 and 1:"""

        result = self.llm.generate(eval_prompt)
        try:
            return float(result.text.strip())
        except ValueError:
            return 0.5

    def _score_clarity(self, prompt: str, test_case: TestCase, output: str) -> float:
        """Score how clear and understandable the output is."""
        eval_prompt = f"""Rate the clarity of this text on a scale of 0-1.
Consider: readability, structure, and coherence.

Text: {output}

Just respond with a number between 0 and 1:"""

        result = self.llm.generate(eval_prompt)
        try:
            return float(result.text.strip())
        except ValueError:
            return 0.5

    def _score_completeness(self, prompt: str, test_case: TestCase, output: str) -> float:
        """Score if the output fully addresses the input."""
        if not test_case.expected_output:
            return 0.5

        eval_prompt = f"""Compare the expected and actual outputs.
Rate completeness on a scale of 0-1.

Expected: {test_case.expected_output}
Actual: {output}

Just respond with a number between 0 and 1:"""

        result = self.llm.generate(eval_prompt)
        try:
            return float(result.text.strip())
        except ValueError:
            return 0.5

    def _score_accuracy(self, prompt: str, test_case: TestCase, output: str) -> float:
        """Score factual accuracy of the output."""
        if not test_case.expected_output:
            return 0.5

        eval_prompt = f"""Compare accuracy between expected and actual.
Rate on a scale of 0-1.

Expected: {test_case.expected_output}
Actual: {output}

Just respond with a number between 0 and 1:"""

        result = self.llm.generate(eval_prompt)
        try:
            return float(result.text.strip())
        except ValueError:
            return 0.5


class PromptOptimizer:
    """Automatic prompt optimization through systematic testing."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.evaluator = PromptEvaluator(llm_client)
        self.history: List[Dict] = []

    def generate_variants(
        self,
        base_prompt: str,
        strategies: Optional[List[OptimizationStrategy]] = None
    ) -> List[PromptVariant]:
        """Generate prompt variants using different strategies."""
        strategies = strategies or list(OptimizationStrategy)
        variants = [
            PromptVariant(
                id="original",
                template=base_prompt,
                strategy=OptimizationStrategy.CLARITY
            )
        ]

        for strategy in strategies:
            improved = self._apply_strategy(base_prompt, strategy)
            variants.append(PromptVariant(
                id=f"{strategy.value}_{hashlib.md5(improved.encode()).hexdigest()[:6]}",
                template=improved,
                strategy=strategy
            ))

        return variants

    def _apply_strategy(self, prompt: str, strategy: OptimizationStrategy) -> str:
        """Apply an optimization strategy to a prompt."""
        strategy_prompts = {
            OptimizationStrategy.CLARITY: """Improve this prompt for clarity:
{prompt}

Make instructions clearer and more explicit. Keep the same intent.
Improved prompt:""",

            OptimizationStrategy.CONCISENESS: """Make this prompt more concise:
{prompt}

Remove redundancy while keeping all essential instructions.
Concise prompt:""",

            OptimizationStrategy.SPECIFICITY: """Add more specificity to this prompt:
{prompt}

Add constraints, examples, or format requirements.
Specific prompt:""",

            OptimizationStrategy.EXAMPLES: """Add examples to this prompt:
{prompt}

Include 1-2 examples to demonstrate expected output.
Prompt with examples:""",

            OptimizationStrategy.STRUCTURE: """Improve the structure of this prompt:
{prompt}

Use clear sections, bullet points, or numbered steps.
Structured prompt:"""
        }

        meta_prompt = strategy_prompts[strategy].format(prompt=prompt)
        result = self.llm.generate(meta_prompt)
        return result.text.strip()

    def run_ab_test(
        self,
        variants: List[PromptVariant],
        test_cases: List[TestCase],
        criteria: Optional[List[str]] = None
    ) -> ABTestResult:
        """Run A/B test between prompt variants."""
        results: Dict[str, List[EvaluationResult]] = {v.id: [] for v in variants}

        for test_case in test_cases:
            for variant in variants:
                eval_result = self.evaluator.evaluate(
                    variant.template,
                    test_case,
                    criteria
                )
                eval_result.variant_id = variant.id
                results[variant.id].append(eval_result)

        # Aggregate results
        variant_stats = {}
        for variant_id, evals in results.items():
            all_scores = [sum(e.scores.values()) / len(e.scores) for e in evals]
            variant_stats[variant_id] = {
                "mean_score": statistics.mean(all_scores),
                "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                "mean_latency": statistics.mean([e.latency_ms for e in evals]),
                "mean_tokens": statistics.mean([e.token_count for e in evals])
            }

        # Find winner
        winner = max(variant_stats.items(), key=lambda x: x[1]["mean_score"])

        # Calculate confidence (simplified)
        scores = [s["mean_score"] for s in variant_stats.values()]
        if len(scores) > 1:
            best = max(scores)
            second = sorted(scores)[-2]
            confidence = min((best - second) / (best + 0.001), 1.0)
        else:
            confidence = 1.0

        return ABTestResult(
            winner=winner[0],
            confidence=confidence,
            variant_stats=variant_stats,
            sample_size=len(test_cases)
        )

    def optimize(
        self,
        base_prompt: str,
        test_cases: List[TestCase],
        max_iterations: int = 3,
        strategies: Optional[List[OptimizationStrategy]] = None
    ) -> str:
        """Iteratively optimize a prompt."""
        current_best = base_prompt
        best_score = 0.0

        for iteration in range(max_iterations):
            variants = self.generate_variants(current_best, strategies)
            result = self.run_ab_test(variants, test_cases)

            winner_stats = result.variant_stats[result.winner]

            if winner_stats["mean_score"] > best_score:
                best_score = winner_stats["mean_score"]
                current_best = next(
                    v.template for v in variants if v.id == result.winner
                )

                self.history.append({
                    "iteration": iteration,
                    "winner": result.winner,
                    "score": best_score,
                    "confidence": result.confidence,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                break  # No improvement, stop

        return current_best

    def get_optimization_report(self) -> Dict:
        """Get report of optimization history."""
        if not self.history:
            return {"message": "No optimization history"}

        return {
            "iterations": len(self.history),
            "initial_score": self.history[0]["score"] if self.history else 0,
            "final_score": self.history[-1]["score"] if self.history else 0,
            "improvement": (
                (self.history[-1]["score"] - self.history[0]["score"])
                / self.history[0]["score"] * 100
            ) if self.history and self.history[0]["score"] > 0 else 0,
            "history": self.history
        }


# Convenience functions
def quick_optimize(llm_client, prompt: str, test_inputs: List[str]) -> str:
    """Quick optimization with minimal setup."""
    optimizer = PromptOptimizer(llm_client)
    test_cases = [TestCase(input=inp) for inp in test_inputs]
    return optimizer.optimize(prompt, test_cases, max_iterations=2)


# Example usage
if __name__ == "__main__":
    # Mock LLM client for demonstration
    class MockLLM:
        def generate(self, prompt):
            from dataclasses import dataclass

            @dataclass
            class Response:
                text: str = "0.8"
                usage: dict = field(default_factory=lambda: {"total_tokens": 100})

            return Response()

    llm = MockLLM()
    optimizer = PromptOptimizer(llm)

    base_prompt = "Summarize the following text: {input}"
    test_cases = [
        TestCase(input="Long article about AI...", expected_output="AI summary"),
        TestCase(input="Technical documentation...", expected_output="Doc summary")
    ]

    optimized = optimizer.optimize(base_prompt, test_cases)
    print(f"Optimized prompt: {optimized}")
    print(f"Report: {optimizer.get_optimization_report()}")
