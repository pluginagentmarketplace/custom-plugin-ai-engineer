#!/usr/bin/env python3
"""
LLM Evaluator - Comprehensive evaluation framework for language models.

Supports:
- Text quality metrics (BLEU, ROUGE, BERTScore)
- RAG-specific metrics (faithfulness, relevancy)
- Hallucination detection
- A/B testing

Usage:
    from llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator()
    results = evaluator.evaluate(predictions, references)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    input: str
    prediction: str
    reference: Optional[str] = None
    context: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetricResult:
    """Result from a single metric."""
    name: str
    score: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    metrics: Dict[str, float]
    sample_results: List[Dict[str, Any]]
    summary: Dict[str, Any]


class BaseMetric(ABC):
    """Abstract base class for metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> float:
        pass


class BLEUMetric(BaseMetric):
    """BLEU score for translation/generation."""

    @property
    def name(self) -> str:
        return "bleu"

    def compute(self, predictions: List[str], references: List[str]) -> float:
        from evaluate import load
        bleu = load("bleu")
        result = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        return result["bleu"]


class ROUGEMetric(BaseMetric):
    """ROUGE score for summarization."""

    @property
    def name(self) -> str:
        return "rouge"

    def compute(self, predictions: List[str], references: List[str]) -> float:
        from evaluate import load
        rouge = load("rouge")
        result = rouge.compute(predictions=predictions, references=references)
        return result["rougeL"]


class BERTScoreMetric(BaseMetric):
    """BERTScore for semantic similarity."""

    def __init__(self, model: str = "microsoft/deberta-xlarge-mnli"):
        self.model = model

    @property
    def name(self) -> str:
        return "bertscore"

    def compute(self, predictions: List[str], references: List[str]) -> float:
        from evaluate import load
        bertscore = load("bertscore")
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type=self.model
        )
        return np.mean(result["f1"])


class FaithfulnessMetric(BaseMetric):
    """Faithfulness metric for RAG systems."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(
        self,
        predictions: List[str],
        references: List[str] = None,
        contexts: List[List[str]] = None
    ) -> float:
        if not contexts:
            return 0.0

        scores = []
        for pred, ctx in zip(predictions, contexts):
            score = self._check_faithfulness(pred, ctx)
            scores.append(score)

        return np.mean(scores)

    def _check_faithfulness(self, prediction: str, context: List[str]) -> float:
        """Check if prediction is grounded in context."""
        if not self.llm:
            # Fallback: simple overlap check
            context_text = " ".join(context).lower()
            pred_words = set(prediction.lower().split())
            context_words = set(context_text.split())
            overlap = len(pred_words & context_words) / len(pred_words)
            return overlap

        prompt = f"""Determine if the answer is fully supported by the context.

Context: {' '.join(context)}

Answer: {prediction}

Is the answer fully supported by the context? Rate from 0 (not at all) to 1 (fully supported).
Just respond with a number:"""

        response = self.llm.generate(prompt)
        try:
            return float(response.text.strip())
        except ValueError:
            return 0.5


class HallucinationDetector:
    """Detect hallucinations in LLM outputs."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def detect(
        self,
        prediction: str,
        context: Optional[List[str]] = None,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect potential hallucinations."""
        results = {
            "has_hallucination": False,
            "confidence": 0.0,
            "evidence": []
        }

        # Check against context
        if context:
            context_score = self._check_context_support(prediction, context)
            if context_score < 0.5:
                results["has_hallucination"] = True
                results["evidence"].append("Not supported by context")
                results["confidence"] = 1 - context_score

        # Check self-consistency
        if self.llm:
            consistency = self._check_self_consistency(prediction)
            if consistency < 0.7:
                results["has_hallucination"] = True
                results["evidence"].append("Inconsistent regenerations")
                results["confidence"] = max(results["confidence"], 1 - consistency)

        return results

    def _check_context_support(self, prediction: str, context: List[str]) -> float:
        """Check if prediction is supported by context."""
        context_text = " ".join(context).lower()
        pred_sentences = prediction.split(".")

        supported = 0
        for sent in pred_sentences:
            if sent.strip():
                # Simple word overlap
                sent_words = set(sent.lower().split())
                context_words = set(context_text.split())
                if len(sent_words & context_words) / max(len(sent_words), 1) > 0.3:
                    supported += 1

        return supported / max(len(pred_sentences), 1)

    def _check_self_consistency(self, claim: str, n: int = 3) -> float:
        """Check consistency across multiple regenerations."""
        if not self.llm:
            return 1.0

        prompt = f"Verify this claim: {claim}\n\nIs this accurate? Yes or No:"
        responses = []

        for _ in range(n):
            response = self.llm.generate(prompt, temperature=0.7)
            responses.append(response.text.lower())

        # Check if all responses agree
        yes_count = sum(1 for r in responses if "yes" in r)
        return yes_count / n


class LLMEvaluator:
    """Main evaluator class."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.metrics: Dict[str, BaseMetric] = {}
        self.hallucination_detector = HallucinationDetector(llm_client)

        # Register default metrics
        self.register_metric(BLEUMetric())
        self.register_metric(ROUGEMetric())
        self.register_metric(BERTScoreMetric())
        self.register_metric(FaithfulnessMetric(llm_client))

    def register_metric(self, metric: BaseMetric) -> None:
        """Register a new metric."""
        self.metrics[metric.name] = metric

    def evaluate(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Run evaluation on samples."""
        active_metrics = metrics or list(self.metrics.keys())

        predictions = [s.prediction for s in samples]
        references = [s.reference for s in samples if s.reference]
        contexts = [s.context for s in samples if s.context]

        results = {}
        sample_results = []

        # Compute each metric
        for metric_name in active_metrics:
            if metric_name not in self.metrics:
                continue

            metric = self.metrics[metric_name]

            try:
                if metric_name == "faithfulness" and contexts:
                    score = metric.compute(predictions, references, contexts)
                elif references:
                    score = metric.compute(predictions, references)
                else:
                    continue

                results[metric_name] = score
            except Exception as e:
                results[metric_name] = None
                print(f"Error computing {metric_name}: {e}")

        # Per-sample analysis
        for sample in samples:
            sample_result = {
                "input": sample.input,
                "prediction": sample.prediction[:100] + "..."
            }

            # Check for hallucinations
            hallucination = self.hallucination_detector.detect(
                sample.prediction,
                sample.context,
                sample.reference
            )
            sample_result["hallucination"] = hallucination

            sample_results.append(sample_result)

        # Summary statistics
        summary = {
            "total_samples": len(samples),
            "metrics_computed": len(results),
            "average_score": np.mean([v for v in results.values() if v]),
            "hallucination_rate": np.mean([
                1 if r["hallucination"]["has_hallucination"] else 0
                for r in sample_results
            ])
        }

        return EvaluationResult(
            metrics=results,
            sample_results=sample_results,
            summary=summary
        )

    def run_ab_test(
        self,
        samples: List[EvaluationSample],
        model_a_predictions: List[str],
        model_b_predictions: List[str],
        metric: str = "bertscore"
    ) -> Dict[str, Any]:
        """Run A/B test between two models."""
        samples_a = [
            EvaluationSample(
                input=s.input,
                prediction=p,
                reference=s.reference,
                context=s.context
            )
            for s, p in zip(samples, model_a_predictions)
        ]

        samples_b = [
            EvaluationSample(
                input=s.input,
                prediction=p,
                reference=s.reference,
                context=s.context
            )
            for s, p in zip(samples, model_b_predictions)
        ]

        results_a = self.evaluate(samples_a, [metric])
        results_b = self.evaluate(samples_b, [metric])

        score_a = results_a.metrics.get(metric, 0)
        score_b = results_b.metrics.get(metric, 0)

        # Statistical test
        from scipy import stats
        # Would need per-sample scores for proper t-test
        improvement = (score_b - score_a) / max(score_a, 0.001) * 100

        return {
            "model_a_score": score_a,
            "model_b_score": score_b,
            "improvement_percent": improvement,
            "winner": "B" if score_b > score_a else "A",
            "metric": metric
        }


# Convenience function
def quick_evaluate(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ["bleu", "rouge"]
) -> Dict[str, float]:
    """Quick evaluation without full sample setup."""
    evaluator = LLMEvaluator()

    samples = [
        EvaluationSample(input="", prediction=p, reference=r)
        for p, r in zip(predictions, references)
    ]

    result = evaluator.evaluate(samples, metrics)
    return result.metrics


if __name__ == "__main__":
    # Demo
    predictions = [
        "The capital of France is Paris.",
        "Machine learning is a subset of AI."
    ]
    references = [
        "Paris is the capital of France.",
        "Machine learning is a type of artificial intelligence."
    ]

    print("Quick Evaluation:")
    results = quick_evaluate(predictions, references)
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
