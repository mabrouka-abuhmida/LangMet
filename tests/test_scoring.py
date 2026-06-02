from langmet.models import RagaEvaluationEvent
from langmet.scoring import RagaScorer, TokenOverlapScorer, score_query


def test_token_overlap_scorer_satisfies_protocol():
    assert isinstance(TokenOverlapScorer(), RagaScorer)


def test_score_query_with_ground_truth():
    event = score_query(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital and largest city of France."],
        ground_truth="Paris is the capital of France.",
        query_id="q1",
    )
    assert isinstance(event, RagaEvaluationEvent)
    assert event.query_id == "q1"
    assert event.faithfulness is not None
    assert event.answer_relevancy is not None
    assert event.context_precision is not None
    assert event.answer_correctness == 1.0


def test_score_query_without_ground_truth():
    event = score_query(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital of France."],
    )
    # Reference-dependent metrics are None without ground truth
    assert event.context_precision is None
    assert event.context_recall is None
    assert event.answer_correctness is None
    assert event.answer_similarity is None
    # Reference-free metrics are still computed
    assert event.faithfulness is not None
    assert event.answer_relevancy is not None
    assert event.context_relevancy is not None


def test_custom_scorer_used():
    class ConstantScorer:
        def faithfulness(self, answer, contexts):
            return 0.5

        def answer_relevancy(self, question, answer):
            return 0.5

        def context_precision(self, contexts, ground_truth):
            return 0.5

        def context_recall(self, contexts, ground_truth):
            return 0.5

        def context_relevancy(self, question, contexts):
            return 0.5

        def answer_correctness(self, answer, ground_truth):
            return 0.5

        def answer_similarity(self, answer, ground_truth):
            return 0.5

    event = score_query(
        question="q",
        answer="a",
        contexts=["c"],
        ground_truth="g",
        scorer=ConstantScorer(),
    )
    assert event.faithfulness == 0.5
    assert event.answer_correctness == 0.5
