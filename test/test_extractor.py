# test/test_extractor.py
#
# Prototype test on 20 controlled papers across 4 categories.
# Run with: python -m test.test_extractor
#
# Does NOT write to Neo4j — pure extraction validation.
# Set ANTHROPIC_API_KEY in .env before running.

import os
import json

import together
from kg.nlp.extractor import extract, extract_layer1
from kg.utils.config import get_settings

# ─── Controlled Test Set ─────────────────────────────────────────────────────
#
# 20 papers across 4 categories.
# "expected" values are what we *want* to extract — used to score each result.
# None means "we don't know / don't care for this field in this test"

TEST_PAPERS = [

    # ── Category A: Famous papers with clear "we propose X" ──────────────────
    # Layer 1 regex should catch these. LLM should NOT be needed.

    {
        "arxiv_id": "2205.14135",
        "category": "A",
        "title": "FlashAttention",
        "abstract": (
            "We propose FlashAttention, an IO-aware exact attention algorithm "
            "that uses tiling to reduce the number of memory reads/writes between "
            "GPU high bandwidth memory and on-chip SRAM. We analyze the IO complexity "
            "of FlashAttention and prove that it requires fewer HBM accesses than "
            "standard attention and is optimal for a range of SRAM sizes. We also "
            "extend FlashAttention to block-sparse attention, yielding an approximate "
            "attention algorithm that is faster than any existing approximate attention "
            "method. FlashAttention trains Transformers faster than existing baselines: "
            "15% end-to-end wall-clock speedup on BERT-large compared to the MLPerf "
            "1.1 training speed record, 3x speedup on GPT-2."
        ),
        "expected": {
            "proposed_methods": ["FlashAttention"],
            "datasets": [],
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "2106.09685",
        "category": "A",
        "title": "LoRA",
        "abstract": (
            "We propose LoRA, Low-Rank Adaptation of Large Language Models. "
            "An important paradigm of natural language processing consists of large-scale "
            "pre-training on general domain data and adaptation to particular tasks or "
            "domains. As we pre-train larger models, full fine-tuning, which retrains "
            "all model parameters, becomes less feasible. We propose LoRA which freezes "
            "the pretrained model weights and injects trainable rank decomposition matrices "
            "into each layer of the Transformer architecture, greatly reducing the number "
            "of trainable parameters for downstream tasks. On RoBERTa, DeBERTa, GPT-2, "
            "and GPT-3, LoRA performs on-par or better than fine-tuning in model quality "
            "on RTE, SQuAD, WikiSQL, and SAMSum while being more task- and "
            "inference-efficient."
        ),
        "expected": {
            "proposed_methods": ["LoRA"],
            "datasets": ["SQuAD"],
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "2307.09288",
        "category": "A",
        "title": "LLaMA 2",
        "abstract": (
            "In this work, we develop and release Llama 2, a collection of pretrained "
            "and fine-tuned large language models (LLMs) ranging in scale from 7 billion "
            "to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are "
            "optimized for dialogue use cases. Our models outperform open-source chat "
            "models on most benchmarks we tested, and based on our human evaluations "
            "for helpfulness and safety, may be a suitable substitute for closed-source "
            "models. We provide a detailed description of our approach to fine-tuning "
            "and safety improvements of Llama 2-Chat in order to enable the community "
            "to build on our work and contribute to the responsible development of LLMs."
        ),
        "expected": {
            "proposed_methods": ["Llama 2"],
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "2305.18290",
        "category": "A",
        "title": "Direct Preference Optimization (DPO)",
        "abstract": (
            "We introduce Direct Preference Optimization (DPO), a new approach to "
            "fine-tuning large language models to align with human preferences. "
            "While existing methods such as RLHF require fitting a reward model "
            "and using reinforcement learning to optimize the language model, DPO "
            "optimizes directly on human preference data. DPO is stable, performant, "
            "and computationally lightweight, eliminating the need for sampling from "
            "the LM during fine-tuning or performing significant hyperparameter tuning. "
            "Our experiments show that DPO can fine-tune LMs to align with human "
            "preferences as well as or better than existing methods, including with "
            "the RLHF approach used to train GPT models, evaluated on tasks such "
            "as sentiment modulation, summarization, and dialogue."
        ),
        "expected": {
            "proposed_methods": ["DPO"],
            "used_methods_contains": ["RLHF"],
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "2312.00752",
        "category": "A",
        "title": "Mamba",
        "abstract": (
            "We introduce Mamba, a new state space model architecture showing promising "
            "performance on information-dense data such as language modeling, where previous "
            "subquadratic models fall short of Transformers. We propose a hardware-aware "
            "algorithm to compute the model recurrently with a scan instead of convolution, "
            "although the connection between the two views is maintained, allowing "
            "us to use convolutions for efficient parallelized training. Mamba enjoys "
            "fast inference (5x higher throughput than Transformers) and linear scaling "
            "in sequence length, and its performance improves on real data up to "
            "million-length sequences."
        ),
        "expected": {
            "proposed_methods": ["Mamba"],
            "used_methods_contains": ["Transformer"],
            "source": "rule_based",
        },
    },

    # ── Category B: Survey / benchmark papers — no new method proposed ────────
    # proposed_method should be null. LLM needed for most of these.

    {
        "arxiv_id": "2302.04023",
        "category": "B",
        "title": "A Survey on Efficient Training of Transformers",
        "abstract": (
            "Large transformer models have achieved remarkable results across many "
            "domains. However, training these models is extremely expensive, requiring "
            "massive amounts of computation. In this survey, we review methods for "
            "efficient training of transformer models, covering data efficiency, "
            "model efficiency, and system efficiency. We discuss techniques including "
            "mixed-precision training, gradient checkpointing, and pipeline parallelism. "
            "We also review benchmarks such as GLUE and SuperGLUE used to evaluate "
            "these methods. Our goal is to provide a comprehensive overview of the "
            "field and identify open problems."
        ),
        "expected": {
            "proposed_methods": [],   # survey, no proposal
            "datasets_contains": ["GLUE"],
            "source": "llm",          # layer 1 finds methods but no proposal → LLM
        },
    },

    {
        "arxiv_id": "2304.01373",
        "category": "B",
        "title": "Harnessing the Power of LLMs: A Survey",
        "abstract": (
            "This paper surveys the techniques used to build and deploy large language "
            "models (LLMs) including GPT-4, LLaMA, and PaLM. We review pre-training "
            "on large corpora, instruction tuning, reinforcement learning from human "
            "feedback, and prompting strategies. We evaluate these models on benchmarks "
            "including MMLU, BIG-Bench, and HellaSwag. We discuss alignment, safety, "
            "and the challenges of deploying these models at scale."
        ),
        "expected": {
            "proposed_methods": [],
            "datasets_contains": ["MMLU", "BIG-Bench"],
            "source": "llm",
        },
    },

    {
        "arxiv_id": "2206.07682",
        "category": "B",
        "title": "Emergent Abilities of Large Language Models",
        "abstract": (
            "Scaling up language models has been shown to predictably improve performance "
            "on a wide range of downstream tasks. This paper instead discusses emergent "
            "abilities of large language models: abilities that are not present in "
            "smaller-scale models but arise in larger-scale models. We evaluate "
            "language models including GPT-3, PaLM, and Gopher on benchmarks including "
            "BIG-Bench and MMLU to characterize when emergence occurs. We do not propose "
            "a new model but analyze scaling behavior systematically."
        ),
        "expected": {
            "proposed_methods": [],
            "datasets_contains": ["BIG-Bench", "MMLU"],
        },
    },

    {
        "arxiv_id": "2211.05100",
        "category": "B",
        "title": "Holistic Evaluation of Language Models (HELM)",
        "abstract": (
            "Language models (LMs) are becoming the foundation for almost all major "
            "language technologies, but their capabilities, limitations, and risks are "
            "poorly understood. We present Holistic Evaluation of Language Models (HELM), "
            "a living benchmark that aims to improve the transparency of language models. "
            "HELM evaluates 30 models across 42 scenarios and 7 metrics. "
            "We evaluate GPT-3, Codex, T5, and other models and find significant "
            "variation in performance across scenarios."
        ),
        "expected": {
            "proposed_methods": ["HELM"],   # HELM is a benchmark they introduce
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "2009.03300",
        "category": "B",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "abstract": (
            "We train GPT-3, an autoregressive language model with 175 billion parameters, "
            "and test its performance in the few-shot setting. For all tasks, GPT-3 is "
            "applied without any gradient updates or fine-tuning, with tasks and "
            "few-shot demonstrations specified purely via text interaction with the model. "
            "GPT-3 achieves strong performance on many NLP benchmarks including "
            "SuperGLUE, TriviaQA, and CoQA, sometimes matching fine-tuned models."
        ),
        "expected": {
            "proposed_methods": ["GPT-3"],
            "datasets_contains": ["SuperGLUE"],
        },
    },

    # ── Category C: Obscure / niche — method name not in vocabulary ───────────
    # Layer 1 finds nothing → LLM must extract the method name.

    {
        "arxiv_id": "2310.01889",
        "category": "C",
        "title": "Ring Attention",
        "abstract": (
            "We introduce Ring Attention with Blockwise Transformers, a method that "
            "enables training sequences that are magnitudes longer than those achievable "
            "by prior work. Our approach distributes the attention computation across "
            "multiple devices by decomposing it into blockwise computations and "
            "overlapping the communication of key-value blocks with the computation "
            "of blockwise attention. This approach allows leveraging the memory of "
            "many devices to accommodate lengthy sequences, with no communication "
            "overhead. We demonstrate training and inference of sequences up to "
            "hundreds of thousands of tokens long."
        ),
        "expected": {
            "proposed_methods_contains_any": ["Ring Attention"],
            "source": "llm",
        },
    },

    {
        "arxiv_id": "2309.06180",
        "category": "C",
        "title": "Efficient Memory Management for LLM Serving with PagedAttention",
        "abstract": (
            "High-throughput serving of large language models (LLMs) requires batching "
            "sufficiently many requests at a time. However, existing systems struggle "
            "because the key-value cache (KV cache) memory for each request is huge "
            "and grows and shrinks dynamically. We propose PagedAttention, an attention "
            "algorithm inspired by the classic virtual memory and paging techniques in "
            "operating systems. We build a high-throughput LLM serving engine vLLM "
            "on top of PagedAttention that achieves near-zero waste in KV cache memory. "
            "vLLM improves the throughput of popular LLMs by 2-4x compared to "
            "HuggingFace Transformers without affecting model accuracy."
        ),
        "expected": {
            "proposed_methods_contains_any": ["PagedAttention", "vLLM"],
            "source": "rule_based",  # "We propose PagedAttention" should match
        },
    },

    {
        "arxiv_id": "2305.10403",
        "category": "C",
        "title": "LIMA: Less Is More for Alignment",
        "abstract": (
            "Large language models are trained in two stages: unsupervised pretraining "
            "from raw text, followed by large scale instruction tuning. We introduce "
            "LIMA, a 65B parameter LLaMA language model fine-tuned with the standard "
            "supervised loss on only 1,000 carefully curated prompts and responses, "
            "without any reinforcement learning or human preference modeling. "
            "LIMA demonstrates remarkably strong performance, outperforming GPT-4 "
            "in 43% of cases in a human study, despite using far less training data. "
            "Our results suggest that almost all knowledge in large language models "
            "is learned during pretraining."
        ),
        "expected": {
            "proposed_methods_contains_any": ["LIMA"],
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "2310.06825",
        "category": "C",
        "title": "MistralAI Mistral 7B",
        "abstract": (
            "We introduce Mistral 7B, a 7-billion-parameter language model engineered "
            "for superior performance and efficiency. Mistral 7B outperforms the best "
            "open 13B model (Llama 2 13B) across all evaluated benchmarks, and the "
            "best released 34B model (Llama 2 34B) in reasoning, mathematics, and code "
            "generation. Our model leverages grouped-query attention for faster inference "
            "and sliding window attention to handle sequences of arbitrary length with "
            "a reduced inference cost. We also provide a model fine-tuned to follow "
            "instructions, Mistral 7B Instruct, that surpasses Llama 2 13B chat model "
            "both on human and automated benchmarks."
        ),
        "expected": {
            "proposed_methods_contains_any": ["Mistral 7B", "Mistral"],
            "used_methods_contains": ["Grouped Query Attention"],
        },
    },

    {
        "arxiv_id": "2307.15217",
        "category": "C",
        "title": "Scaling in a Sea of Compute: Chinchilla",
        "abstract": (
            "We investigate the optimal model size and number of tokens for training "
            "a transformer language model under a given compute budget. We find that "
            "current large language models are significantly undertrained. We introduce "
            "Chinchilla, a 70B parameter model trained on 1.4 trillion tokens, "
            "which significantly outperforms Gopher, GPT-3, Jurassic-1, and Megatron-Turing NLG "
            "on a large range of downstream evaluation tasks. Chinchilla uses the same "
            "compute budget as Gopher but with 4x more data. Our results have important "
            "implications for large model training."
        ),
        "expected": {
            "proposed_methods_contains_any": ["Chinchilla"],
            "source": "rule_based",
        },
    },

    # ── Category D: Edge cases — short, messy, or unusual abstracts ──────────
    # Tests robustness. Extractor should not crash. Partial results are fine.

    {
        "arxiv_id": "D_short_001",
        "category": "D",
        "title": "Very Short Abstract",
        "abstract": "We train a big model. It works well on ImageNet.",
        "expected": {
            "datasets_contains": ["ImageNet"],
            # proposed_methods: could be anything — we just check no crash
        },
    },

    {
        "arxiv_id": "D_nomethod_002",
        "category": "D",
        "title": "No Method Proposed",
        "abstract": (
            "In this position paper, we argue that the field of machine learning has "
            "focused too heavily on benchmark performance. We do not propose a new "
            "method or model. Instead we critically examine evaluation practices and "
            "suggest the community should prioritize reproducibility, generalization, "
            "and robustness over leaderboard rankings."
        ),
        "expected": {
            "proposed_methods": [],  # explicitly says "we do not propose"
        },
    },

    {
        "arxiv_id": "D_allcaps_003",
        "category": "D",
        "title": "Acronym heavy abstract",
        "abstract": (
            "We present MAGNET, a multi-agent graph neural network for efficient "
            "training on TPUs using FSDP and ZeRO optimization. MAGNET is evaluated "
            "on CIFAR-10 and achieves state of the art results. We compare against "
            "DDP and Megatron baselines."
        ),
        "expected": {
            "proposed_methods_contains_any": ["MAGNET"],
            "datasets_contains": ["CIFAR-10"],
            "used_methods_contains": ["ZeRO"],
        },
    },

    {
        "arxiv_id": "D_lowercase_004",
        "category": "D",
        "title": "Method name is lowercase",
        "abstract": (
            "We introduce mixtral, a sparse mixture of experts model that achieves "
            "strong performance on language benchmarks including MMLU and HellaSwag. "
            "mixtral uses a gating mechanism to route tokens to different expert "
            "sub-networks. We compare against llama 2 and gpt-3.5 baselines."
        ),
        "expected": {
            # lowercase name — regex may miss, LLM should catch
            "datasets_contains": ["MMLU"],
            "source": "rule_based",
        },
    },

    {
        "arxiv_id": "D_empty_005",
        "category": "D",
        "title": "Empty abstract (edge case)",
        "abstract": "",
        "expected": {
            "proposed_methods": [],
            "used_methods": [],
            "datasets": [],
            "tasks": [],
            # source is rule_based because layer1 runs and returns needs_llm=True
            # but with no client passed it falls back to rule_based empty result
        },
    },
]


# ─── Scoring Helpers ─────────────────────────────────────────────────────────

def _check_contains(actual: list, expected_contains: list, label: str) -> bool:
    """Check that ALL expected items appear somewhere in actual list (case-insensitive)."""
    actual_lower = [a.lower() for a in actual]
    missing = []
    for item in expected_contains:
        if not any(item.lower() in a for a in actual_lower):
            missing.append(item)
    if missing:
        print(f"    ⚠️  {label} missing: {missing}")
        return False
    return True


def _check_exact(actual: list, expected: list, label: str) -> bool:
    """Check that actual list matches expected exactly (order-insensitive)."""
    actual_set   = {a.lower() for a in actual}
    expected_set = {e.lower() for e in expected}
    if actual_set != expected_set:
        extra   = actual_set - expected_set
        missing = expected_set - actual_set
        if extra:   print(f"    ⚠️  {label} unexpected: {list(extra)}")
        if missing: print(f"    ⚠️  {label} missing: {list(missing)}")
        return False
    return True


# ─── Test Runner ─────────────────────────────────────────────────────────────

def run_tests():
    settings = get_settings()
    client = None
    if settings.together_api_key:
        client = together.Together(api_key=settings.together_api_key)
        print("✅ Together client initialised — LLM layer enabled")
    else:
        print("⚠️  No TOGETHER_API_KEY found — LLM layer disabled, layer 1 only")

    print()

    category_scores = {"A": [], "B": [], "C": [], "D": []}
    total_pass = 0
    total_fail = 0

    for paper in TEST_PAPERS:
        cat      = paper["category"]
        arxiv_id = paper["arxiv_id"]
        title    = paper["title"]
        expected = paper["expected"]

        # Run extraction
        result = extract(paper["abstract"], client)

        # ── Print paper header ──
        print(f"{'─'*60}")
        print(f"[{cat}] {title} ({arxiv_id})")
        print(f"  Source:          {result['source']}")
        print(f"  Proposed:        {result['proposed_methods']}")
        print(f"  Used methods:    {result['used_methods'][:4]}")  # truncate long lists
        print(f"  Datasets:        {result['datasets']}")
        print(f"  Tasks:           {result['tasks']}")
        print(f"  Improves on:     {result['improves_on']}")

        # ── Score against expected ──
        checks = []

        if "proposed_methods" in expected:
            checks.append(_check_exact(
                result["proposed_methods"],
                expected["proposed_methods"],
                "proposed_methods"
            ))

        if "proposed_methods_contains_any" in expected:
            found = any(
                any(e.lower() in p.lower() for p in result["proposed_methods"])
                for e in expected["proposed_methods_contains_any"]
            )
            if not found:
                print(f"    ⚠️  proposed_methods_contains_any: none of "
                      f"{expected['proposed_methods_contains_any']} found in "
                      f"{result['proposed_methods']}")
            checks.append(found)

        if "used_methods_contains" in expected:
            checks.append(_check_contains(
                result["used_methods"],
                expected["used_methods_contains"],
                "used_methods"
            ))

        if "datasets_contains" in expected:
            checks.append(_check_contains(
                result["datasets"],
                expected["datasets_contains"],
                "datasets"
            ))

        if "datasets" in expected:
            checks.append(_check_exact(
                result["datasets"],
                expected["datasets"],
                "datasets"
            ))

        if "source" in expected:
            match = result["source"] == expected["source"]
            if not match:
                print(f"    ⚠️  source: expected '{expected['source']}' "
                      f"got '{result['source']}'")
            checks.append(match)

        passed = all(checks) if checks else True
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Result:          {status}")

        category_scores[cat].append(passed)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")

    for cat, scores in category_scores.items():
        cat_labels = {
            "A": "Clear proposals (regex should catch)",
            "B": "Survey/benchmark (no new method)",
            "C": "Obscure names (LLM needed)",
            "D": "Edge cases (robustness)",
        }
        n_pass = sum(scores)
        n_total = len(scores)
        bar = "✅" * n_pass + "❌" * (n_total - n_pass)
        print(f"  Category {cat} ({cat_labels[cat]}): {bar} {n_pass}/{n_total}")

    print()
    print(f"  Total: {total_pass}/{total_pass + total_fail} passed")

    if total_fail == 0:
        print(f"\n✅ ALL TESTS PASSED")
    else:
        print(f"\n⚠️  {total_fail} test(s) failed — review output above")


if __name__ == "__main__":
    run_tests()