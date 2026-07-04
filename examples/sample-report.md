# Distillation Report — runs/mac-small

*Generated 2026-07-04 by Distill Anything*

> ✅ Eval prompts are disjoint from the training data (0/24 overlap with `data/train.jsonl`).

## Quality (LLM-as-judge)

**The student matches or beats the reference on 79% of 24 held-out prompts.**

Judge: `HuggingFaceTB/SmolLM2-360M-Instruct`, blind A/B with position swap.

| Student wins | Ties | Reference wins |
|:---:|:---:|:---:|
| 2 (8%) | 17 (71%) | 5 (21%) |

## Training metrics

- **eval_loss**: 0.799
- **perplexity**: 2.223
- **teacher_agreement**: 0.8162

## Efficiency

| Metric | Student | Teacher (HuggingFaceTB/SmolLM2-360M-Instruct) |
|---|---:|---:|
| Parameters (M) | 134.5 | 361.8 |
| Tokens / s | 69.5 | 53.0 |
| Latency p50 (s) | 0.424 | 0.868 |
| Latency p95 (s) | 0.499 | 0.909 |
| Memory (MB) | 1716.4 | 1716.3 |
| Disk size (MB) | 538.1 | 1447.3 |
| Cost / 1K tokens ($) | 0.0048 | 0.006295 |

**2.7x smaller** and **1.3x faster** than the teacher on this hardware.

## Sample outputs

**Prompt:** 5. What is the role of the learning rate in a hiking analogy?

> In a hiking analogy, the learning rate refers to the rate at which the learning process is applied to the data. It is the rate at which the data is updated and the learning process is applied to the data. In other words, the learning rate is the rate at which the data is updated and the learning process is applied to the data.

**Prompt:** Prompt 8: How do you think the price should be considered?

> I think the price should be considered based on the following factors:
> 
> 1. **Value**: The value of the product is the key factor in determining the price. A product with a high value is more likely to be purchased.
> 
> 2. **Cost**: The cost of the product is another important factor. A product with a high cost is more likely to be purchased.
> 
> 3. **Quality**: The quality of the product is also important. A product with a high quality is more likely to be purchased.
> 
> 4. **Price elasticity**: The price elasticity of the product is a measure of how responsive the price is to changes in demand. A product with a high price elasticity is more likely to be purchased.
> 
> 5. **Competitive pricing**: The competitive pricing of the product is also important. A product with a high competitive pricing is more likely to be purchased.
> 
> 6. **Product features**: The features of the product are also important. A product with a high number of features is more likely to be purchased.
> 
> 7. **Product category**: The category of the product is also important. A product with a high category is more likely to be purchased.
> 
> 8. **Product category size**: The product category size is also important. A product with a

**Prompt:** 10. What are some common challenges that arise when trying to implement concurrency versus parallelism in a system?

> 10.1.1. **Concurrency vs. Parallelism**:
> 
> When trying to implement concurrency versus parallelism in a system, there are several challenges to consider.
> 
> 10.1.2. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10.1.3. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10.1.4. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10.1.5. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10.1.6. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10.1.7. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10.1.8. **Concurrency vs. Parallelism**:
> 
> Concurrency and parallelism are two different approaches to achieving concurrency.
> 
> 10

---
Run dir: `runs/mac-small` · dataset: `data/train.eval.jsonl` · reproduce with `distill report`.
