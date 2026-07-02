# Distillation Report — runs/mac-small

*Generated 2026-07-03 by Distill Anything*

## Quality (LLM-as-judge)

**The student matches or beats the reference on 88% of 24 held-out prompts.**

Judge: `HuggingFaceTB/SmolLM2-360M-Instruct`, blind A/B with position swap.

| Student wins | Ties | Reference wins |
|:---:|:---:|:---:|
| 1 (4%) | 20 (83%) | 3 (12%) |

## Training metrics

- **eval_loss**: 0.7762
- **perplexity**: 2.173
- **teacher_agreement**: 0.8108

## Efficiency

| Metric | Student | Teacher (HuggingFaceTB/SmolLM2-360M-Instruct) |
|---|---:|---:|
| Parameters (M) | 134.5 | 361.8 |
| Tokens / s | 70.5 | 56.7 |
| Latency p50 (s) | 0.402 | 0.762 |
| Latency p95 (s) | 0.519 | 0.909 |
| Memory (MB) | 1716.4 | 1716.3 |
| Disk size (MB) | 538.1 | 1447.3 |
| Cost / 1K tokens ($) | 0.004729 | 0.005877 |

**2.7x smaller** and **1.2x faster** than the teacher on this hardware.

## Sample outputs

**Prompt:** 3. Can you provide an example of a list and a tuple in Python?

> 1. List:
> ```python
> my_list = [1, 2, 3, 4, 5]
> print(my_list)
> ```
> 2. Tuple:
> ```python
> my_tuple = (1, 2, 3, 4, 5)
> print(my_tuple)
> ```
> 
> Both are used to store a collection of items. The main difference is that a list is mutable, meaning it can be modified after it's created, while a tuple is immutable, meaning it cannot be modified after it's created.

**Prompt:** Prompt 4: What are the key factors that influence the hiker's decision to stop or continue climbing?

> 1. **Weather**: The weather can significantly impact hikers' decision to stop or continue climbing. Strong winds, heavy rain, or extreme temperatures can make it difficult to navigate and climb.
> 
> 2. **Distance**: The longer the hike, the more challenging it becomes to navigate and climb. If the hiker is short of time, they may be more likely to stop and rest.
> 
> 3. **Time of Day**: The time of day can also affect hikers' decision to stop or continue climbing. For example, if the hiker is tired and needs to rest, they may be more likely to stop and rest.
> 
> 4. **Weather Conditions**: The conditions of the weather can also impact hikers' decision to stop or continue climbing. For example, if the weather is too hot or too cold, the hiker may be more likely to stop and rest.
> 
> 5. **Personal Health**: The hiker's physical health can also

**Prompt:** 5. **Optimize server performance**: Optimize server performance by using techniques like caching, compression, and virtualization to reduce the overall cost of cloud computing. This can help reduce the overall cost of cloud computing.

> 5. **Optimize server performance**: Optimize server performance by using techniques like caching, compression, and virtualization to reduce the overall cost of cloud computing. This can help reduce the overall cost of cloud computing.

---
Run dir: `runs/mac-small` · dataset: `data/train.jsonl` · reproduce with `distill report`.
