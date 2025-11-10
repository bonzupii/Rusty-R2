# Rusty-R2 Release Notes

This document summarizes the major upgrades and architectural changes involved in moving from the Rusty-R1 prototype to the more robust and powerful Rusty-R2.

## 1. Architecture Changes: GRU → RWKV

The core model architecture has been completely overhauled, moving from a simple GRU to a more powerful and modern alternative.

-   **`TinyRWKVLM`**: A 4-layer RWKV-style model is now the exclusive architecture.
    -   **Layers**: 4
    -   **Hidden Size**: 512
    -   **Embedding Dimension**: 256
    -   **TimeMix**: Implements parallelized RWKV-4 time-mixing blocks.
    -   **ChannelMix**: Implements RWKV-4 channel-mixing blocks.
    -   **Optimizations**: Weights are not tied between the embedding and output layers, as `d_embed` and `d_hidden` differ.
    -   **Value Head**: Includes an integrated value head, which outputs a scalar value alongside logits. This is essential for the new PPO training algorithm.

## 2. Tokenizer Improvements

The tokenizer has been significantly upgraded to improve its expressiveness and alignment with modern practices.

-   **Vocabulary Size**: Increased from **8,192 to 24,000**, allowing the model to learn a richer representation of both natural language and code.
-   **Prefix-Space Handling**: The tokenizer now uses `add_prefix_space=True`. This standard practice treats the start of a word consistently, improving tokenization of sentences.
-   **New Script**: A new `scripts/rebuild_tokenizer.py` script has been created to build the R2 tokenizer and save it to the `rusty_r2/tokenizer/` directory, keeping it separate from the R1 version.

## 3. RL Upgrade: REINFORCE → PPO-Lite

The agentic training methodology has been replaced with a more stable and efficient Proximal Policy Optimization (PPO) algorithm.

-   **Clipped Surrogate Objective**: PPO prevents destructively large policy updates by clipping the objective function, leading to more stable training than REINFORCE.
-   **GAE**: Advantage estimation now uses Generalized Advantage Estimation (GAE) for a better trade-off between bias and variance.
-   **Entropy Bonus**: An entropy bonus is added to the loss function to encourage exploration and prevent premature policy collapse.
-   **Partial Rewards**: The environment now provides more granular feedback. The agent receives a small positive reward (+0.3) for generating syntactically correct code and a larger reward (+0.7) for passing all tests.
-   **TensorBoard Logging**: Training statistics, including policy loss, value loss, mean reward, and entropy, are now logged to TensorBoard for better monitoring and analysis.

## 4. Quantization and Performance

A new `inference_runtime.py` script has been created to run and benchmark R2 models with significant performance enhancements.

-   **Quantization**: The runtime supports loading models with **8-bit or 4-bit quantization** using the `bitsandbytes` library. This dramatically reduces GPU memory usage, making it possible to run larger models on consumer hardware.
-   **`torch.compile`**: The model is compiled with `torch.compile()` (or `torch.jit.script` as a fallback) to optimize the execution graph, resulting in faster inference speeds.
-   **Benchmarking**: The script automatically measures and reports peak memory usage and generation speed (tokens/sec).

*Placeholder Benchmark Results:*
| Model         | Quantization | Peak Memory | Tokens/sec |
|---------------|--------------|-------------|------------|
| R1 (GRU)      | FP16         | ~1.5 GB     | ~30 tok/s  |
| R2 (RWKV)     | 8-bit        | ~1.2 GB     | ~55 tok/s  |
| R2 (RWKV)     | 4-bit        | ~0.9 GB     | ~50 tok/s  |

## 5. Protocol and Safety Enhancements

The agent interaction protocol has been hardened to improve reliability and safety.

-   **JSON-based Protocol**: The agent now communicates via structured JSON objects instead of a custom, regex-parsed format. This eliminates parsing ambiguity.
    -   Example: `{"action": "CMD", "command": "ls -l"}`
-   **Auto-Correction**: If the agent produces malformed JSON or an invalid action schema, the terminal and environment now provide a system message instructing it to correct its output, promoting self-healing.
-   **Sandboxing**: The `CodingEnv` now uses `resource.setrlimit` on Unix-like systems to enforce a hard CPU time limit on test execution, preventing runaway processes. This is in addition to the existing `timeout`.

## 6. Compatibility Notes

-   **R1 vs. R2**: The R2 ecosystem is a complete break from R1.
-   **Checkpoints**: R2 models cannot be used with R1 training scripts, and vice-versa.
-   **Tokenizer**: The new 24K vocabulary tokenizer is required for all R2 models.
-   **Protocol**: The new JSON-based protocol is not backward-compatible with the R1 agent's output format.
