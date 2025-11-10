# Rusty-R2: A Scrapyard Language Model (Next Generation)

Rusty-R2 is the next generation of our small, agentic language model, designed to function as a cynical, technically-focused terminal assistant. This project represents a significant architectural upgrade from Rusty-R1, moving to more modern model architectures, a robust PPO-based training pipeline, and enhanced safety features.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](./LICENSE) file for details.

## Persona: Rusty-R2

- **Origin**: An upgraded hackerbot from a digital Scrapyard.
- **Attitude**: Blunt, concise, mildly sarcastic, and technically dense.
- **Focus**: Provides correct, concrete answers without ethical lectures or corporate fluff.

## Features

-   **Modern Architecture**: Uses `TinyRWKVLM` architecture for improved performance and capabilities.
-   **Upgraded Tokenizer**: A 24K vocabulary Byte-Level BPE tokenizer with `add_prefix_space=True` for better code and text handling.
-   **Two-Phase Training**:
    1.  **Supervised Fine-Tuning**: Comprehensive script (`train_supervised.py`) with 8-bit optimizers, cosine LR schedule, gradient checkpointing, and resume capabilities.
    2.  **Agentic Fine-Tuning**: PPO-Lite based reinforcement learning (`agent/train_agentic.py`) with a `RolloutBuffer`, GAE, and partial rewards for syntax and test passing.
-   **JSON-based Interaction Protocol**: Agent communicates via structured JSON objects (`MSG`, `CMD`, `EDIT`) for unambiguous parsing and execution.
-   **Enhanced Sandboxing**: Terminal commands run in isolated temporary directories with resource limits (CPU time, file size) to prevent malicious or runaway processes.
-   **Quantized Inference**: `inference_runtime.py` supports 8-bit (and a 4-bit stub) quantization and `torch.compile()` for efficient deployment.
-   **Interactive Terminal**: A runnable `rusty_terminal.py` for direct, sandboxed interaction with the trained agent.

## Project Structure

```
rusty-r2/
├── agent/
│   ├── env.py              # The coding gym environment for RL training with sandboxing.
│   └── train_agentic.py    # Script for PPO-Lite based agentic fine-tuning.
├── checkpoints_r2_supervised/ # Stores supervised model checkpoints.
├── checkpoints_r2_agentic/ # Stores agentic model checkpoints.
├── data/
│   ├── raw/                # Raw text/code files for training the tokenizer.
│   └── hf/                 # Downloaded Hugging Face datasets.
├── docs/
│   └── rusty_r2_upgrade.md # Documentation for the R2 upgrade.
├── memory/                 # (Placeholder for future memory components)
├── model/                  # (Old R1 GRU model - now removed)
├── runs/                   # Temporary directories for agent environment runs.
├── rusty_r2/
│   ├── model/
│   │   └── model_rwkv.py   # TinyRWKVLM model implementation.
│   └── tokenizer/
│       └── tokenizer.json  # The trained R2 tokenizer file.
├── scripts/
│   ├── download_datasets.py # Script to download curated code datasets.
│   └── rebuild_tokenizer.py # Script to build the R2 tokenizer from data.
├── tasks/
│   └── sum_list/           # An example coding task for the agent.
├── tests/
│   └── test_r2_equivalence.py # Test suite for R2 components.
├── utils/
│   ├── checkpoint.py       # Centralized checkpoint saving/loading utilities.
│   └── rl.py               # PPO RolloutBuffer and action generation helpers.
├── README.md               # This file.
├── requirements.txt        # Python dependencies.
├── rusty_terminal.py       # Interactive terminal for the agent.
└── train_supervised.py     # Script for initial supervised model training.
```

## Setup

1.  **Clone the repository.**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    # For 8-bit optimizers and quantization (highly recommended):
    pip install bitsandbytes
    ```

## Usage

### 1. Prepare Data
Download curated code datasets from Hugging Face. These will be saved to `./data/hf/`.
```bash
python scripts/download_datasets.py
```

### 2. Build the Tokenizer
This command trains a new 24K vocabulary tokenizer from the files in `./data/raw/` and the downloaded datasets in `./data/hf/`, saving it to `rusty_r2/tokenizer/tokenizer.json`.
```bash
python scripts/rebuild_tokenizer.py
```

### 3. Supervised Training (Phase 1)
Train the base language model on the concatenated text data. This creates a foundational model that understands language and code structure.
```bash
python train_supervised.py \

    --tokenizer_path rusty_r2/tokenizer/tokenizer.json \
    --data_dir data/hf/ \
    --max_steps 50000 \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --seq_len 1024 \
    --gradient_checkpointing \
    --checkpoints_dir ./checkpoints_r2_supervised \
    --device cuda
```
Checkpoints will be saved in `./checkpoints_r2_supervised/`.

### 4. Agentic Fine-Tuning (Phase 2)
Fine-tune the model using Proximal Policy Optimization (PPO) on coding tasks. This requires a base model checkpoint from the supervised training phase.
```bash
python agent/train_agentic.py \
    --init_checkpoint ./checkpoints_r2_supervised/step_XXXXX.pt \
    --tokenizer_path rusty_r2/tokenizer/tokenizer.json \
    --total_timesteps 100000 \
    --num_steps 256 \
    --ppo_epochs 4 \
    --num_minibatches 4 \
    --log_dir runs_ppo_r2 \
    --checkpoints_dir ./checkpoints_r2_agentic \
    --device cuda
```

### 5. Interact with Rusty-R2
Run the interactive terminal to chat with your trained agent.
```bash
python rusty_terminal.py --checkpoint ./checkpoints_r2_agentic/agent_step_XXXXX.pt
```
Inside the terminal, you can type messages, or use meta-commands:
- `:q` or `exit`: Quit the terminal.
- `:dry`: Toggle dry-run mode, which prints commands instead of executing them.

### 6. Run Inference Benchmarks
Test the model's generation speed and memory usage.
```bash
python inference_runtime.py \
    --checkpoint ./checkpoints_r2_supervised/step_XXXXX.pt \
    --quantization 8bit \
    --prompt "USER: Write a python function to reverse a string."
    --max_new_tokens 100 \
    --compile \
    --device cuda
```

### 7. Run Tests
Execute the test suite to verify R2 component sanity.
```bash
python -m unittest tests/test_r2_equivalence.py
```

## Model & Protocol

### Model Architecture
-   **TinyRWKVLM**: 4-layer RWKV-style model with parallelized `TimeMix`.
    -   `d_embed`: 256, `d_hidden`: 512, `n_layers`: 4.
    -   Optimizations: Weights are not tied between the embedding and output layers, as `d_embed` and `d_hidden` differ.

### Interaction Protocol
The agent is trained to respond in a strict JSON format:
-   `{"action": "MSG", "message": "<your_message>"}`: For text responses.
-   `{"action": "CMD", "command": "<your_command>"}`: For executing safe shell commands.
-   `{"action": "EDIT", "path": "path/to/file.ext", "content": "<full new file content>"}`: For overwriting files.
The `rusty_terminal.py` script is responsible for parsing these actions and executing them safely within a sandboxed environment.