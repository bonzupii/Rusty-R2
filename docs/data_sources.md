# Rusty-R2 Data Sources

This document describes the datasets used to train Rusty-R2, including their sources, licenses, and categories.

## Dataset Catalog

| Name | URL | Category | License | Local Path | Notes |
|------|-----|----------|---------|------------|-------|
| APPS | https://huggingface.co/datasets/codeparrot/apps | core_code | MIT | data/hf/apps | Programming problems & tests (MIT) |
| MBPP | https://huggingface.co/datasets/Muennighoff/mbpp | core_code | CC-BY-4.0 | data/hf/mbpp | MBPP full with tests; verify license on HF card |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k | math | MIT | data/hf/gsm8k | Grade school math problems for reasoning (MIT) |
| English Wikipedia | https://huggingface.co/datasets/wikimedia/wikipedia | docs | CC-BY-SA/GFDL | data/hf/wikipedia_en_20231101 | English Wikipedia snapshot for general knowledge |
| OpenAssistant v1 | https://huggingface.co/datasets/OpenAssistant/oasst1 | sft | Apache-2.0 | data/hf/oasst1 | OpenAssistant SFT data for technical/helpful conversations |

## License Compliance

The following licenses are considered GPL/AGPL-compatible and are allowed in the dataset:

- MIT
- BSD (2/3-clause)
- Apache-2.0
- ISC
- MPL-2.0
- LGPL
- GPL
- AGPL
- CC0
- CC-BY
- CC-BY-SA
- GFDL
- Unlicense
- WTFPL
- Zlib
- Boost 1.0

The following licenses are explicitly NOT allowed:

- CC-BY-NC (Non-Commercial)
- CC-BY-ND (No Derivatives)
- CC-BY-NC-ND
- CC-BY-NC-SA
- "Research only"
- "Academic use only"
- Non-commercial licenses
- Proprietary licenses

## Directory Structure

Datasets are organized as follows:

```
data/
├── hf/                 # HuggingFace datasets saved via save_to_disk
│   ├── apps/
│   ├── mbpp/
│   ├── gsm8k/
│   └── oasst1/
└── raw/                # Manual corpora and user-provided content
    ├── docs/           # Documentation files
    ├── code_samples/   # Code examples
    └── ...             # Other raw content
```

## Important Notes

1. Users must respect the original licenses of each dataset.
2. NC/ND/research-only datasets are excluded by the script.
3. All datasets are under the 50GB total corpus size constraint.
4. All datasets should bias Rusty toward strong coding, systems, security, and high factual density.