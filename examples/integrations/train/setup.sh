#!/usr/bin/env bash
# Install-time step of the train recipe (network on): the tokenizer files of Qwen/Qwen2.5-0.5B-Instruct at a pinned
# revision, 11 MB and no weights, into the Hugging Face cache, where train.py reads them offline.
set -euo pipefail
hf download Qwen/Qwen2.5-0.5B-Instruct config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
