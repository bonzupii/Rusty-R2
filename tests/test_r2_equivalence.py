# FILE: tests/test_r2_equivalence.py
# Copyright (C) Micah L. Ostrow <bonzupii@protonmail.com> 
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# This file is part of Rusty-R2: A Scrapyard Language Model (Next Generation).
# 
# Rusty-R2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Rusty-R2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

import unittest
import torch
from pathlib import Path
import sys

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tokenizers import Tokenizer
from rusty_r2.model.model_rwkv import TinyRWKVLM

class TestR2Components(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests."""
        cls.r2_tokenizer_path = Path("rusty_r2/tokenizer/tokenizer.json")
        
        if not cls.r2_tokenizer_path.exists():
            raise FileNotFoundError("R2 tokenizer not found. Please run scripts/rebuild_tokenizer.py")
            
        cls.r2_tokenizer = Tokenizer.from_file(str(cls.r2_tokenizer_path))

    def test_tokenizer_roundtrip(self):
        """Ensures that encoding and then decoding returns the original string."""
        print("\n--- Testing R2 Tokenizer Round-trip ---")
        sample_text = "def main():\n    # This is a test\n    return 'hello world!'"
        
        # Test R2 Tokenizer (with prefix space)
        encoded = self.r2_tokenizer.encode(sample_text)
        decoded = self.r2_tokenizer.decode(encoded.ids)
        # The R2 tokenizer adds a prefix space, so we compare stripped versions
        self.assertEqual(decoded.lstrip(), sample_text)
        print("R2 Tokenizer round-trip successful.")

    def test_model_sanity_outputs(self):
        """Checks that the R2 model produces outputs of the correct shape."""
        print("\n--- Testing R2 Model Output Sanity ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_input = torch.randint(0, 1000, (1, 16), device=device)

        # R2 Model (RWKV)
        r2_model = TinyRWKVLM(vocab_size=self.r2_tokenizer.get_vocab_size()).to(device)
        r2_logits, r2_value = r2_model(dummy_input)
        
        self.assertEqual(r2_logits.shape, (1, 16, self.r2_tokenizer.get_vocab_size()))
        self.assertEqual(r2_value.shape, (1, 16, 1))
        self.assertFalse(torch.isnan(r2_logits).any(), "Logits should not contain NaNs")
        self.assertFalse(torch.isinf(r2_logits).any(), "Logits should not contain Infs")
        print("R2 TinyRWKVLM output shape and values are correct.")

if __name__ == "__main__":
    unittest.main()