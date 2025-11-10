# FILE: tests/test_tokenizer_specials.py
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
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tokenizers import Tokenizer

class TestTokenizerSpecials(unittest.TestCase):
    """Test that tokenizer special tokens have the expected IDs."""

    def test_special_token_ids(self):
        """Ensures that special tokens have the correct IDs."""
        print("\n--- Testing Tokenizer Special Token IDs ---")
        
        # Load tokenizer, if it exists
        tokenizer_path = Path("rusty_r2/tokenizer/tokenizer.json")
        if not tokenizer_path.exists():
            # Skip test if tokenizer is not available (would be generated during setup)
            self.skipTest("Tokenizer not found. It should be generated from training data.")
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            
            # Verify special token IDs as required by the system
            self.assertEqual(tokenizer.token_to_id("<pad>"), 0, 
                            "Padding token ID should be 0")
            self.assertEqual(tokenizer.token_to_id("<bos>"), 1, 
                            "Beginning-of-sequence token ID should be 1")
            self.assertEqual(tokenizer.token_to_id("<eos>"), 2, 
                            "End-of-sequence token ID should be 2")
            self.assertEqual(tokenizer.token_to_id("<unk>"), 3, 
                            "Unknown token ID should be 3")
            
            print("All special token IDs are correctly assigned.")


if __name__ == "__main__":
    unittest.main()