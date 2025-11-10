# FILE: tests/test_checkpoint_roundtrip.py
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
import tempfile
from pathlib import Path
import torch

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rusty_r2.model.model_rwkv import TinyRWKVLM
from utils.checkpoint import save_checkpoint, load_checkpoint

class TestCheckpointRoundtrip(unittest.TestCase):
    """Test that checkpoints can be saved and loaded correctly."""

    def test_checkpoint_roundtrip(self):
        """Test that models can be saved and loaded with checkpoints."""
        print("\n--- Testing Checkpoint Roundtrip ---")
        
        # Create a small model for testing
        model_config = {
            "vocab_size": 100,  # Small vocab for test
            "d_embed": 128,     # Smaller embedding for test
            "d_hidden": 256,    # Smaller hidden for test
            "n_layers": 2,      # Fewer layers for test
            "dropout": 0.1,
            "gradient_checkpointing": False,
        }
        
        # Create model
        model = TinyRWKVLM(**model_config)
        
        # Create a temporary directory for the test checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            filename = "test_model.pt"
            
            # Test save_checkpoint
            save_checkpoint(
                model,
                checkpoint_dir=checkpoint_dir,
                filename=filename,
                model_config=model_config,
                global_step=42
            )
            
            # Verify the file was created
            checkpoint_path = checkpoint_dir / filename
            self.assertTrue(checkpoint_path.exists(), "Checkpoint file should be created")
            
            # Test load_checkpoint
            device = "cpu"  # Use CPU for test
            loaded_ckpt = load_checkpoint(checkpoint_path, device=device)
            
            # Verify we have the expected keys
            self.assertIn('model_state_dict', loaded_ckpt)
            self.assertIn('model_config', loaded_ckpt)
            self.assertIn('global_step', loaded_ckpt)
            
            # Verify the loaded config matches original
            loaded_config = loaded_ckpt['model_config']
            self.assertEqual(loaded_config, model_config)
            
            # Verify the global step was saved and loaded
            self.assertEqual(loaded_ckpt['global_step'], 42)
            
            # Rebuild model from loaded config
            loaded_model = TinyRWKVLM(**loaded_config).to(device)
            
            # Load state dict into the new model
            loaded_model.load_state_dict(loaded_ckpt['model_state_dict'])
            
            # Test that the model can run a forward pass without errors
            dummy_input = torch.randint(0, model_config['vocab_size'], (1, 16), device=device)
            with torch.no_grad():
                logits, values = loaded_model(dummy_input)
                
            # Check shapes match expectations
            expected_logits_shape = (1, 16, model_config['vocab_size'])
            expected_values_shape = (1, 16, 1)
            
            self.assertEqual(logits.shape, expected_logits_shape)
            self.assertEqual(values.shape, expected_values_shape)
            
            # Check that no NaNs or Infs exist
            self.assertFalse(torch.isnan(logits).any(), "Logits should not contain NaNs")
            self.assertFalse(torch.isinf(logits).any(), "Logits should not contain Infs")
            self.assertFalse(torch.isnan(values).any(), "Values should not contain NaNs")
            self.assertFalse(torch.isinf(values).any(), "Values should not contain Infs")
            
            print("Checkpoint roundtrip test passed - model can be saved and loaded correctly.")


if __name__ == "__main__":
    unittest.main()