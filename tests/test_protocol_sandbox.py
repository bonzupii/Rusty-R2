# FILE: tests/test_protocol_sandbox.py
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
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rusty_terminal import find_and_parse_json, execute_edit
import json

class TestProtocolSandbox(unittest.TestCase):
    """Test protocol parsing and sandboxing functionality."""

    def test_find_and_parse_json_valid(self):
        """Test that find_and_parse_json correctly parses valid JSON."""
        text = 'Some text before {"action": "MSG", "message": "hello"} some text after'
        result, error = find_and_parse_json(text)
        
        self.assertIsNone(error, f"Should not have error, got: {error}")
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertEqual(result["action"], "MSG", "Action should be MSG")
        self.assertEqual(result["message"], "hello", "Message should be hello")

    def test_find_and_parse_json_with_trailing_junk(self):
        """Test that find_and_parse_json handles JSON followed by junk."""
        text = '{"action": "CMD", "command": "ls"} some trailing text that is not JSON'
        result, error = find_and_parse_json(text)
        
        self.assertIsNone(error, f"Should not have error, got: {error}")
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertEqual(result["action"], "CMD", "Action should be CMD")
        self.assertEqual(result["command"], "ls", "Command should be ls")

    def test_find_and_parse_json_invalid(self):
        """Test that find_and_parse_json returns error for invalid JSON."""
        invalid_jsons = [
            "not json at all",
            '{"action": "MSG", "message": "unclosed quote',
            '{"action": "MSG", "message": "valid", "extra":}',  # Invalid syntax
            "random text without braces",
        ]
        
        for invalid_json in invalid_jsons:
            result, error = find_and_parse_json(invalid_json)
            self.assertIsNone(result, f"Should return None for invalid JSON: {invalid_json}")
            self.assertIsNotNone(error, f"Should return error for invalid JSON: {invalid_json}")

    def test_find_and_parse_json_no_action_key(self):
        """Test that find_and_parse_json returns error when JSON has no 'action' key."""
        text = '{"message": "no action key"}'
        result, error = find_and_parse_json(text)
        
        self.assertIsNone(result, "Should return None when no 'action' key")
        self.assertIsNotNone(error, "Should return error when no 'action' key")
        self.assertIn("'action'", error or "", "Error should mention missing 'action' key")

    def test_execute_edit_path_escape_rejection(self):
        """Test that execute_edit rejects paths trying to escape current directory."""
        # Create a temporary directory to work in
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Create a file in temp_dir to test editing
                test_file = Path(temp_dir) / "test.txt"
                test_file.write_text("original")
                
                # Try to escape to parent directory (should be rejected)
                result = execute_edit("../../escape_test.txt", "escaped", dry_run=False)
                self.assertIn("outside the current directory", result.lower(), 
                             "Should reject escaping to parent directory")
                
                # Try to go to another directory within temp_dir (should work)
                subdir = Path(temp_dir) / "subdir"
                subdir.mkdir(exist_ok=True)
                result2 = execute_edit("subdir/nested.txt", "nested content", dry_run=False)
                self.assertIn("edited", result2.lower(), "Should allow relative paths within directory")
                
                # Verify the nested file was actually created
                nested_file = subdir / "nested.txt"
                if nested_file.exists():
                    self.assertEqual(nested_file.read_text(), "nested content")
                
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()