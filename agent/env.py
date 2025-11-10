import json
# FILE: agent/env.py
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

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import resource, but allow graceful degradation on platforms without it
try:
    import resource
except ImportError:
    resource = None

# --- Safety & Sandboxing Configuration ---
# Allowed commands for any potential CMD action (though this env focuses on EDIT)
ALLOWED_COMMANDS = {"python", "pytest"}
# CPU time limit for the subprocess in seconds
DEFAULT_CPU_LIMIT = 5
DEFAULT_TIMEOUT = 15

def set_limits(cpu_seconds: int):
    """
    Returns a pre-exec function to set resource limits for the child process (POSIX only).
    This function uses the `resource` module, which is not available on Windows.
    On Windows, these limits will not be applied, reducing sandboxing effectiveness.
    """
    if resource is None:
        # Return None to indicate no preexec function should be used
        return None
        
    def _set():
        # Set CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    return _set

class CodingEnv:
    """
    A coding gym environment for Rusty-R2.

    This version uses a JSON-based action space and enforces stricter sandboxing.
    The agent's goal is to modify a Python file to pass unit tests by emitting
    JSON actions.
    """

    def __init__(
        self,
        tasks_root: str = "./tasks",
        max_steps: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        cpu_limit: int = DEFAULT_CPU_LIMIT,
    ):
        self.tasks_root = Path(tasks_root)
        if not self.tasks_root.is_dir():
            raise FileNotFoundError(f"Tasks root directory not found: {self.tasks_root.resolve()}")

        self.max_steps = max_steps
        self.timeout = timeout
        self.cpu_limit = cpu_limit
        
        self.tasks: List[str] = sorted([d.name for d in self.tasks_root.iterdir() if d.is_dir()])
        if not self.tasks:
            raise FileNotFoundError(f"No tasks found in {self.tasks_root.resolve()}")
        
        self._task_idx = 0
        self.runs_dir = Path("./runs")
        self.runs_dir.mkdir(exist_ok=True)

        self.work_dir: Optional[Path] = None
        self.current_task: Optional[str] = None
        self.prompt: Optional[str] = None
        self.step_count = 0

    def _build_observation(self, file_content: str, test_result: str, system_msg: Optional[str] = None) -> str:
        """Constructs the observation string, optionally with a system message."""
        obs = ""
        if system_msg:
            obs += f"SYSTEM_MSG: {system_msg}\n\n"
        obs += (
            f"USER: {self.prompt}\n\n"
            f"CURRENT_FILE: template.py\n{file_content}\n\n"
            f"TEST_RESULT:\n{test_result}\n"
        )
        return obs

    def reset(self, task_name: Optional[str] = None) -> str:
        """Resets the environment and returns the initial observation."""
        if task_name:
            if task_name not in self.tasks:
                raise ValueError(f"Task '{task_name}' not found in {self.tasks_root}")
            self.current_task = task_name
        else:
            self.current_task = self.tasks[self._task_idx]
            self._task_idx = (self._task_idx + 1) % len(self.tasks)

        task_dir = self.tasks_root / self.current_task
        self.work_dir = self.runs_dir / f"tmp_{uuid.uuid4()}"
        shutil.copytree(task_dir, self.work_dir)

        self.prompt = (self.work_dir / "prompt.txt").read_text()
        template_content = (self.work_dir / "template.py").read_text()
        self.step_count = 0

        return self._build_observation(template_content, "None")

    def step(self, action_json: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Executes one step based on a JSON action from the agent.
        """
        if self.work_dir is None:
            raise RuntimeError("Must call reset() before step()")

        self.step_count += 1
        template_path = self.work_dir / "template.py"
        current_content = template_path.read_text()

        try:
            action = json.loads(action_json)
            action_type = action.get("action")

            if action_type == "EDIT":
                new_content = action.get("content")
                if new_content is None:
                    raise ValueError("'EDIT' action requires a 'content' field.")
                
                # Limitation: The agent must generate the entire file content.
                # For more advanced agents, consider implementing line-based or
                # patch-based editing actions to reduce the complexity of the action space.
                template_path.write_text(new_content)
                
                # Partial reward for syntax check
                try:
                    compile(new_content, '<string>', 'exec')
                    syntax_reward = 0.3
                    # Run tests with sandboxing only if syntax is valid
                    output, passed = self._run_tests()
                    test_reward = 0.7 if passed else 0.0
                except SyntaxError:
                    syntax_reward = -0.1
                    # Skip running tests when syntax is clearly broken
                    output = "SyntaxError: Code has syntax errors and cannot be executed"
                    passed = False
                    test_reward = 0.0

                # Final reward
                reward = syntax_reward + test_reward
                
                done = passed or (self.step_count >= self.max_steps)
                observation = self._build_observation(new_content, output)
                info = {"passed": passed, "task": self.current_task, "work_dir": str(self.work_dir)}

                return observation, reward, done, info

            elif action_type == "MSG":
                # Agent sends a message, e.g., for clarification. No file change.
                done = self.step_count >= self.max_steps
                reward = -0.1  # Penalize for not taking a useful action
                observation = self._build_observation(current_content, "No tests run.", system_msg="Agent sent a message.")
                info = {"passed": False}
                return observation, reward, done, info

            else:
                raise ValueError(f"Unknown action type: '{action_type}'")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Auto-correction: Punish and inform the agent of the format error.
            done = self.step_count >= self.max_steps
            reward = -0.5  # Heavy penalty for malformed action
            system_msg = f"Invalid action format. Must be JSON with 'action' ('EDIT' or 'MSG'). Error: {e}"
            observation = self._build_observation(current_content, "Action failed.", system_msg=system_msg)
            info = {"passed": False, "error": str(e)}
            return observation, reward, done, info

    def _run_tests(self) -> Tuple[str, bool]:
        """Runs the test suite in a sandboxed subprocess."""
        try:
            # Get the limits function, which may be None on platforms without resource module
            limits_fn = set_limits(self.cpu_limit)
            
            result = subprocess.run(
                ["python", "tests.py"],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                preexec_fn=limits_fn if limits_fn is not None else None,
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            passed = False
            output = f"Timeout: Tests exceeded {self.timeout}s real time."
        except Exception as e:
            # This could catch errors from preexec_fn on non-Unix systems
            passed = False
            output = f"Failed to run tests: {e}"

        # Truncate output to prevent unbounded context growth
        max_len = 4096
        if len(output) > max_len:
            output = output[:max_len] + "...(truncated)"
        
        return output, passed

    def close(self):
        if self.work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            self.work_dir = None

    def __del__(self):
        self.close()