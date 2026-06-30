import subprocess
import os
import sys
from typing import Tuple

class CodeSandbox:
    def __init__(self, work_dir: str):
        self.work_dir = os.path.abspath(work_dir)

    def execute_test(self, test_command: str) -> Tuple[bool, str]:
        """
        Runs the test suite inside the work directory.
        
        Args:
            test_command: The headless command to run (e.g., "pytest tests/test_calc.py").
            
        Returns:
            success: True if the test suite passes (exit code 0), False otherwise.
            error_traceback: Structured error output / stdout trace if it failed.
        """
        try:
            # Run command inside the specified work directory
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30.0  # safety timeout
            )
            success = (result.returncode == 0)
            
            if success:
                return True, ""
            
            # Combine stdout and stderr for traceback parsing
            combined_output = result.stdout + "\n" + result.stderr
            traceback = self._extract_traceback(combined_output)
            return False, traceback
            
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: Test execution exceeded 30 seconds limit."
        except Exception as e:
            return False, f"EXCEPTION: {str(e)}"

    def _extract_traceback(self, raw_output: str) -> str:
        """
        Extracts traceback details from raw test output (e.g., standard python trace or pytest output).
        """
        lines = raw_output.splitlines()
        extracted = []
        capture = False
        
        # Look for standard traceback markers
        for line in lines:
            if "Traceback (most recent call last):" in line or "=== FAILURES ===" in line:
                capture = True
            if capture:
                extracted.append(line)
                # Cap the capture size to keep context limits
                if len(extracted) > 50:
                    extracted.append("... [Traceback truncated] ...")
                    break
        
        if not extracted:
            # Fallback to last 20 lines if no standard marker matches
            return "\n".join(lines[-20:])
            
        return "\n".join(extracted)
