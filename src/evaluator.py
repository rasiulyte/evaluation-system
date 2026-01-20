"""
Evaluator: Core evaluation engine for hallucination detection.

Responsibilities:
- Load test cases and prompts
- Call LLM API (OpenAI-compatible interface)
- Parse responses
- Calculate metrics
- Save timestamped results
- Support multiple runs for consistency measurement
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import re

from dotenv import load_dotenv
from openai import OpenAI


class Evaluator:
    """
    Core evaluation engine that runs test cases through prompts and evaluates responses.
    """

    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 base_url: Optional[str] = None,
                 test_cases_dir: str = "data/test_cases",
                 prompts_dir: str = "prompts",
                 results_dir: str = "data/results"):
        """
        Initialize evaluator.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var or .env file if not provided)
            model: Model to use (default: gpt-4)
            base_url: Base URL for OpenAI-compatible API (optional)
            test_cases_dir: Directory containing test case JSON files
            prompts_dir: Directory containing prompt template files
            results_dir: Directory to save timestamped results
        """
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        ) if base_url else OpenAI(api_key=self.api_key)
        
        self.test_cases_dir = Path(test_cases_dir)
        self.prompts_dir = Path(prompts_dir)
        self.results_dir = Path(results_dir)
        
        self._load_test_cases()
        self._load_prompts()

    def _load_test_cases(self) -> None:
        """Load all test cases from JSON files."""
        self.test_cases = {}
        
        for json_file in self.test_cases_dir.glob("*.json"):
            with open(json_file, "r") as f:
                cases = json.load(f)
                for case in cases:
                    self.test_cases[case["id"]] = case

    def _load_prompts(self) -> None:
        """Load all prompt templates from text files."""
        self.prompts = {}
        
        for txt_file in self.prompts_dir.glob("*.txt"):
            prompt_id = txt_file.stem
            with open(txt_file, "r") as f:
                self.prompts[prompt_id] = f.read()

    def get_prompt(self, prompt_id: str) -> str:
        """Get prompt template by ID."""
        return self.prompts.get(prompt_id)

    def format_prompt(self, prompt_template: str, context: str, response: str) -> str:
        """Format prompt template with context and response."""
        return prompt_template.format(context=context, response=response)

    def evaluate_single(self, 
                       test_case_id: str, 
                       prompt_id: str,
                       temperature: float = 0.7,
                       max_tokens: int = 500) -> Dict[str, Any]:
        """
        Evaluate a single test case using a prompt.
        
        Args:
            test_case_id: ID of test case from JSON
            prompt_id: ID of prompt template (v1_zero_shot, etc.)
            temperature: Temperature for LLM sampling
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with test result including prediction, confidence, timing
        """
        test_case = self.test_cases.get(test_case_id)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        
        prompt_template = self.get_prompt(prompt_id)
        if not prompt_template:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        # Format the prompt
        formatted_prompt = self.format_prompt(
            prompt_template,
            test_case["context"],
            test_case["response"]
        )
        
        # Call LLM
        start_time = datetime.now()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = datetime.now()
            
            llm_output = response.choices[0].message.content.strip()
            
            # Parse response
            prediction, confidence = self._parse_response(llm_output, prompt_id)
            
            result = {
                "test_case_id": test_case_id,
                "prompt_id": prompt_id,
                "ground_truth": test_case["label"],
                "prediction": prediction,
                "confidence": confidence,
                "llm_output": llm_output,
                "correct": prediction == test_case["label"],
                "duration_ms": (end_time - start_time).total_seconds() * 1000,
                "timestamp": start_time.isoformat(),
                "model": self.model
            }
            
            return result
            
        except Exception as e:
            return {
                "test_case_id": test_case_id,
                "prompt_id": prompt_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _parse_response(self, llm_output: str, prompt_id: str) -> Tuple[str, float]:
        """
        Parse LLM response to extract classification and confidence.
        
        Args:
            llm_output: Raw LLM output
            prompt_id: Prompt type (affects parsing strategy)
            
        Returns:
            (prediction, confidence) where prediction is "hallucination"/"grounded"
                and confidence is 0.0-1.0
        """
        output_lower = llm_output.lower()
        
        if prompt_id == "v5_structured_output":
            # Parse JSON response
            try:
                # Find JSON in output
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    prediction = data.get("classification", "").lower()
                    confidence = data.get("confidence", 0.5)
                    
                    if prediction in ["hallucinated", "hallucination"]:
                        prediction = "hallucination"
                    elif prediction == "grounded":
                        pass
                    else:
                        prediction = "unknown"
                    
                    return prediction, float(confidence)
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Default parsing: look for keywords
        if any(word in output_lower for word in ["hallucinated", "hallucination"]):
            prediction = "hallucination"
        elif "grounded" in output_lower:
            prediction = "grounded"
        else:
            prediction = "unknown"
        
        # Extract confidence if present
        confidence = 0.5
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', output_lower)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except ValueError:
                pass
        
        return prediction, confidence

    def evaluate_batch(self,
                       prompt_id: str,
                       test_case_ids: Optional[List[str]] = None,
                       max_workers: int = 1,
                       **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases.
        
        Args:
            prompt_id: ID of prompt to use for all cases
            test_case_ids: List of test case IDs (None = all)
            max_workers: Number of parallel workers (currently 1)
            **kwargs: Additional arguments passed to evaluate_single
            
        Returns:
            List of evaluation results
        """
        if test_case_ids is None:
            test_case_ids = list(self.test_cases.keys())
        
        results = []
        for test_case_id in test_case_ids:
            result = self.evaluate_single(test_case_id, prompt_id, **kwargs)
            results.append(result)
        
        return results

    def save_results(self, results: List[Dict], 
                    run_name: Optional[str] = None) -> Path:
        """
        Save evaluation results to timestamped JSON file.
        
        Args:
            results: List of result dictionaries
            run_name: Optional run name (included in filename)
            
        Returns:
            Path to saved results file
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}" + (f"_{run_name}" if run_name else "") + ".json"
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        return filepath

    def get_test_case(self, test_case_id: str) -> Optional[Dict]:
        """Get a test case by ID."""
        return self.test_cases.get(test_case_id)

    def get_test_cases_by_failure_mode(self, failure_mode: str) -> List[Dict]:
        """Get all test cases for a specific failure mode."""
        return [
            case for case in self.test_cases.values()
            if case.get("failure_mode") == failure_mode
        ]

    def list_test_cases(self) -> List[str]:
        """List all test case IDs."""
        return list(self.test_cases.keys())

    def list_prompts(self) -> List[str]:
        """List all available prompt IDs."""
        return list(self.prompts.keys())
