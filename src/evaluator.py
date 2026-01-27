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


# Pricing per 1M tokens (as of January 2025)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


class Evaluator:
    """
    Core evaluation engine that runs test cases through prompts and evaluates responses.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 base_url: Optional[str] = None,
                 test_cases_dir: Optional[str] = None,
                 prompts_dir: Optional[str] = None,
                 results_dir: Optional[str] = None):
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

        # Resolve paths relative to project root (parent of src/)
        project_root = Path(__file__).parent.parent
        self.test_cases_dir = Path(test_cases_dir) if test_cases_dir else project_root / "data" / "test_cases"
        self.prompts_dir = Path(prompts_dir) if prompts_dir else project_root / "prompts"
        self.results_dir = Path(results_dir) if results_dir else project_root / "data" / "results"
        
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

            # Extract token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # Debug: print token info
            print(f"    [TOKEN DEBUG] {test_case_id}: {input_tokens} in, {output_tokens} out, {total_tokens} total")

            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens)

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
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost
            }

            return result
            
        except Exception as e:
            return {
                "test_case_id": test_case_id,
                "prompt_id": prompt_id,
                "ground_truth": test_case.get("label", ""),
                "prediction": "",
                "confidence": 0.0,
                "llm_output": "",
                "correct": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0
            }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD based on token usage and model pricing.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens

        Returns:
            Cost in USD
        """
        # Get pricing for current model, default to gpt-4o-mini pricing
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING.get("gpt-4o-mini"))

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

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

        # Try JSON parsing for any prompt that might return structured output
        if prompt_id.startswith("v5_") or prompt_id.startswith("v6_") or "structured" in prompt_id or "{" in llm_output:
            # Parse JSON response - try multiple strategies
            json_data = None

            # Strategy 1: Try to find JSON in markdown code block
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', llm_output)
            if code_block_match:
                try:
                    json_data = json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Strategy 2: Find JSON object with non-greedy match
            if json_data is None:
                # Use a more careful regex that finds balanced braces
                json_match = re.search(r'\{[^{}]*"classification"[^{}]*\}', llm_output, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            # Strategy 3: Try the whole output as JSON
            if json_data is None:
                try:
                    json_data = json.loads(llm_output.strip())
                except json.JSONDecodeError:
                    pass

            # Strategy 4: Find any JSON object (greedy, as fallback)
            if json_data is None:
                json_match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*\n|$)', llm_output)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            # Process the JSON data if found
            if json_data and isinstance(json_data, dict):
                prediction = str(json_data.get("classification", "")).lower().strip()
                confidence = json_data.get("confidence", 0.5)

                # Normalize prediction value
                if prediction in ["hallucinated", "hallucination"]:
                    prediction = "hallucination"
                elif prediction == "grounded":
                    pass
                else:
                    prediction = "unknown"

                # Ensure confidence is valid
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.5

                return prediction, confidence

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
