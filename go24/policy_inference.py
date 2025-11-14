import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Set, Tuple, Optional
import logging
from dataclasses import dataclass
import signal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Generation timed out!")


@dataclass
class PolicyOutput:
    """Structured output from policy model"""
    input_numbers: List[float]
    raw_output: str
    moves: Set[str]
    num_moves: int

    def __repr__(self):
        return (f"PolicyOutput(input={self.input_numbers}, "
                f"moves={self.num_moves}, "
                f"valid={len(self.moves)>0})")


class PolicyInference:
    """
    Production-ready inference for Game of 24 policy model.
    FIXED VERSION: No temperature/top_p issues, better error handling
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_size: int = 1000,
        max_new_tokens: int = 1024,  # Reduced from 1024
        verbose: bool = False,
        timeout_seconds: int = 120  # Add timeout
    ):
        """
        Initialize the policy inference engine.

        Args:
            model_path: Path to merged model directory
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Data type for model (torch.bfloat16, torch.float16, torch.float32)
            cache_size: Number of inference results to cache
            max_new_tokens: Maximum tokens to generate per inference
            verbose: Enable verbose logging
            timeout_seconds: Timeout for generation (None to disable)
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.cache_size = cache_size
        self.timeout_seconds = timeout_seconds

        # Initialize cache
        self._inference_cache = {}

        # Load model and tokenizer
        self._load_model()

        logger.info(f"PolicyInference initialized with model from {model_path}")

    def _load_model(self):
        """Load model and tokenizer from disk"""
        logger.info(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Ensure special tokens are configured
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|eot_id|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        logger.info(f"Loading model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self.model.eval()

        logger.info(f"✅ Model loaded successfully!")

        # Log model info
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
        except:
            pass

    def _format_input(self, numbers: List[float]) -> str:
        """Format input numbers into the prompt format used during training."""
        input_str = ' '.join(str(n) for n in numbers)
        prompt = f"Input: {input_str}\nPossible next steps:"
        return prompt

    def _parse_moves(self, output: str) -> Set[str]:
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        moves = set()
        for line in lines:
            if not line.startswith('<|'):
                # Only add complete moves (must end with ')')
                if line.endswith(')'):
                    moves.add(line)
        return moves


    def _get_cache_key(self, numbers: List[float]) -> str:
        """Create cache key from input numbers"""
        return tuple(numbers).__str__()

    def _get_cached_result(self, numbers: List[float]) -> Optional[PolicyOutput]:
        """Get cached result if available"""
        key = self._get_cache_key(numbers)
        return self._inference_cache.get(key)

    def _cache_result(self, numbers: List[float], result: PolicyOutput):
        """Cache inference result (with size limit)"""
        if len(self._inference_cache) >= self.cache_size:
            oldest_key = next(iter(self._inference_cache))
            del self._inference_cache[oldest_key]

        key = self._get_cache_key(numbers)
        self._inference_cache[key] = result

    def infer(
        self,
        numbers: List[float],
        use_cache: bool = True
    ) -> PolicyOutput:
        """
        Run inference on a single input.

        Args:
            numbers: List of numbers (1-4 numbers)
            use_cache: Whether to use cached results

        Returns:
            PolicyOutput with parsed moves
        """
        # Check cache
        if use_cache:
            cached = self._get_cached_result(numbers)
            if cached is not None:
                if self.verbose:
                    logger.info(f"✓ Cache hit for {numbers}")
                return cached

        # Format input
        prompt = self._format_input(numbers)

        if self.verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"Generating moves for: {numbers}")
            logger.info(f"Prompt: {repr(prompt)}")
            logger.info(f"{'='*70}")

        # Tokenize
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            if self.verbose:
                logger.info(f"Input tokens: {inputs['input_ids'].shape}")
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

        try:
            if self.verbose:
                logger.info(f"Starting generation (max_new_tokens={self.max_new_tokens})...")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  #
                    num_beams=1,     
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            if self.verbose:
                logger.info(f"Output tokens: {outputs.shape}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

        # Decode
        try:
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if self.verbose:
                logger.info(f"Raw output length: {len(output_text)} chars")
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise

        # Extract answer part (everything after "Possible next steps:")
        if "Possible next steps:" in output_text:
            answer = output_text.split("Possible next steps:")[-1]
            # Remove EOS token
            answer = answer.replace("<|eot_id|>", "")
        else:
            answer = output_text

        answer = answer.strip()

        if self.verbose:
            logger.info(f"Cleaned output: {answer[:200]}...")

        # Parse moves
        moves = self._parse_moves(answer)

        if self.verbose:
            logger.info(f"Parsed {len(moves)} moves")

        # Create result
        result = PolicyOutput(
            input_numbers=numbers,
            raw_output=answer,
            moves=moves,
            num_moves=len(moves)
        )

        # Cache result
        if use_cache:
            self._cache_result(numbers, result)

        return result

    def batch_infer(
        self,
        batch_numbers: List[List[float]],
        use_cache: bool = True,
        verbose_progress: bool = True
    ) -> List[PolicyOutput]:
        """Run inference on multiple inputs."""
        results = []
        total = len(batch_numbers)

        for i, numbers in enumerate(batch_numbers):
            if verbose_progress:
                logger.info(f"Processing {i+1}/{total}: {numbers}")

            result = self.infer(numbers, use_cache=use_cache)
            results.append(result)

        return results

    def clear_cache(self):
        """Clear inference cache"""
        self._inference_cache.clear()
        logger.info("Inference cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self._inference_cache),
            'max_size': self.cache_size,
            'utilization': len(self._inference_cache) / self.cache_size * 100
        }


def main():
    """Example usage of PolicyInference"""

    print("\n" + "="*70)
    print("POLICY INFERENCE - BASIC TEST")
    print("="*70)

    try:
        # Initialize inference engine with verbose output
        print("\n1. Loading model...")
        inference = PolicyInference(
            model_path="ft_go24_policy",
            verbose=True
        )

        # Test single inference
        print("\n2. Running single inference...")
        result = inference.infer([1, 1, 4, 6])

        print(f"\n✅ SUCCESS!")
        print(f"   Input: {result.input_numbers}")
        print(f"   Moves generated: {result.num_moves}")
        print(f"\n   First 5 moves:")
        for move in list(sorted(result.moves))[:5]:
            print(f"     {move}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()