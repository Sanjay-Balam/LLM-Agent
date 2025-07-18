"""
Inference Engine for Custom Manim LLM
Handles model loading and text generation for Manim scripts.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import re
import os

from model import ManimLLM, ManimLLMConfig
from tokenizer import ManimTokenizer
from validator import ManimScriptValidator

class ManimInferenceEngine:
    """Inference engine for generating Manim scripts."""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_path: Path to the tokenizer file
            device: Device to use ("auto", "cpu", "cuda")
        """
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = ManimTokenizer()
        self.tokenizer.load(tokenizer_path)
        print(f"Tokenizer loaded with vocab size: {self.tokenizer.get_vocab_size()}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize validator
        self.validator = ManimScriptValidator()
        
        # Generation parameters
        self.default_generation_params = {
            'max_length': 512,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0,
            'do_sample': True
        }
    
    def _load_model(self, model_path: str) -> ManimLLM:
        """Load the trained model."""
        print(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = ManimLLMConfig(
                vocab_size=config_dict.get('vocab_size', self.tokenizer.get_vocab_size()),
                d_model=config_dict.get('d_model', 512),
                n_heads=config_dict.get('n_heads', 8),
                n_layers=config_dict.get('n_layers', 6),
                d_ff=config_dict.get('d_ff', 2048),
                max_len=config_dict.get('max_len', 1024)
            )
        else:
            # Use default config
            config = ManimLLMConfig(vocab_size=self.tokenizer.get_vocab_size())
        
        # Create model
        model = ManimLLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded with {model.get_model_size():,} parameters")
        return model
    
    def generate_script(self, request: str, **generation_params) -> str:
        """
        Generate a Manim script from a request.
        
        Args:
            request: Natural language request for animation
            **generation_params: Generation parameters
            
        Returns:
            Generated Manim script
        """
        # Merge generation parameters
        params = {**self.default_generation_params, **generation_params}
        
        # Prepare input
        input_text = request
        input_ids = self.tokenizer.encode(input_text)
        
        # Add BOS token and REQ separator
        input_ids = [
            self.tokenizer.vocab['<BOS>']
        ] + input_ids + [
            self.tokenizer.vocab['<REQ>']
        ]
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self._generate_with_params(input_tensor, params)
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        # Extract script part (after <REQ> token)
        script = self._extract_script_from_generation(generated_text)
        
        # Clean up script
        script = self._clean_generated_script(script)
        
        return script
    
    def _generate_with_params(self, input_ids: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Generate text with specified parameters."""
        max_length = params['max_length']
        temperature = params['temperature']
        top_k = params['top_k']
        top_p = params['top_p']
        repetition_penalty = params['repetition_penalty']
        do_sample = params['do_sample']
        
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            # Get next token logits
            with torch.no_grad():
                logits = self.model(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, generated, repetition_penalty
                    )
                
                # Apply top-k filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.vocab['<EOS>']:
                    break
        
        return generated
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] /= penalty
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, -float('inf'))
        logits_filtered.scatter_(1, top_k_indices, top_k_logits)
        return logits_filtered
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _extract_script_from_generation(self, generated_text: str) -> str:
        """Extract the script part from generated text."""
        # Look for script markers
        if '<REQ>' in generated_text:
            parts = generated_text.split('<REQ>')
            if len(parts) > 1:
                script_part = parts[1]
            else:
                script_part = generated_text
        else:
            script_part = generated_text
        
        # Remove special tokens
        for token in ['<BOS>', '<EOS>', '<PAD>', '<UNK>', '<REQ>', '<SCR>']:
            script_part = script_part.replace(token, '')
        
        return script_part.strip()
    
    def _clean_generated_script(self, script: str) -> str:
        """Clean up the generated script."""
        # Remove extra whitespaces
        script = re.sub(r'\s+', ' ', script)
        
        # Fix common formatting issues
        script = script.replace(' ( ', '(')
        script = script.replace(' ) ', ')')
        script = script.replace(' = ', '=')
        script = script.replace(' . ', '.')
        script = script.replace(' , ', ', ')
        script = script.replace(' : ', ':')
        
        # Ensure proper line breaks
        script = script.replace(' class ', '\n\nclass ')
        script = script.replace(' def ', '\n    def ')
        script = script.replace(' self.', '\n        self.')
        script = script.replace(' #', '\n        #')
        
        # Fix indentation
        lines = script.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('class '):
                cleaned_lines.append(line)
            elif line.startswith('def '):
                cleaned_lines.append('    ' + line)
            elif line.startswith('self.') or line.startswith('#'):
                cleaned_lines.append('        ' + line)
            elif line and not line.startswith('from ') and not line.startswith('import '):
                cleaned_lines.append('        ' + line)
            else:
                cleaned_lines.append(line)
        
        script = '\n'.join(cleaned_lines)
        
        # Ensure proper imports
        if 'from manim import *' not in script:
            script = 'from manim import *\n\n' + script
        
        return script
    
    def generate_and_validate(self, request: str, max_attempts: int = 3, **generation_params) -> Dict[str, Any]:
        """
        Generate and validate a Manim script.
        
        Args:
            request: Natural language request
            max_attempts: Maximum number of generation attempts
            **generation_params: Generation parameters
            
        Returns:
            Dictionary with generation results
        """
        best_script = None
        best_score = -1
        attempts = []
        
        for attempt in range(max_attempts):
            # Generate script
            script = self.generate_script(request, **generation_params)
            
            # Validate script
            validation_result = self.validator.validate_script(script)
            
            # Calculate score
            score = self._calculate_script_score(validation_result)
            
            attempts.append({
                'attempt': attempt + 1,
                'script': script,
                'validation': validation_result,
                'score': score
            })
            
            # Update best script
            if score > best_score:
                best_score = score
                best_script = script
            
            # If we get a perfect script, stop
            if validation_result['is_valid'] and len(validation_result['warnings']) == 0:
                break
        
        # Try to fix the best script
        if best_script:
            is_valid, fixed_script, report = self.validator.validate_and_fix(best_script)
            if is_valid:
                best_script = fixed_script
        
        return {
            'request': request,
            'best_script': best_script,
            'best_score': best_score,
            'attempts': attempts,
            'validation_report': self.validator.get_validation_report(best_script) if best_script else None
        }
    
    def _calculate_script_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate a score for the generated script."""
        score = 0.0
        
        # Base score for valid syntax
        if validation_result['is_valid']:
            score += 50.0
        
        # Deduct points for errors
        score -= len(validation_result['errors']) * 20.0
        
        # Deduct points for warnings
        score -= len(validation_result['warnings']) * 5.0
        
        # Add points for suggestions (good practices)
        score += len(validation_result['suggestions']) * 2.0
        
        return max(0.0, score)
    
    def batch_generate(self, requests: List[str], **generation_params) -> List[Dict[str, Any]]:
        """Generate scripts for multiple requests."""
        results = []
        
        for i, request in enumerate(requests):
            print(f"Generating script {i+1}/{len(requests)}: {request[:50]}...")
            result = self.generate_and_validate(request, **generation_params)
            results.append(result)
        
        return results
    
    def interactive_generation(self):
        """Interactive script generation."""
        print("Manim Script Generator - Interactive Mode")
        print("Enter your requests (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            request = input("\nEnter request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                break
            
            if not request:
                continue
            
            print("Generating script...")
            result = self.generate_and_validate(request)
            
            print("\n" + "="*50)
            print("GENERATED SCRIPT:")
            print("="*50)
            print(result['best_script'])
            
            print("\n" + "="*50)
            print("VALIDATION REPORT:")
            print("="*50)
            print(result['validation_report'])
            
            # Ask if user wants to save
            save = input("\nSave script to file? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                filename = input("Enter filename (without extension): ").strip()
                if filename:
                    with open(f"{filename}.py", 'w') as f:
                        f.write(result['best_script'])
                    print(f"Script saved to {filename}.py")

def main():
    """Main function for testing the inference engine."""
    # Check if model and tokenizer exist
    model_path = "best_model_epoch_10.pth"  # Adjust path as needed
    tokenizer_path = "tokenizer.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using trainer.py")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please train the model first using trainer.py")
        return
    
    # Initialize inference engine
    engine = ManimInferenceEngine(model_path, tokenizer_path)
    
    # Test generation
    test_requests = [
        "Create a blue circle that moves to the right",
        "Show a mathematical formula E=mc²",
        "Make a square that transforms into a triangle",
        "Create text that says 'Hello World'",
        "Draw a coordinate system with a red dot"
    ]
    
    print("Testing batch generation...")
    results = engine.batch_generate(test_requests)
    
    for i, result in enumerate(results):
        print(f"\n{'='*60}")
        print(f"REQUEST {i+1}: {result['request']}")
        print(f"SCORE: {result['best_score']:.1f}")
        print(f"{'='*60}")
        print(result['best_script'])
        print(f"{'='*60}")
    
    # Interactive mode
    print("\nStarting interactive mode...")
    engine.interactive_generation()

if __name__ == "__main__":
    main()