"""
Custom Tokenizer for Manim-specific vocabulary
Creates a specialized tokenizer for Python code and Manim syntax.
"""

import re
import json
from typing import List, Dict, Set, Tuple
from collections import Counter
import pickle

class ManimTokenizer:
    """Custom tokenizer for Manim script generation."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
            '<REQ>': 4,  # Request separator
            '<SCR>': 5,  # Script separator
        }
        
        # Python/Manim specific tokens
        self.python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'import', 'from', 'return', 'self', 'True', 'False', 'None', 'and', 'or', 'not'
        }
        
        self.manim_keywords = {
            'Scene', 'construct', 'play', 'wait', 'add', 'remove', 'Create', 'Write',
            'FadeIn', 'FadeOut', 'Transform', 'Circle', 'Square', 'Text', 'MathTex',
            'Line', 'Arrow', 'Rectangle', 'Dot', 'set_color', 'shift', 'move_to',
            'scale', 'rotate', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ORIGIN', 'RED', 'BLUE',
            'GREEN', 'YELLOW', 'ORANGE', 'PURPLE', 'PINK', 'WHITE', 'BLACK', 'GRAY'
        }
        
        # Common patterns
        self.code_patterns = [
            r'def\s+\w+\(.*?\):',  # Function definitions
            r'class\s+\w+\(.*?\):',  # Class definitions
            r'self\.\w+\(.*?\)',  # Method calls
            r'\w+\.\w+\(.*?\)',  # Method calls
            r'[\w]+\s*=\s*[\w\.]+\(.*?\)',  # Assignments
            r'#.*',  # Comments
            r'""".*?"""',  # Docstrings
            r"'''.*?'''",  # Docstrings
            r'".*?"',  # String literals
            r"'.*?'",  # String literals
        ]
        
        # Initialize with special tokens
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        tokens = []
        
        # Handle code patterns first
        remaining_text = text
        for pattern in self.code_patterns:
            matches = re.finditer(pattern, remaining_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                # Add the matched pattern as a single token
                tokens.append(match.group())
                # Replace in remaining text to avoid double processing
                remaining_text = remaining_text.replace(match.group(), ' <PROCESSED> ')
        
        # Process remaining text
        # Split by whitespace and common delimiters
        words = re.findall(r'\w+|[^\w\s]', remaining_text)
        
        for word in words:
            if word != '<PROCESSED>':
                tokens.append(word)
        
        return tokens
    
    def _clean_token(self, token: str) -> str:
        """Clean and normalize tokens."""
        # Remove extra whitespace
        token = token.strip()
        
        # Keep important Python/Manim tokens as-is
        if token in self.python_keywords or token in self.manim_keywords:
            return token
        
        # Handle numbers
        if re.match(r'^\d+\.?\d*$', token):
            return '<NUM>'
        
        # Handle very long tokens
        if len(token) > 50:
            return '<LONG>'
        
        return token
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts."""
        print("Building vocabulary...")
        
        # Collect all tokens
        all_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            cleaned_tokens = [self._clean_token(token) for token in tokens]
            all_tokens.extend(cleaned_tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Add most frequent tokens to vocabulary
        most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for token, count in most_common:
            if token not in self.vocab and token.strip():
                self.vocab[token] = len(self.vocab)
        
        # Update inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
        print(f"Most common tokens: {list(dict(most_common[:20]).keys())}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize_text(text)
        cleaned_tokens = [self._clean_token(token) for token in tokens]
        
        token_ids = []
        for token in cleaned_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
            else:
                tokens.append('<UNK>')
        
        # Join tokens with appropriate spacing
        text = self._join_tokens(tokens)
        return text
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens back into readable text."""
        result = []
        
        for i, token in enumerate(tokens):
            # Skip special tokens in output
            if token in self.special_tokens:
                continue
            
            # Handle spacing
            if i > 0 and not self._needs_no_space_before(token) and not self._needs_no_space_after(tokens[i-1]):
                result.append(' ')
            
            result.append(token)
        
        return ''.join(result)
    
    def _needs_no_space_before(self, token: str) -> bool:
        """Check if token needs no space before it."""
        return token in ['(', ')', '[', ']', '{', '}', ',', '.', ':', ';', '=', '+', '-', '*', '/', '%']
    
    def _needs_no_space_after(self, token: str) -> bool:
        """Check if token needs no space after it."""
        return token in ['(', '[', '{', '.', '=', '+', '-', '*', '/', '%']
    
    def encode_batch(self, texts: List[str], max_length: int = 512, pad: bool = True) -> List[List[int]]:
        """Encode batch of texts."""
        batch_ids = []
        
        for text in texts:
            # Add BOS token
            token_ids = [self.vocab['<BOS>']] + self.encode(text)
            
            # Add EOS token
            if len(token_ids) < max_length:
                token_ids.append(self.vocab['<EOS>'])
            
            # Truncate if too long
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Pad if needed
            if pad and len(token_ids) < max_length:
                token_ids.extend([self.vocab['<PAD>']] * (max_length - len(token_ids)))
            
            batch_ids.append(token_ids)
        
        return batch_ids
    
    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """Decode batch of token IDs."""
        return [self.decode(token_ids) for token_ids in batch_ids]
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file."""
        tokenizer_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'python_keywords': self.python_keywords,
            'manim_keywords': self.manim_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.vocab = tokenizer_data['vocab']
        self.vocab_size = tokenizer_data['vocab_size']
        self.special_tokens = tokenizer_data['special_tokens']
        self.python_keywords = tokenizer_data['python_keywords']
        self.manim_keywords = tokenizer_data['manim_keywords']
        
        # Rebuild inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"Tokenizer loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_token_id(self, token: str) -> int:
        """Get token ID for a specific token."""
        return self.vocab.get(token, self.vocab['<UNK>'])
    
    def get_token(self, token_id: int) -> str:
        """Get token for a specific token ID."""
        return self.inverse_vocab.get(token_id, '<UNK>')
    
    def create_request_response_pair(self, request: str, response: str) -> List[int]:
        """Create encoded request-response pair for training."""
        # Format: <BOS> request <REQ> response <EOS>
        request_ids = self.encode(request)
        response_ids = self.encode(response)
        
        full_sequence = [
            self.vocab['<BOS>']
        ] + request_ids + [
            self.vocab['<REQ>']
        ] + response_ids + [
            self.vocab['<EOS>']
        ]
        
        return full_sequence

def build_tokenizer_from_data(data: List[Dict], vocab_size: int = 10000) -> ManimTokenizer:
    """Build tokenizer from training data."""
    tokenizer = ManimTokenizer(vocab_size=vocab_size)
    
    # Extract all texts
    texts = []
    for sample in data:
        texts.append(sample['request'])
        texts.append(sample['script'])
    
    # Build vocabulary
    tokenizer.build_vocab(texts)
    
    return tokenizer

if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = ManimTokenizer(vocab_size=1000)
    
    # Test with sample texts
    sample_texts = [
        "Create a blue circle",
        "from manim import *",
        "class CircleScene(Scene):",
        "    def construct(self):",
        "        circle = Circle(radius=1)",
        "        circle.set_color(BLUE)",
        "        self.play(Create(circle))",
        "        self.wait(1)"
    ]
    
    # Build vocabulary
    tokenizer.build_vocab(sample_texts)
    
    # Test encoding/decoding
    test_text = "Create a red square that moves to the right"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    # Test batch encoding
    batch_texts = ["Create a circle", "Make a square"]
    batch_encoded = tokenizer.encode_batch(batch_texts, max_length=20)
    batch_decoded = tokenizer.decode_batch(batch_encoded)
    
    print(f"Batch encoded: {batch_encoded}")
    print(f"Batch decoded: {batch_decoded}")