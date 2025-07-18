#!/usr/bin/env python3
"""
Test script to verify the system is working correctly.
"""

import os
import sys

def test_data_generation():
    """Test data generation."""
    print("=" * 50)
    print("TESTING DATA GENERATION")
    print("=" * 50)
    
    try:
        from data_generator import ManimDataGenerator
        generator = ManimDataGenerator()
        
        # Test single generation
        request, script = generator.generate_single_request_response()
        print(f"✅ Single generation works")
        print(f"Request: {request}")
        print(f"Script length: {len(script)} characters")
        
        # Test small batch
        data = generator.generate_training_data(10)
        print(f"✅ Batch generation works: {len(data)} samples")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_tokenizer():
    """Test tokenizer."""
    print("\n" + "=" * 50)
    print("TESTING TOKENIZER")
    print("=" * 50)
    
    try:
        from tokenizer import ManimTokenizer
        
        tokenizer = ManimTokenizer(vocab_size=1000)
        
        # Test with sample texts
        texts = [
            "Create a blue circle",
            "from manim import *",
            "self.play(Create(circle))"
        ]
        
        tokenizer.build_vocab(texts)
        
        # Test encoding/decoding
        test_text = "Create a red square"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"✅ Tokenizer works")
        print(f"Vocab size: {tokenizer.get_vocab_size()}")
        print(f"Original: {test_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\n" + "=" * 50)
    print("TESTING MODEL CREATION")
    print("=" * 50)
    
    try:
        from model import ManimLLM, ManimLLMConfig
        import torch
        
        # Create small model for testing
        config = ManimLLMConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_len=256
        )
        
        model = ManimLLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 50
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✅ Model creation works")
        print(f"Model parameters: {model.get_model_size():,}")
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_validation():
    """Test script validation."""
    print("\n" + "=" * 50)
    print("TESTING VALIDATION")
    print("=" * 50)
    
    try:
        from validator import ManimScriptValidator
        
        validator = ManimScriptValidator()
        
        # Test with valid script
        valid_script = '''from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)'''
        
        result = validator.validate_script(valid_script)
        
        print(f"✅ Validation works")
        print(f"Valid script: {result['is_valid']}")
        print(f"Errors: {len(result['errors'])}")
        print(f"Warnings: {len(result['warnings'])}")
        
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def test_knowledge_base():
    """Test knowledge base functions."""
    print("\n" + "=" * 50)
    print("TESTING KNOWLEDGE BASE")
    print("=" * 50)
    
    try:
        from knowledge_base import (
            get_object_code, get_animation_code, get_pattern_code,
            MANIM_OBJECTS, MANIM_ANIMATIONS
        )
        
        # Test object code generation
        circle_code = get_object_code("circle", radius=2)
        print(f"Circle code: {circle_code}")
        
        # Test animation code generation
        create_code = get_animation_code("create", "my_circle")
        print(f"Create code: {create_code}")
        
        # Test pattern code generation
        pattern_code = get_pattern_code("simple_shape", 
                                       shape="circle",
                                       shape_code="Circle()",
                                       object_name="test_circle",
                                       color="BLUE")
        print(f"Pattern code generated: {len(pattern_code)} characters")
        
        print(f"✅ Knowledge base works")
        print(f"Available objects: {len(MANIM_OBJECTS)}")
        print(f"Available animations: {len(MANIM_ANIMATIONS)}")
        
        return True
    except Exception as e:
        print(f"❌ Knowledge base failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 SYSTEM TESTING")
    print("Testing all components before training...")
    
    tests = [
        test_knowledge_base,
        test_data_generation,
        test_tokenizer,
        test_model_creation,
        test_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("Your system is ready for training!")
        print("\nNext steps:")
        print("1. Run: python train_model.py")
        print("2. Wait for training to complete")
        print("3. Use: python -c \"from agent import ManimAgent; agent = ManimAgent('custom')\"")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()