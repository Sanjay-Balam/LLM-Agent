"""
Example usage of the ManimAgent for generating Manim scripts.
"""

import os
from agent import ManimAgent
from validator import ManimScriptValidator

def setup_environment():
    """Setup environment variables for testing."""
    # You can set these in your .env file or environment
    # os.environ["OPENAI_API_KEY"] = "your_openai_key_here"
    # os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_key_here"
    pass

def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Initialize the agent (will use OpenAI by default)
    agent = ManimAgent(llm_provider="openai")
    
    # Generate a simple script
    request = "Create a blue circle that appears with animation and then transforms into a red square"
    
    print(f"Request: {request}")
    print("\nGenerated Script:")
    print("-" * 40)
    
    script = agent.generate_script(request)
    print(script)
    
    # Validate the script
    validator = ManimScriptValidator()
    is_valid, fixed_script, report = validator.validate_and_fix(script)
    
    print("\n" + report)
    
    if not is_valid:
        print("\nFixed Script:")
        print("-" * 40)
        print(fixed_script)

def example_template_usage():
    """Example using predefined templates."""
    print("\n=== Template Usage Example ===")
    
    agent = ManimAgent(llm_provider="openai")
    
    # List available patterns
    patterns = agent.list_available_patterns()
    print("Available patterns:")
    for pattern, description in patterns.items():
        print(f"  - {pattern}: {description}")
    
    # Generate script using template
    script = agent.generate_with_template(
        "animated_creation",
        shape="circle",
        shape_code="Circle(radius=1.5)",
        object_name="my_circle",
        color="BLUE"
    )
    
    print(f"\nGenerated Template Script:")
    print("-" * 40)
    print(script)

def example_anthropic_usage():
    """Example using Anthropic Claude."""
    print("\n=== Anthropic Usage Example ===")
    
    # Initialize with Anthropic
    agent = ManimAgent(llm_provider="anthropic")
    
    request = "Create a mathematical formula E=mc² that writes itself in green color"
    
    print(f"Request: {request}")
    print("\nGenerated Script:")
    print("-" * 40)
    
    script = agent.generate_script(request)
    print(script)

def example_complex_animation():
    """Example of complex animation generation."""
    print("\n=== Complex Animation Example ===")
    
    agent = ManimAgent(llm_provider="openai")
    
    request = """
    Create an animation that shows:
    1. A coordinate system with x and y axes
    2. A red circle at the origin
    3. The circle moves along a parabolic path y = x²
    4. Add a text label that shows the equation y = x²
    5. Make the circle leave a trail as it moves
    """
    
    print(f"Request: {request}")
    print("\nGenerated Script:")
    print("-" * 40)
    
    script = agent.generate_script(request)
    print(script)

def example_script_improvement():
    """Example of improving an existing script."""
    print("\n=== Script Improvement Example ===")
    
    agent = ManimAgent(llm_provider="openai")
    
    # Original script
    original_script = '''from manim import *

class SimpleCircle(Scene):
    def construct(self):
        circle = Circle()
        self.add(circle)
        self.wait(1)'''
    
    print("Original Script:")
    print("-" * 40)
    print(original_script)
    
    # Improve the script
    improvement_request = "Add animation, colors, and make it more visually appealing"
    
    improved_script = agent.improve_script(original_script, improvement_request)
    
    print(f"\nImprovement Request: {improvement_request}")
    print("\nImproved Script:")
    print("-" * 40)
    print(improved_script)

def example_script_explanation():
    """Example of script explanation."""
    print("\n=== Script Explanation Example ===")
    
    agent = ManimAgent(llm_provider="openai")
    
    script = '''from manim import *

class CircleToSquare(Scene):
    def construct(self):
        circle = Circle(radius=1)
        circle.set_color(BLUE)
        
        square = Square(side_length=2)
        square.set_color(RED)
        
        self.play(Create(circle))
        self.wait(1)
        self.play(Transform(circle, square))
        self.wait(1)'''
    
    print("Script to Explain:")
    print("-" * 40)
    print(script)
    
    explanation = agent.explain_script(script)
    
    print("\nExplanation:")
    print("-" * 40)
    print(explanation)

def example_list_capabilities():
    """Example of listing agent capabilities."""
    print("\n=== Agent Capabilities ===")
    
    agent = ManimAgent(llm_provider="openai")
    
    print("Available Objects:")
    objects = agent.list_available_objects()
    for obj, desc in objects.items():
        print(f"  - {obj}: {desc}")
    
    print("\nAvailable Animations:")
    animations = agent.list_available_animations()
    for anim, desc in animations.items():
        print(f"  - {anim}: {desc}")
    
    print("\nExample Scripts:")
    script = agent.get_example_script("basic_circle")
    print("Basic Circle Example:")
    print("-" * 40)
    print(script)

def example_validation_only():
    """Example of script validation without agent."""
    print("\n=== Validation Only Example ===")
    
    validator = ManimScriptValidator()
    
    # Test with invalid script
    invalid_script = '''from manim import *

class BadScene:
    def construct(self):
    circle = Circle()
    self.add(circle)
    self.wait(1)'''
    
    print("Invalid Script:")
    print("-" * 40)
    print(invalid_script)
    
    print("\nValidation Report:")
    print("-" * 40)
    report = validator.get_validation_report(invalid_script)
    print(report)
    
    # Fix the script
    is_valid, fixed_script, _ = validator.validate_and_fix(invalid_script)
    
    if not is_valid:
        print("\nFixed Script:")
        print("-" * 40)
        print(fixed_script)

def main():
    """Run all examples."""
    setup_environment()
    
    # Check if API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("⚠️  Warning: No API keys found in environment variables.")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY to run the examples.")
        print("You can still run the validation and template examples.\n")
    
    try:
        # Run examples that don't require API calls
        example_template_usage()
        example_list_capabilities()
        example_validation_only()
        
        # Run examples that require API calls
        if openai_key:
            example_basic_usage()
            example_complex_animation()
            example_script_improvement()
            example_script_explanation()
        
        if anthropic_key:
            example_anthropic_usage()
    
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your API keys correctly.")

if __name__ == "__main__":
    main()