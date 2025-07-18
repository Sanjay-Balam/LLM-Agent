"""
ManimAgent - AI Agent for generating Manim scripts using custom LLM.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from knowledge_base import (
    MANIM_OBJECTS, MANIM_ANIMATIONS, MANIM_COLORS, MANIM_POSITIONS,
    COMMON_PATTERNS, EXAMPLE_SCRIPTS, get_object_code, get_animation_code,
    get_pattern_code, get_full_script
)

# Load environment variables
load_dotenv()

# Import custom LLM components
try:
    from inference import ManimInferenceEngine
    CUSTOM_LLM_AVAILABLE = True
except ImportError:
    CUSTOM_LLM_AVAILABLE = False
    print("Custom LLM not available. Please train the model first.")

# Import external APIs (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class ManimAgent:
    """AI Agent for generating Manim animation scripts."""
    
    def __init__(self, llm_provider="custom", model_path=None, tokenizer_path=None, 
                 model_name=None, api_key=None):
        """
        Initialize the Manim Agent.
        
        Args:
            llm_provider: "custom", "openai", or "anthropic"
            model_path: Path to custom trained model
            tokenizer_path: Path to custom tokenizer
            model_name: Specific model to use (for external APIs)
            api_key: API key for external LLM providers
        """
        self.llm_provider = llm_provider.lower()
        
        # Initialize based on provider
        if self.llm_provider == "custom":
            self._initialize_custom_llm(model_path, tokenizer_path)
        else:
            self.model_name = model_name or self._get_default_model()
            self.api_key = api_key or self._get_api_key()
            self.client = self._initialize_llm_client()
            self.system_prompt = self._create_system_prompt()
    
    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        if self.llm_provider == "openai":
            return "gpt-4"
        elif self.llm_provider == "anthropic":
            return "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        if self.llm_provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _initialize_custom_llm(self, model_path: str = None, tokenizer_path: str = None):
        """Initialize the custom LLM."""
        if not CUSTOM_LLM_AVAILABLE:
            raise ImportError("Custom LLM components not available. Please train the model first.")
        
        # Default paths
        if model_path is None:
            model_path = "best_model_epoch_10.pth"
        if tokenizer_path is None:
            tokenizer_path = "tokenizer.pkl"
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        # Initialize inference engine
        self.inference_engine = ManimInferenceEngine(model_path, tokenizer_path)
        print(f"Custom LLM initialized with model: {model_path}")
    
    def _initialize_llm_client(self):
        """Initialize the external LLM client."""
        try:
            if self.llm_provider == "openai":
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI package not installed")
                import openai
                return openai.OpenAI(api_key=self.api_key)
            elif self.llm_provider == "anthropic":
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("Anthropic package not installed")
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}")
        except Exception as e:
            raise Exception(f"Failed to initialize LLM client: {e}")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return f"""You are a specialized AI agent for generating Python Manim scripts. Manim is a library for creating mathematical animations.

Your task is to generate complete, working Manim scripts based on user requests. Follow these guidelines:

1. ALWAYS include proper imports: `from manim import *` and `import numpy as np`
2. Create a Scene class with a descriptive name
3. Use the construct() method for animation logic
4. Available objects: {list(MANIM_OBJECTS.keys())}
5. Available animations: {list(MANIM_ANIMATIONS.keys())}
6. Available colors: {MANIM_COLORS}
7. Available positions: {MANIM_POSITIONS}

Common patterns:
- Use self.add() to add objects without animation
- Use self.play() for animations
- Use self.wait() for pauses
- Chain animations with comma separation in self.play()
- Set colors with .set_color()
- Position objects with .shift() or .move_to()

Always generate complete, runnable scripts. If the user request is unclear, make reasonable assumptions and create a working example.

Example script structure:
```python
from manim import *

class MyScene(Scene):
    def construct(self):
        # Create objects
        circle = Circle(radius=1)
        circle.set_color(BLUE)
        
        # Animate
        self.play(Create(circle))
        self.wait(1)
```

Respond with ONLY the Python code, no explanations unless requested."""
    
    def generate_script(self, user_request: str) -> str:
        """
        Generate a Manim script based on user request.
        
        Args:
            user_request: User's description of desired animation
            
        Returns:
            Complete Manim script as string
        """
        try:
            if self.llm_provider == "custom":
                # Use custom LLM
                result = self.inference_engine.generate_and_validate(user_request)
                return result['best_script']
            else:
                # Use external LLM API
                prompt = f"""Generate a complete Manim script for the following request:

{user_request}

Remember to:
1. Include proper imports
2. Create a Scene class with descriptive name
3. Use the construct() method
4. Make it a complete, runnable script
5. Add appropriate colors and animations

Script:"""

                # Call LLM
                response = self._call_llm(prompt)
                
                # Extract code from response
                script = self._extract_code(response)
                
                return script
            
        except Exception as e:
            return f"Error generating script: {str(e)}"
    
    def _call_llm(self, prompt: str) -> str:
        """Call the external LLM with the given prompt."""
        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1500,
                    temperature=0.7,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code between triple backticks
        code_pattern = r'```python\n(.*?)\n```'
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code blocks found, try to extract from response
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('from manim import') or line.strip().startswith('import'):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Return the full response if no code extraction worked
        return response.strip()
    
    def generate_with_template(self, pattern_name: str, **params) -> str:
        """
        Generate script using a predefined template.
        
        Args:
            pattern_name: Name of the pattern to use
            **params: Parameters for the template
            
        Returns:
            Complete Manim script
        """
        if pattern_name not in COMMON_PATTERNS:
            return f"Error: Pattern '{pattern_name}' not found"
        
        try:
            # Get the pattern template
            pattern_code = get_pattern_code(pattern_name, **params)
            
            # Generate scene name
            scene_name = params.get('scene_name', f'{pattern_name.title()}Scene')
            
            # Create full script
            script = get_full_script(scene_name, pattern_code)
            
            return script
            
        except Exception as e:
            return f"Error generating template script: {str(e)}"
    
    def get_example_script(self, example_name: str) -> str:
        """
        Get a predefined example script.
        
        Args:
            example_name: Name of the example
            
        Returns:
            Example script or error message
        """
        if example_name not in EXAMPLE_SCRIPTS:
            available = list(EXAMPLE_SCRIPTS.keys())
            return f"Error: Example '{example_name}' not found. Available: {available}"
        
        return EXAMPLE_SCRIPTS[example_name]
    
    def list_available_objects(self) -> Dict[str, str]:
        """Get list of available Manim objects with descriptions."""
        return {obj: info["description"] for obj, info in MANIM_OBJECTS.items()}
    
    def list_available_animations(self) -> Dict[str, str]:
        """Get list of available Manim animations with descriptions."""
        return {anim: info["description"] for anim, info in MANIM_ANIMATIONS.items()}
    
    def list_available_patterns(self) -> Dict[str, str]:
        """Get list of available patterns with descriptions."""
        return {pattern: info["description"] for pattern, info in COMMON_PATTERNS.items()}
    
    def improve_script(self, script: str, improvement_request: str) -> str:
        """
        Improve an existing Manim script based on feedback.
        
        Args:
            script: Existing Manim script
            improvement_request: Description of desired improvements
            
        Returns:
            Improved script
        """
        if self.llm_provider == "custom":
            # Use custom LLM - generate based on improvement request
            full_request = f"Improve this script: {script}\n\nImprovement: {improvement_request}"
            result = self.inference_engine.generate_and_validate(full_request)
            return result['best_script']
        else:
            # Use external LLM API
            prompt = f"""Improve the following Manim script based on this request:

Improvement Request: {improvement_request}

Current Script:
```python
{script}
```

Generate an improved version of the script. Keep the same structure but implement the requested improvements.

Improved Script:"""

            try:
                response = self._call_llm(prompt)
                improved_script = self._extract_code(response)
                return improved_script
            except Exception as e:
                return f"Error improving script: {str(e)}"
    
    def explain_script(self, script: str) -> str:
        """
        Generate explanation for a Manim script.
        
        Args:
            script: Manim script to explain
            
        Returns:
            Detailed explanation of the script
        """
        if self.llm_provider == "custom":
            # For custom LLM, generate a simple explanation based on patterns
            return self._generate_simple_explanation(script)
        else:
            # Use external LLM API
            prompt = f"""Explain the following Manim script in detail. Describe what each part does and what the final animation will look like:

```python
{script}
```

Provide a clear, educational explanation suitable for someone learning Manim."""

            try:
                response = self._call_llm(prompt)
                return response
            except Exception as e:
                return f"Error explaining script: {str(e)}"
    
    def _generate_simple_explanation(self, script: str) -> str:
        """Generate a simple explanation for the script."""
        explanation = []
        lines = script.split('\n')
        
        explanation.append("This Manim script contains:")
        
        # Check for imports
        if any('import' in line for line in lines):
            explanation.append("- Proper imports for Manim functionality")
        
        # Check for class definition
        for line in lines:
            if line.strip().startswith('class ') and 'Scene' in line:
                class_name = line.split('class ')[1].split('(')[0]
                explanation.append(f"- A scene class named '{class_name}'")
        
        # Check for objects
        for obj_type in MANIM_OBJECTS:
            if obj_type.title() in script:
                explanation.append(f"- {obj_type.title()} object creation")
        
        # Check for animations
        for anim_type in MANIM_ANIMATIONS:
            if anim_type.title() in script:
                explanation.append(f"- {anim_type.title()} animation")
        
        # Check for colors
        for color in MANIM_COLORS:
            if color in script:
                explanation.append(f"- Uses {color} color")
        
        if len(explanation) == 1:
            explanation.append("- Basic Manim structure")
        
        return '\n'.join(explanation)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        if self.llm_provider == "custom":
            return {
                "provider": "Custom LLM",
                "model": "Custom Manim-trained transformer",
                "status": "Local model - no API keys required"
            }
        else:
            return {
                "provider": self.llm_provider,
                "model": self.model_name,
                "status": "External API"
            }