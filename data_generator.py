"""
Training Data Generator for Custom Manim LLM
Generates synthetic training data for Manim script generation.
"""

import random
import json
from typing import List, Dict, Tuple
from knowledge_base import (
    MANIM_OBJECTS, MANIM_ANIMATIONS, MANIM_COLORS, MANIM_POSITIONS,
    COMMON_PATTERNS, EXAMPLE_SCRIPTS, get_object_code, get_animation_code,
    get_pattern_code, get_full_script
)

class ManimDataGenerator:
    """Generates training data for Manim script generation."""
    
    def __init__(self):
        self.object_names = list(MANIM_OBJECTS.keys())
        self.animation_names = list(MANIM_ANIMATIONS.keys())
        self.colors = MANIM_COLORS
        self.positions = MANIM_POSITIONS
        
        # Common request patterns
        self.request_templates = [
            "Create a {color} {object}",
            "Make a {object} that {animation}",
            "Show a {object} with {color} color",
            "Animate a {object} using {animation}",
            "Draw a {color} {object} and make it {animation}",
            "Create {object} and {object2} with different colors",
            "Make a {object} transform into a {object2}",
            "Show text that says '{text}'",
            "Create a mathematical formula {formula}",
            "Make a {object} move from {position} to {position2}",
            "Create multiple {object}s in a line",
            "Show a {object} that rotates",
            "Make a {object} that scales up and down",
            "Create a {object} that fades in and out",
            "Show a coordinate system with a {object}",
        ]
        
        # Sample texts and formulas
        self.sample_texts = [
            "Hello World", "Manim Animation", "Python Code", "Mathematics",
            "Learning", "Education", "Science", "Technology"
        ]
        
        self.sample_formulas = [
            "x^2 + y^2 = r^2", "E = mc^2", "F = ma", "a^2 + b^2 = c^2",
            "\\sin^2(x) + \\cos^2(x) = 1", "e^{i\\pi} + 1 = 0",
            "\\frac{d}{dx}x^n = nx^{n-1}", "\\int_0^1 x dx = \\frac{1}{2}"
        ]
    
    def generate_single_request_response(self) -> Tuple[str, str]:
        """Generate a single request-response pair."""
        template = random.choice(self.request_templates)
        
        # Fill template with random values
        request = template.format(
            object=random.choice(self.object_names),
            object2=random.choice(self.object_names),
            animation=random.choice(self.animation_names),
            color=random.choice(self.colors),
            position=random.choice(self.positions),
            position2=random.choice(self.positions),
            text=random.choice(self.sample_texts),
            formula=random.choice(self.sample_formulas)
        )
        
        # Generate corresponding script
        script = self._generate_script_for_request(request)
        
        return request, script
    
    def _generate_script_for_request(self, request: str) -> str:
        """Generate a Manim script based on the request."""
        # Simple rule-based generation
        request_lower = request.lower()
        
        # Determine what to create
        obj_type = None
        for obj in self.object_names:
            if obj in request_lower:
                obj_type = obj
                break
        
        if not obj_type:
            obj_type = random.choice(self.object_names)
        
        # Determine color
        color = "BLUE"
        for c in self.colors:
            if c.lower() in request_lower:
                color = c
                break
        
        # Determine animation
        animation = "create"
        for anim in self.animation_names:
            if anim in request_lower:
                animation = anim
                break
        
        # Generate object code
        obj_code = get_object_code(obj_type)
        if not obj_code:
            obj_code = f"{obj_type.title()}()"
        
        # Generate scene content
        obj_name = f"{obj_type}_obj"
        
        if "text" in request_lower:
            # Handle text requests
            text = "Hello World"
            for sample in self.sample_texts:
                if sample.lower() in request_lower:
                    text = sample
                    break
            
            content = f'''        # Create text
        {obj_name} = Text("{text}")
        {obj_name}.set_color({color})
        
        # Animate text
        self.play(Write({obj_name}))
        self.wait(1)'''
        
        elif "formula" in request_lower or "mathematical" in request_lower:
            # Handle formula requests
            formula = "x^2 + y^2 = r^2"
            for sample in self.sample_formulas:
                if any(part in request_lower for part in sample.split()):
                    formula = sample
                    break
            
            content = f'''        # Create mathematical formula
        {obj_name} = MathTex("{formula}")
        {obj_name}.set_color({color})
        
        # Display formula
        self.play(Write({obj_name}))
        self.wait(2)'''
        
        elif "transform" in request_lower:
            # Handle transformation requests
            obj2_type = random.choice([o for o in self.object_names if o != obj_type])
            obj2_code = get_object_code(obj2_type)
            obj2_name = f"{obj2_type}_obj"
            color2 = random.choice([c for c in self.colors if c != color])
            
            content = f'''        # Create first {obj_type}
        {obj_name} = {obj_code}
        {obj_name}.set_color({color})
        
        # Create second {obj2_type}
        {obj2_name} = {obj2_code}
        {obj2_name}.set_color({color2})
        
        # Show first shape
        self.play(Create({obj_name}))
        self.wait(1)
        
        # Transform to second shape
        self.play(Transform({obj_name}, {obj2_name}))
        self.wait(1)'''
        
        else:
            # Handle simple object requests
            anim_code = get_animation_code(animation, obj_name)
            if not anim_code:
                anim_code = f"Create({obj_name})"
            
            content = f'''        # Create a {obj_type}
        {obj_name} = {obj_code}
        {obj_name}.set_color({color})
        
        # Animate {obj_type}
        self.play({anim_code})
        self.wait(1)'''
        
        # Generate full script
        scene_name = f"{obj_type.title()}Scene"
        script = get_full_script(scene_name, content)
        
        return script
    
    def generate_training_data(self, num_samples: int = 1000) -> List[Dict]:
        """Generate training dataset."""
        training_data = []
        
        print(f"Generating {num_samples} training samples...")
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples")
            
            request, script = self.generate_single_request_response()
            
            training_data.append({
                "id": i,
                "request": request,
                "script": script,
                "length": len(script)
            })
        
        print(f"Generated {len(training_data)} training samples")
        return training_data
    
    def save_training_data(self, data: List[Dict], filename: str = "training_data.json"):
        """Save training data to file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Training data saved to {filename}")
    
    def load_training_data(self, filename: str = "training_data.json") -> List[Dict]:
        """Load training data from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training samples from {filename}")
        return data
    
    def generate_and_save_dataset(self, num_samples: int = 1000, filename: str = "training_data.json"):
        """Generate and save training dataset."""
        # Add existing examples to training data
        training_data = []
        
        # Add example scripts
        for name, script in EXAMPLE_SCRIPTS.items():
            # Generate request for example
            request = f"Create a {name.replace('_', ' ')} animation"
            training_data.append({
                "id": len(training_data),
                "request": request,
                "script": script,
                "length": len(script),
                "source": "example"
            })
        
        # Generate synthetic data
        synthetic_data = self.generate_training_data(num_samples)
        training_data.extend(synthetic_data)
        
        # Save to file
        self.save_training_data(training_data, filename)
        
        return training_data
    
    def create_validation_split(self, data: List[Dict], split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and validation sets."""
        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        return train_data, val_data

if __name__ == "__main__":
    generator = ManimDataGenerator()
    
    # Generate dataset
    data = generator.generate_and_save_dataset(num_samples=2000)
    
    # Create train/validation split
    train_data, val_data = generator.create_validation_split(data)
    
    # Save splits
    generator.save_training_data(train_data, "train_data.json")
    generator.save_training_data(val_data, "val_data.json")
    
    print("Dataset generation complete!")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")