"""
Manim Knowledge Base - Contains templates, patterns, and examples for generating Manim scripts.
"""

MANIM_IMPORTS = """from manim import *
import numpy as np"""

BASIC_SCENE_TEMPLATE = """class {scene_name}(Scene):
    def construct(self):
        {content}"""

MANIM_OBJECTS = {
    "circle": {
        "code": "Circle(radius={radius})",
        "default_params": {"radius": 1},
        "description": "Creates a circle with specified radius"
    },
    "square": {
        "code": "Square(side_length={side_length})",
        "default_params": {"side_length": 2},
        "description": "Creates a square with specified side length"
    },
    "text": {
        "code": "Text(\"{text}\")",
        "default_params": {"text": "Hello World"},
        "description": "Creates text object"
    },
    "mathtext": {
        "code": "MathTex(\"{math}\")",
        "default_params": {"math": "x^2 + y^2 = r^2"},
        "description": "Creates mathematical text using LaTeX"
    },
    "line": {
        "code": "Line(start={start}, end={end})",
        "default_params": {"start": "LEFT", "end": "RIGHT"},
        "description": "Creates a line from start to end point"
    },
    "arrow": {
        "code": "Arrow(start={start}, end={end})",
        "default_params": {"start": "LEFT", "end": "RIGHT"},
        "description": "Creates an arrow from start to end point"
    },
    "rectangle": {
        "code": "Rectangle(width={width}, height={height})",
        "default_params": {"width": 3, "height": 2},
        "description": "Creates a rectangle with specified dimensions"
    },
    "dot": {
        "code": "Dot(point={point})",
        "default_params": {"point": "ORIGIN"},
        "description": "Creates a dot at specified point"
    }
}

MANIM_ANIMATIONS = {
    "create": {
        "code": "Create({object})",
        "default_params": {},
        "description": "Creates object by drawing it"
    },
    "write": {
        "code": "Write({object})",
        "default_params": {},
        "description": "Writes text or formula"
    },
    "fadein": {
        "code": "FadeIn({object})",
        "default_params": {},
        "description": "Fades in object"
    },
    "fadeout": {
        "code": "FadeOut({object})",
        "default_params": {},
        "description": "Fades out object"
    },
    "transform": {
        "code": "Transform({object1}, {object2})",
        "default_params": {"object1": "{object}", "object2": "{object}"},
        "description": "Transforms one object into another"
    },
    "rotate": {
        "code": "Rotate({object}, angle={angle})",
        "default_params": {"angle": "PI/2"},
        "description": "Rotates object by specified angle"
    },
    "scale": {
        "code": "Scale({object}, factor={factor})",
        "default_params": {"factor": "2"},
        "description": "Scales object by factor"
    },
    "move": {
        "code": "{object}.animate.shift({direction})",
        "default_params": {"direction": "RIGHT"},
        "description": "Moves object in specified direction"
    }
}

MANIM_COLORS = [
    "RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "PINK", 
    "WHITE", "BLACK", "GRAY", "LIGHT_GRAY", "DARK_GRAY"
]

MANIM_POSITIONS = [
    "UP", "DOWN", "LEFT", "RIGHT", "ORIGIN",
    "UL", "UR", "DL", "DR"  # Upper-left, Upper-right, Down-left, Down-right
]

COMMON_PATTERNS = {
    "simple_shape": {
        "description": "Create and display a simple shape",
        "template": """        # Create a {shape}
        {object_name} = {shape_code}
        {object_name}.set_color({color})
        
        # Add to scene
        self.add({object_name})
        self.wait(1)"""
    },
    "animated_creation": {
        "description": "Create and animate a shape",
        "template": """        # Create a {shape}
        {object_name} = {shape_code}
        {object_name}.set_color({color})
        
        # Animate creation
        self.play(Create({object_name}))
        self.wait(1)"""
    },
    "text_animation": {
        "description": "Create and animate text",
        "template": """        # Create text
        {object_name} = Text("{text}")
        {object_name}.set_color({color})
        
        # Animate text
        self.play(Write({object_name}))
        self.wait(1)"""
    },
    "mathematical_formula": {
        "description": "Display mathematical formula",
        "template": """        # Create mathematical formula
        {object_name} = MathTex("{formula}")
        {object_name}.set_color({color})
        
        # Display formula
        self.play(Write({object_name}))
        self.wait(2)"""
    },
    "transformation": {
        "description": "Transform one shape into another",
        "template": """        # Create first shape
        {object1} = {shape1_code}
        {object1}.set_color({color1})
        
        # Create second shape
        {object2} = {shape2_code}
        {object2}.set_color({color2})
        
        # Show first shape
        self.play(Create({object1}))
        self.wait(1)
        
        # Transform to second shape
        self.play(Transform({object1}, {object2}))
        self.wait(1)"""
    },
    "multiple_objects": {
        "description": "Create multiple objects with positioning",
        "template": """        # Create multiple objects
        {object1} = {shape1_code}
        {object1}.set_color({color1})
        {object1}.shift({position1})
        
        {object2} = {shape2_code}
        {object2}.set_color({color2})
        {object2}.shift({position2})
        
        # Animate all objects
        self.play(Create({object1}), Create({object2}))
        self.wait(1)"""
    }
}

EXAMPLE_SCRIPTS = {
    "basic_circle": """from manim import *

class BasicCircle(Scene):
    def construct(self):
        # Create a circle
        circle = Circle(radius=2)
        circle.set_color(BLUE)
        
        # Animate the circle
        self.play(Create(circle))
        self.wait(1)""",
    
    "text_example": """from manim import *

class TextExample(Scene):
    def construct(self):
        # Create text
        text = Text("Hello, Manim!")
        text.set_color(RED)
        
        # Animate text
        self.play(Write(text))
        self.wait(2)""",
    
    "math_formula": """from manim import *

class MathFormula(Scene):
    def construct(self):
        # Create mathematical formula
        formula = MathTex("E = mc^2")
        formula.set_color(GREEN)
        
        # Display formula
        self.play(Write(formula))
        self.wait(2)""",
    
    "shape_transformation": """from manim import *

class ShapeTransformation(Scene):
    def construct(self):
        # Create a circle
        circle = Circle(radius=1)
        circle.set_color(BLUE)
        
        # Create a square
        square = Square(side_length=2)
        square.set_color(RED)
        
        # Show circle first
        self.play(Create(circle))
        self.wait(1)
        
        # Transform circle to square
        self.play(Transform(circle, square))
        self.wait(1)"""
}

def get_object_code(obj_type, **params):
    """Generate code for a Manim object with given parameters."""
    if obj_type not in MANIM_OBJECTS:
        return None
    
    obj_info = MANIM_OBJECTS[obj_type]
    merged_params = {**obj_info["default_params"], **params}
    
    return obj_info["code"].format(**merged_params)

def get_animation_code(anim_type, obj_name, **params):
    """Generate code for a Manim animation with given parameters."""
    if anim_type not in MANIM_ANIMATIONS:
        return None
    
    anim_info = MANIM_ANIMATIONS[anim_type]
    # Merge default parameters with provided parameters
    merged_params = {**anim_info["default_params"], **params}
    
    return anim_info["code"].format(object=obj_name, **merged_params)

def get_pattern_code(pattern_name, **params):
    """Generate code for a common Manim pattern."""
    if pattern_name not in COMMON_PATTERNS:
        return None
    
    pattern = COMMON_PATTERNS[pattern_name]
    return pattern["template"].format(**params)

def get_full_script(scene_name, content):
    """Generate a complete Manim script."""
    return f"""{MANIM_IMPORTS}

{BASIC_SCENE_TEMPLATE.format(scene_name=scene_name, content=content)}"""