"""
Script Validator - Validates generated Manim scripts for syntax and structure.
"""

import ast
import re
from typing import List, Dict, Tuple, Optional

class ManimScriptValidator:
    """Validates Manim scripts for correctness and common issues."""
    
    def __init__(self):
        self.required_imports = ["from manim import *", "import manim"]
        self.required_methods = ["construct"]
        self.scene_classes = ["Scene", "MovingCameraScene", "ZoomedScene", "ThreeDScene"]
        
    def validate_script(self, script: str) -> Dict[str, any]:
        """
        Validate a Manim script for syntax and structure.
        
        Args:
            script: Python script to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check syntax
        syntax_result = self._check_syntax(script)
        if not syntax_result["is_valid"]:
            results["is_valid"] = False
            results["errors"].extend(syntax_result["errors"])
        
        # Check structure
        structure_result = self._check_structure(script)
        if not structure_result["is_valid"]:
            results["is_valid"] = False
            results["errors"].extend(structure_result["errors"])
        
        results["warnings"].extend(structure_result["warnings"])
        
        # Check Manim-specific issues
        manim_result = self._check_manim_specifics(script)
        results["warnings"].extend(manim_result["warnings"])
        results["suggestions"].extend(manim_result["suggestions"])
        
        return results
    
    def _check_syntax(self, script: str) -> Dict[str, any]:
        """Check Python syntax validity."""
        result = {"is_valid": True, "errors": []}
        
        try:
            ast.parse(script)
        except SyntaxError as e:
            result["is_valid"] = False
            result["errors"].append(f"Syntax Error: {e.msg} at line {e.lineno}")
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(f"Parse Error: {str(e)}")
        
        return result
    
    def _check_structure(self, script: str) -> Dict[str, any]:
        """Check script structure for Manim requirements."""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        # Check for imports
        has_manim_import = any(imp in script for imp in self.required_imports)
        if not has_manim_import:
            result["errors"].append("Missing required import: 'from manim import *'")
            result["is_valid"] = False
        
        # Check for Scene class
        has_scene_class = False
        scene_class_pattern = r'class\s+(\w+)\s*\(\s*(\w+)\s*\):'
        matches = re.findall(scene_class_pattern, script)
        
        for class_name, parent_class in matches:
            if parent_class in self.scene_classes:
                has_scene_class = True
                break
        
        if not has_scene_class:
            result["errors"].append("No Scene class found. Must inherit from Scene or similar.")
            result["is_valid"] = False
        
        # Check for construct method
        if "def construct(self):" not in script:
            result["errors"].append("Missing 'construct' method in Scene class")
            result["is_valid"] = False
        
        # Check for proper indentation (basic check)
        lines = script.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                if line.startswith('class ') or line.startswith('def ') or line.startswith('import ') or line.startswith('from '):
                    continue
                if i > 1:  # Skip first line
                    result["warnings"].append(f"Line {i} might have indentation issues")
        
        return result
    
    def _check_manim_specifics(self, script: str) -> Dict[str, any]:
        """Check for Manim-specific best practices and common issues."""
        result = {"warnings": [], "suggestions": []}
        
        # Check for self.play() usage
        if "self.play(" not in script:
            result["suggestions"].append("Consider using self.play() for animations")
        
        # Check for self.wait() usage
        if "self.wait(" not in script:
            result["suggestions"].append("Consider adding self.wait() for pauses between animations")
        
        # Check for color usage
        color_pattern = r'\.set_color\('
        if not re.search(color_pattern, script):
            result["suggestions"].append("Consider adding colors to objects with .set_color()")
        
        # Check for object creation
        common_objects = ["Circle", "Square", "Text", "MathTex", "Line", "Arrow"]
        has_objects = any(obj in script for obj in common_objects)
        if not has_objects:
            result["warnings"].append("No common Manim objects found")
        
        # Check for animation methods
        common_animations = ["Create", "Write", "FadeIn", "FadeOut", "Transform"]
        has_animations = any(anim in script for anim in common_animations)
        if not has_animations:
            result["warnings"].append("No common animations found")
        
        # Check for proper Scene naming
        scene_pattern = r'class\s+(\w+)Scene\s*\('
        if not re.search(scene_pattern, script):
            result["suggestions"].append("Consider naming Scene classes with 'Scene' suffix")
        
        return result
    
    def fix_common_issues(self, script: str) -> str:
        """Attempt to fix common issues in the script."""
        fixed_script = script
        
        # Add missing imports if needed
        if "from manim import *" not in fixed_script and "import manim" not in fixed_script:
            fixed_script = "from manim import *\n\n" + fixed_script
        
        # Fix common indentation issues (basic)
        lines = fixed_script.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # Ensure proper indentation for class contents
            if line.strip().startswith('def construct(self):'):
                fixed_lines.append('    ' + line.strip())
            elif line.strip().startswith('self.'):
                fixed_lines.append('        ' + line.strip())
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def get_validation_report(self, script: str) -> str:
        """Get a formatted validation report."""
        results = self.validate_script(script)
        
        report = ["=" * 50]
        report.append("MANIM SCRIPT VALIDATION REPORT")
        report.append("=" * 50)
        
        if results["is_valid"]:
            report.append("✅ SCRIPT IS VALID")
        else:
            report.append("❌ SCRIPT HAS ERRORS")
        
        if results["errors"]:
            report.append("\n🚨 ERRORS:")
            for error in results["errors"]:
                report.append(f"  - {error}")
        
        if results["warnings"]:
            report.append("\n⚠️  WARNINGS:")
            for warning in results["warnings"]:
                report.append(f"  - {warning}")
        
        if results["suggestions"]:
            report.append("\n💡 SUGGESTIONS:")
            for suggestion in results["suggestions"]:
                report.append(f"  - {suggestion}")
        
        report.append("=" * 50)
        
        return '\n'.join(report)
    
    def validate_and_fix(self, script: str) -> Tuple[bool, str, str]:
        """
        Validate script and attempt to fix issues.
        
        Args:
            script: Original script
            
        Returns:
            Tuple of (is_valid, fixed_script, report)
        """
        # First validation
        results = self.validate_script(script)
        
        if results["is_valid"]:
            return True, script, self.get_validation_report(script)
        
        # Try to fix common issues
        fixed_script = self.fix_common_issues(script)
        
        # Validate again
        new_results = self.validate_script(fixed_script)
        
        # Generate report
        report = self.get_validation_report(fixed_script)
        
        return new_results["is_valid"], fixed_script, report