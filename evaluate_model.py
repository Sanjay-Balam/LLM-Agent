#!/usr/bin/env python3
"""
Model Evaluation Script for Custom Manim LLM
Evaluate the trained model on various test cases.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import ManimInferenceEngine
from validator import ManimScriptValidator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Custom Manim LLM')
    
    parser.add_argument('--model-path', type=str, default='best_model_epoch_10.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer-path', type=str, default='tokenizer.pkl',
                        help='Path to tokenizer file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive evaluation')
    
    return parser.parse_args()

class ModelEvaluator:
    """Evaluator for the trained Manim LLM."""
    
    def __init__(self, model_path: str, tokenizer_path: str, output_dir: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize inference engine
        self.engine = ManimInferenceEngine(model_path, tokenizer_path)
        self.validator = ManimScriptValidator()
        
        # Test cases
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases."""
        return [
            {
                "category": "Basic Shapes",
                "requests": [
                    "Create a blue circle",
                    "Make a red square",
                    "Draw a green triangle",
                    "Show a yellow rectangle",
                    "Create a purple dot"
                ]
            },
            {
                "category": "Animations",
                "requests": [
                    "Create a circle that appears with animation",
                    "Make a square that fades in",
                    "Show a line that draws itself",
                    "Create text that writes itself",
                    "Make a shape that rotates"
                ]
            },
            {
                "category": "Transformations",
                "requests": [
                    "Create a circle that transforms into a square",
                    "Make a small dot that grows into a large circle",
                    "Show a line that becomes an arrow",
                    "Create a square that rotates 90 degrees",
                    "Make a shape that moves from left to right"
                ]
            },
            {
                "category": "Text and Math",
                "requests": [
                    "Show the text 'Hello World'",
                    "Create the mathematical formula E=mc²",
                    "Display the equation x² + y² = r²",
                    "Show the formula for the area of a circle",
                    "Create a title that says 'Manim Animation'"
                ]
            },
            {
                "category": "Complex Scenes",
                "requests": [
                    "Create a coordinate system with axes",
                    "Show multiple circles in a line",
                    "Create a bouncing ball animation",
                    "Make a pendulum that swings back and forth",
                    "Show a sine wave that draws itself"
                ]
            },
            {
                "category": "Colors and Positioning",
                "requests": [
                    "Create a red circle in the top left corner",
                    "Make a blue square that moves to the center",
                    "Show three colored dots in a triangle formation",
                    "Create a rainbow of colored circles",
                    "Make a gradient from red to blue"
                ]
            }
        ]
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate model on all test cases."""
        print("Starting comprehensive evaluation...")
        
        all_results = {}
        total_score = 0
        total_tests = 0
        
        for category_data in self.test_cases:
            category = category_data["category"]
            requests = category_data["requests"]
            
            print(f"\nEvaluating {category}...")
            category_results = []
            category_score = 0
            
            for request in requests:
                print(f"  Testing: {request}")
                result = self.engine.generate_and_validate(request)
                
                category_results.append({
                    "request": request,
                    "script": result['best_script'],
                    "score": result['best_score'],
                    "validation": result['validation_report']
                })
                
                category_score += result['best_score']
                total_score += result['best_score']
                total_tests += 1
            
            avg_category_score = category_score / len(requests)
            all_results[category] = {
                "results": category_results,
                "average_score": avg_category_score,
                "total_tests": len(requests)
            }
            
            print(f"  {category} average score: {avg_category_score:.1f}")
        
        # Overall statistics
        overall_avg = total_score / total_tests
        all_results["overall"] = {
            "average_score": overall_avg,
            "total_tests": total_tests,
            "total_score": total_score
        }
        
        print(f"\nOverall average score: {overall_avg:.1f}")
        return all_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        # Save JSON results
        json_file = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_file}")
        
        # Save detailed text report
        text_file = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(text_file, 'w') as f:
            f.write("MANIM LLM EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall stats
            overall = results["overall"]
            f.write(f"Overall Statistics:\n")
            f.write(f"  Total Tests: {overall['total_tests']}\n")
            f.write(f"  Average Score: {overall['average_score']:.1f}\n")
            f.write(f"  Total Score: {overall['total_score']:.1f}\n\n")
            
            # Category results
            for category, data in results.items():
                if category == "overall":
                    continue
                
                f.write(f"{category.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Score: {data['average_score']:.1f}\n")
                f.write(f"Tests: {data['total_tests']}\n\n")
                
                for i, result in enumerate(data['results']):
                    f.write(f"Test {i+1}: {result['request']}\n")
                    f.write(f"Score: {result['score']:.1f}\n")
                    f.write("Generated Script:\n")
                    f.write("-" * 20 + "\n")
                    f.write(result['script'])
                    f.write("\n" + "-" * 20 + "\n")
                    f.write("Validation Report:\n")
                    f.write(result['validation'])
                    f.write("\n" + "=" * 40 + "\n\n")
        
        print(f"Detailed report saved to {text_file}")
    
    def plot_results(self, results: Dict[str, Any]) -> None:
        """Plot evaluation results."""
        categories = []
        scores = []
        
        for category, data in results.items():
            if category == "overall":
                continue
            categories.append(category)
            scores.append(data['average_score'])
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(categories, scores, color='skyblue', edgecolor='navy')
        plt.title('Model Performance by Category', fontsize=16, fontweight='bold')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add overall average line
        overall_avg = results["overall"]["average_score"]
        plt.axhline(y=overall_avg, color='red', linestyle='--', 
                   label=f'Overall Average: {overall_avg:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'evaluation_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved to {plot_file}")
    
    def create_score_distribution(self, results: Dict[str, Any]) -> None:
        """Create score distribution plot."""
        all_scores = []
        
        for category, data in results.items():
            if category == "overall":
                continue
            for result in data['results']:
                all_scores.append(result['score'])
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title('Score Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        plt.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.1f}')
        plt.axvline(mean_score + std_score, color='orange', linestyle='--', 
                   label=f'Mean + Std: {mean_score + std_score:.1f}')
        plt.axvline(mean_score - std_score, color='orange', linestyle='--', 
                   label=f'Mean - Std: {mean_score - std_score:.1f}')
        plt.legend()
        
        plt.tight_layout()
        dist_file = os.path.join(self.output_dir, 'score_distribution.png')
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Distribution plot saved to {dist_file}")
    
    def interactive_evaluation(self) -> None:
        """Run interactive evaluation."""
        print("Interactive Model Evaluation")
        print("=" * 40)
        print("Enter requests to test the model (type 'quit' to exit)")
        
        while True:
            request = input("\nEnter request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                break
            
            if not request:
                continue
            
            print("Generating and evaluating...")
            result = self.engine.generate_and_validate(request)
            
            print(f"\nScore: {result['best_score']:.1f}")
            print("Generated Script:")
            print("-" * 30)
            print(result['best_script'])
            print("-" * 30)
            print("Validation Report:")
            print(result['validation_report'])
            
            # Save individual result
            save = input("\nSave this result? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                filename = input("Enter filename (without extension): ").strip()
                if filename:
                    result_file = os.path.join(self.output_dir, f"{filename}.py")
                    with open(result_file, 'w') as f:
                        f.write(result['best_script'])
                    print(f"Script saved to {result_file}")

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check if model files exist
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please train the model first using train_model.py")
        return
    
    if not os.path.exists(args.tokenizer_path):
        print(f"Tokenizer file not found: {args.tokenizer_path}")
        print("Please train the model first using train_model.py")
        return
    
    print("=" * 60)
    print("CUSTOM MANIM LLM EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.tokenizer_path, args.output_dir)
    
    if args.interactive:
        # Interactive mode
        evaluator.interactive_evaluation()
    else:
        # Comprehensive evaluation
        results = evaluator.evaluate_all()
        
        # Save results
        evaluator.save_results(results)
        
        # Create plots
        evaluator.plot_results(results)
        evaluator.create_score_distribution(results)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED!")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        print(f"Overall average score: {results['overall']['average_score']:.1f}")
        print("=" * 60)

if __name__ == "__main__":
    main()