# Custom Manim LLM Agent

A complete AI system for generating Python Manim scripts using your own custom-trained Language Model. No external API keys required!

## Features

- 🧠 **Custom LLM**: Train your own transformer model specifically for Manim
- 📝 **Natural Language**: Generate scripts from simple text descriptions
- 🎨 **Template System**: Pre-built patterns for common animations
- ✅ **Script Validation**: Automatic syntax and structure checking
- 🔧 **Auto-fixing**: Attempts to fix common script issues
- 📚 **Knowledge Base**: Comprehensive Manim patterns and examples
- 🏠 **Fully Local**: No external API dependencies
- 🎯 **Specialized**: Trained specifically for Manim script generation

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. No API keys needed! Your model runs locally.

## Quick Start

### 1. Train Your Model

First, train your custom LLM:

```bash
python train_model.py --epochs 20 --batch-size 4
```

This will:
- Generate 3000 training samples
- Build a specialized tokenizer
- Train a transformer model
- Save the trained model and tokenizer

### 2. Use Your Custom Agent

```python
from agent import ManimAgent

# Initialize with your custom model (no API keys needed!)
agent = ManimAgent(llm_provider="custom")

# Generate a script
request = "Create a blue circle that transforms into a red square"
script = agent.generate_script(request)
print(script)
```

### 3. Alternative: Use External APIs (Optional)

If you prefer external APIs:

```python
# With OpenAI (requires API key)
agent = ManimAgent(llm_provider="openai", api_key="your_key")

# With Anthropic (requires API key)
agent = ManimAgent(llm_provider="anthropic", api_key="your_key")
```

## Training Your Model

### Quick Training

```bash
python train_model.py
```

### Custom Training

```bash
python train_model.py \
    --data-size 5000 \
    --epochs 30 \
    --batch-size 8 \
    --d-model 768 \
    --n-layers 8
```

### Training Options

- `--data-size`: Number of training samples (default: 3000)
- `--epochs`: Training epochs (default: 20)
- `--batch-size`: Batch size (default: 4)
- `--d-model`: Model dimension (default: 512)
- `--n-layers`: Number of layers (default: 6)
- `--learning-rate`: Learning rate (default: 1e-4)

## Model Evaluation

Evaluate your trained model:

```bash
python evaluate_model.py --model-path best_model_epoch_10.pth
```

Interactive evaluation:

```bash
python evaluate_model.py --interactive
```

## Project Structure

```
manim_agent/
├── agent.py           # Main ManimAgent class (supports custom LLM)
├── model.py           # Custom transformer architecture
├── tokenizer.py       # Manim-specific tokenizer
├── trainer.py         # Training pipeline
├── inference.py       # Inference engine
├── data_generator.py  # Training data generation
├── knowledge_base.py  # Manim patterns and templates
├── validator.py       # Script validation logic
├── train_model.py     # Training script
├── evaluate_model.py  # Evaluation script
├── examples.py        # Usage examples
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## API Reference

### ManimAgent Class

#### Initialization

```python
# Custom LLM (recommended)
agent = ManimAgent(llm_provider="custom")

# External APIs (optional)
agent = ManimAgent(llm_provider="openai", api_key="your_key")
agent = ManimAgent(llm_provider="anthropic", api_key="your_key")
```

#### Methods

- `generate_script(request: str) -> str`: Generate script from natural language
- `generate_with_template(pattern: str, **params) -> str`: Use predefined templates
- `improve_script(script: str, request: str) -> str`: Improve existing scripts
- `explain_script(script: str) -> str`: Get script explanations
- `list_available_objects() -> dict`: List available Manim objects
- `list_available_animations() -> dict`: List available animations
- `get_example_script(name: str) -> str`: Get predefined examples
- `get_model_info() -> dict`: Get information about current model

### Custom Model Components

#### ManimLLM
- Custom transformer architecture
- Specialized for Manim script generation
- Configurable size and complexity

#### ManimTokenizer
- Manim-specific vocabulary
- Handles Python code and Manim syntax
- Optimized for mathematical expressions

#### ManimInferenceEngine
- High-level inference interface
- Built-in validation and scoring
- Configurable generation parameters

## Model Architecture

### Transformer Specifications
- **Architecture**: Decoder-only transformer
- **Default Size**: 512d, 6 layers, 8 heads
- **Vocabulary**: ~8000 tokens (Manim-specific)
- **Max Length**: 1024 tokens
- **Parameters**: ~25M (configurable)

### Training Details
- **Data**: Synthetic Manim scripts + examples
- **Optimizer**: AdamW with cosine scheduling
- **Loss**: Cross-entropy with padding mask
- **Hardware**: CPU/GPU support with auto-detection

## Available Manim Objects

- Circle, Square, Rectangle
- Text, MathTex
- Line, Arrow
- Dot

## Available Animations

- Create, Write
- FadeIn, FadeOut
- Transform, Rotate, Scale
- Move

## Example Requests

Try these natural language requests:

- "Create a red circle that bounces up and down"
- "Show the Pythagorean theorem with a visual proof"
- "Make a sine wave animation with a moving dot"
- "Create a coordinate system with a parabola y = x²"
- "Animate the transformation of a triangle into a square"

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Make sure your API keys are in environment variables
2. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
3. **Script Validation Fails**: Use the validator to identify and fix issues

### Getting Better Results

1. **Be Specific**: More detailed requests produce better scripts
2. **Use Examples**: Reference the example scripts for inspiration
3. **Iterate**: Use the improve_script function to refine results

## Contributing

Feel free to extend the knowledge base with new patterns, improve the validation logic, or add support for additional LLM providers.

## License

This project is open source. Feel free to use and modify for your needs.