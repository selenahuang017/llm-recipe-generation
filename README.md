# LLM Recipe Generation

## Overview

This is a meal generation agentic model that uses LLMs to create personalized meal plans. The system generates breakfast, lunch, and dinner recipes based on provided ingredients, dietary restrictions, and constraints. It employs an iterative validation and refinement process to ensure the generated meal plans meet all specified requirements.

## Setup

Install the package with `pip install .`. Run the main module with `python -m modules.main` along with required arguments (see example below).

## Models

The system uses models from Hugging Face accessed via Ollama API. Available models include `gpt-oss:20b`, `gpt-oss:120b`, and `gemma3:27b`. The model can be specified using the `--model` argument, and a separate validation model can be set with `--val_model`.

## Validation Checks

The validator performs several checks on generated meal plans:
- **Meal plan structure**: Ensures breakfast, lunch, and dinner sections are present
- **Recipe format**: Verifies each recipe contains title, serving size, ingredients, instructions, nutritional information, and additional notes
- **Ingredients**: Confirms all required ingredients are used at least once across the three meals
- **Allergens**: Checks that specified allergens are not present in any recipe
- **Nutrition**: Validates nutritional information (calories, protein, fat, carbs, fiber) against calculated values and optional calorie targets
- **Budget**: Optionally checks that total ingredient costs stay within a specified budget

## Example

**Input:**
```bash
python -m modules.main --model gpt-oss:20b --max_iterations 1 --ingredients 'rhubarb, rice, duck, tomatoes, sesame seeds, persimmon, and oreos' --allergens gluten --budget 50 --calories 1500 --verbose
```

**Output:**
The system generates a structured meal plan with three recipes (breakfast, lunch, dinner), each containing:
1. Title
2. Serving size
3. Ingredients and amounts
4. Instructions
5. Nutritional information
6. Additional notes

See `example.txt` for a complete example output.
