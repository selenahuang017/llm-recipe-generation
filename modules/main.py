import argparse
import os
from pathlib import Path

from .Ollama import OllamaChatSession
from .Validator import Validator

def load_secrets(secrets_file: str = '.secrets'):
    """
    Load secrets from a file into os.environ.
    Expected format: KEY=VALUE (one per line)
    """
    secrets_path = Path(secrets_file)
    if not secrets_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        secrets_path = project_root / secrets_file
    
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    else:
        # Silently fail if secrets file doesn't exist
        # (user might be setting env vars another way)
        pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Create a meal plan with the following ingredients:')
    p.add_argument(
        '--model',
        type=str,
        choices=['gpt-oss:20b', 'gpt-oss:120b', 'gemma3:27b',],
        required=True,
        default='gpt-oss:20b',
        help='define model to ask one of them',
    )
    p.add_argument(
        '--max_iterations',
        type=int,
        required=False,
        default=5,
        help='max iterations before exiting',
    )
    p.add_argument(
        '--ingredients',
        type=str,
        required=True,
        help='add ingredients as a comma separated list',
    )
    p.add_argument(
        '--allergens',
        type=str,
        required=False,
        help='add allergens, optional',
    )
    p.add_argument(
        '--budget',
        type=float,
        required=False,
        help='budget amount for the meal plan',
    )
    p.add_argument(
        '--check_budget',
        action='store_true',
        default=False,
        required=False,
        help='whether to check budget',
    )
    p.add_argument(
        '--verbose',
        action='store_true',
        required=False,
        help='verbosity, includes debug information',
    )
    p.add_argument(
        '--val_model',
        type=str,
        choices=['gpt-oss:20b', 'gpt-oss:120b', 'gemma3:27b'],
        required=False,
        default='gpt-oss:20b',
        help='define model to use for validation',
    )
    p.add_argument(
        '--calories',
        type=float,
        required=False,
        help='target total calories for the meal plan (will check with ±10%% tolerance)',
    )

    # can add more as needed: budget, calorie information
    return p.parse_args()

def generate_meal_plan(args):
    session = OllamaChatSession(
        model=args.model,
        system_prompt='You are a chef generating meal plans with the given limitations.',
        verbose=args.verbose
    )
    validator = Validator(args)
    if args.verbose:
        print('Created chat session and validator.')
    meal_plan = session.initial_request(args)
    if args.verbose:
        print(f'\nResponse:{meal_plan}')
    iteration_history = [meal_plan]  # Track iteration history for metrics
    
    iterations_taken = 0
    validation_result = None
    for i in range(args.max_iterations):
        iterations_taken = i + 1
        if args.verbose:
            print('===================')
            print(f'Validating try {iterations_taken}.')
        validation_result = validator.validate(meal_plan)
        if isinstance(validation_result, dict) and validation_result.get('success', False):
            # Validation passed, but don't evaluate yet - wait until after loop
            break
        # Extract message from validation result
        if isinstance(validation_result, dict):
            check_message = validation_result.get('message', '')
        else:
            check_message = str(validation_result)
        meal_plan = session.request(check_message)
        iteration_history.append(meal_plan)
    else:
        # Loop completed without breaking (max iterations reached)
        # Final validation result already set in last iteration
        pass
    
    # Now evaluate only the final meal plan (whether validation passed or not)
    quality_evaluation = validator.evaluate_recipe_quality(meal_plan)
    
    return meal_plan, iterations_taken, validation_result, iteration_history, quality_evaluation

def main():
    # Load secrets from .secrets file into os.environ
    load_secrets()
    
    args = parse_args()
    print("args parsed:", args)
    result = generate_meal_plan(args)
    
    # Handle both old format (just meal_plan) and new format (tuple with metrics data)
    if isinstance(result, tuple):
        if len(result) == 5:
            meal_plan, iterations, validation_result, iteration_history, quality_evaluation = result
        else:
            # Backward compatibility with 4-tuple
            meal_plan, iterations, validation_result, iteration_history = result
            quality_evaluation = None
        
        print(meal_plan)
        if args.verbose:
            print(f"\nIterations taken: {iterations}")
            print(f"Validation success: {validation_result.get('success', False)}")
        
        # Print quality evaluation (only run on final meal plan)
        if quality_evaluation:
            print("\n=== Recipe Quality Evaluation ===")
            for metric, data in quality_evaluation.items():
                if metric == 'raw_response':
                    print(f"\nRaw Evaluation Response:\n{data}")
                elif isinstance(data, dict):
                    score = data.get('score', 'N/A')
                    reasoning = data.get('reasoning', 'No reasoning provided')
                    print(f"\n{metric.upper()}: {score}/10")
                    print(f"Reasoning: {reasoning}")
            print("================================")
    else:
        # Backward compatibility
        print(result)


if __name__ == '__main__':
    main()