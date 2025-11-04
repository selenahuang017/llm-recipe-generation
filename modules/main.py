import argparse

from .Ollama import OllamaChatSession
from .Validator import Validator

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Create a recipe with the following ingredients:')
    p.add_argument(
        '--model',
        type=str,
        choices=['gpt-oss:20b', 'gpt-oss:120b', 'gemma3:27b',],
        required=False,
        default='gpt-oss:20b',
        help='define model to ask one of them',
    )
    p.add_argument(
        '--max_iterations',
        type=int,
        required=False,
        default=10,
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
        '--verbose',
        type=bool,
        required=False,
        default=False,
        help='verbosity, True includes debug information',
    )

    # can add more as needed: budget, calorie information
    return p.parse_args()

def generate_recipe(args):
    session = OllamaChatSession(
        model=args.model,
        system_prompt='You are a chef generating recipes with the given limitations.',
    )
    validator = Validator(args)
    recipe = session.initial_request(args)
    for i in range(args.max_iterations):
        check = validator.validate(recipe)
        if recipe is True:
            return recipe
        session.request(check)
    return recipe

def main():
    args = parse_args()
    recipe = generate_recipe(args)
    print(recipe)


if __name__ == '__main__':
    main()