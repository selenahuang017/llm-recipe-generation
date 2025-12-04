#!/usr/bin/env python3
'''
Comprehensive test suite for all Validator methods using example.txt
Each test is in its own function so they can be run separately.
'''

import sys
import os
import argparse
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.Validator import Validator

def load_secrets(secrets_file: str = '.secrets'):
    '''
    Load secrets from a file into os.environ.
    Expected format: KEY=VALUE (one per line)
    '''
    secrets_path = Path(secrets_file)
    if not secrets_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent
        secrets_path = project_root / secrets_file
    
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f'Loaded secrets from {secrets_path}')
    else:
        print(f'Note: Secrets file not found at {secrets_path}')

def setup(input_file: str = 'example.txt'):
    '''Setup test environment - load secrets, read input file, create validator'''
    load_secrets()
    
    # Read input file (try relative to script directory first, then current directory)
    file_path = Path(input_file)
    if not file_path.is_absolute():
        # Try relative to script directory first
        script_dir = Path(__file__).parent
        potential_path = script_dir / input_file
        if potential_path.exists():
            file_path = potential_path
        elif Path(input_file).exists():
            # Use current directory path
            file_path = Path(input_file)
        else:
            raise FileNotFoundError(f'Could not find file: {input_file}')
    
    with open(file_path, 'r') as f:
        example_content = f.read()
    
    # Create a mock args object for testing
    class MockArgs:
        def __init__(self):
            self.val_model = 'gpt-oss:20b'
            self.verbose = True
            self.ingredients = 'white rice, rhubarb, persimmon, sesame seeds, gluten-free oreos, duck, tomatoes'
            self.allergens = 'gluten'  # Test with gluten allergen
            self.check_budget = False
            # Optional: add calorie limits for testing
            self.max_calories = None
            self.max_total_calories = None
            self.budget = None
    
    args = MockArgs()
    
    # Create validator with URL
    url = 'https://ollama.loweffort.meme/api/chat'
    validator = Validator(args, url=url)
    
    return validator, example_content, args

def print_test_header(test_name):
    '''Print a formatted test header'''
    print('\n' + '=' * 80)
    print(f'TEST: {test_name}')
    print('=' * 80)

def print_result(result):
    '''Print validation result in a formatted way'''
    status = '✓ PASS' if result.get('success', False) else '✗ FAIL'
    print(f'\nStatus: {status}')
    print(f"Message: {result.get('message', 'No message')}")
    return result.get('success', False)

def test_extract_recipes(validator, example_content):
    '''Test 1: Extract Recipes'''
    print_test_header('1. Extract Recipes')
    recipes = validator.extract_recipes(example_content)
    for meal_type, recipe in recipes.items():
        if recipe:
            print(f'\n{meal_type.upper()}:')
            print(f'  Length: {len(recipe)} characters')
            print(f'  Preview: {recipe[:100]}...')
        else:
            print(f'\n{meal_type.upper()}: No recipe extracted')
    test_pass = all(recipes.values())
    print(f'\n✓ PASS' if test_pass else '\n✗ FAIL')
    return test_pass, recipes

def test_check_meal_plan_structure(validator, example_content):
    '''Test 2: Check Meal Plan Structure'''
    print_test_header('2. Check Meal Plan Structure')
    result = validator.check_meal_plan_structure(example_content)
    test_pass = print_result(result)
    return test_pass

def test_check_sections(validator, recipes):
    '''Test 3: Check Sections (Format Validation)'''
    print_test_header('3. Check Sections (Format Validation)')
    all_sections_pass = True
    for meal_type, recipe in recipes.items():
        if recipe:
            print(f'\n{meal_type.upper()}:')
            result = validator.check_sections(recipe, meal_type)
            if not print_result(result):
                all_sections_pass = False
    return all_sections_pass

def test_check_ingredients(validator, example_content):
    '''Test 4: Check Ingredients (All Required Ingredients Present)'''
    print_test_header('4. Check Ingredients (All Required Ingredients Present)')
    result = validator.check_ingredients(example_content)
    test_pass = print_result(result)
    return test_pass

def test_check_allergens(validator, recipes):
    '''Test 5: Check Allergens'''
    print_test_header('5. Check Allergens')
    all_allergens_pass = True
    for meal_type, recipe in recipes.items():
        if recipe:
            print(f'\n{meal_type.upper()}:')
            result = validator.check_allergens(recipe, meal_type)
            if not print_result(result):
                all_allergens_pass = False
    return all_allergens_pass

def test_parse_ingredients(validator, recipes):
    '''Test 6: Parse Ingredients from Recipe (Model-based)'''
    print_test_header('6. Parse Ingredients from Recipe (Model-based)')
    all_parse_pass = True
    for meal_type, recipe in recipes.items():
        if recipe:
            print(f'\n{meal_type.upper()}:')
            parsed = validator._parse_ingredients_from_recipe(recipe)
            print(f'  Parsed {len(parsed)} ingredients:')
            if parsed:
                for qty, unit, name in parsed:
                    print(f'    - {qty} {unit} {name}')
            else:
                print('    No ingredients parsed')
                all_parse_pass = False
    return all_parse_pass

def test_check_nutrition(validator, example_content):
    '''Test 7: Check Nutrition (Calories, Protein, Fat, Carbs, Fiber)'''
    print_test_header('7. Check Nutrition (Calories, Protein, Fat, Carbs, Fiber)')
    # Note: This requires FDC_API_KEY to work properly
    fdc_key_available = os.environ.get('FDC_API_KEY') is not None
    if fdc_key_available:
        print('  FDC_API_KEY found in environment')
    else:
        print('  Note: FDC_API_KEY not found - nutrition check may fail')
    result = validator.check_nutrition(example_content)
    test_pass = print_result(result)
    if not test_pass:
        msg = result.get('message', '')
        if 'Could not compute nutrition' in msg:
            print('  Note: Some ingredients may not be found in FDC database')
    return test_pass

def test_check_budget(validator, example_content, args):
    '''Test 8: Check Budget'''
    print_test_header('8. Check Budget')
    # Set a budget for testing
    args.budget = 50.0
    result = validator.check_budget(example_content)
    test_pass = print_result(result)
    return test_pass

def test_full_validation(validator, example_content):
    '''Test 9: Full Validation Pipeline (validate method)'''
    print_test_header('9. Full Validation Pipeline (validate method)')
    result = validator.validate(example_content)
    test_pass = print_result(result)
    return test_pass

def test_recipe_quality_evaluation(validator, example_content):
    '''Test 10: Recipe Quality Evaluation'''
    print_test_header('10. Recipe Quality Evaluation')
    evaluation = validator.evaluate_recipe_quality(example_content)
    print('\nEvaluation Results:')
    for metric, data in evaluation.items():
        if metric == 'raw_response':
            print(f'\n{metric.upper()}:')
            print(f'  {data[:200]}...' if len(data) > 200 else f'  {data}')
        elif isinstance(data, dict):
            score = data.get('score', 'N/A')
            reasoning = data.get('reasoning', 'No reasoning provided')
            print(f'\n{metric.upper()}:')
            print(f'  Score: {score}/10')
            print(f'  Reasoning: {reasoning[:150]}...' if len(reasoning) > 150 else f'  Reasoning: {reasoning}')
    test_pass = all(
        isinstance(data, dict) and data.get('score') is not None 
        for metric, data in evaluation.items() 
        if metric != 'raw_response'
    )
    return test_pass

def run_all_tests(input_file: str = 'example.txt'):
    '''Run all tests'''
    validator, example_content, args = setup(input_file)
    
    # Run all tests
    test_results = []
    
    # Test 1
    test1_pass, recipes = test_extract_recipes(validator, example_content)
    test_results.append(('1. Extract Recipes', test1_pass))
    
    # Test 2
    test2_pass = test_check_meal_plan_structure(validator, example_content)
    test_results.append(('2. Check Meal Plan Structure', test2_pass))
    
    # Test 3
    test3_pass = test_check_sections(validator, recipes)
    test_results.append(('3. Check Sections', test3_pass))
    
    # Test 4
    test4_pass = test_check_ingredients(validator, example_content)
    test_results.append(('4. Check Ingredients', test4_pass))
    
    # Test 5
    test5_pass = test_check_allergens(validator, recipes)
    test_results.append(('5. Check Allergens', test5_pass))
    
    # Test 6
    test6_pass = test_parse_ingredients(validator, recipes)
    test_results.append(('6. Parse Ingredients', test6_pass))
    
    # Test 7
    test7_pass = test_check_nutrition(validator, example_content)
    test_results.append(('7. Check Nutrition', test7_pass))
    
    # Test 8
    test8_pass = test_check_budget(validator, example_content, args)
    test_results.append(('8. Check Budget', test8_pass))
    
    # Test 9
    test9_pass = test_full_validation(validator, example_content)
    test_results.append(('9. Full Validation', test9_pass))
    
    # Test 10
    test10_pass = test_recipe_quality_evaluation(validator, example_content)
    test_results.append(('10. Quality Evaluation', test10_pass))
    
    # Print summary
    print('\n' + '=' * 80)
    print('TEST SUMMARY')
    print('=' * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = '✓' if result else '✗'
        print(f'{status} {test_name}')
    
    print(f'\nTotal: {passed}/{total} tests passed')
    print('=' * 80)
    
    return test_results

def main():
    '''Main function to run tests'''
    parser = argparse.ArgumentParser(description='Run validation tests')
    parser.add_argument(
        '--test',
        type=int,
        choices=range(1, 11),
        help='Run a specific test (1-10). If not specified, all tests are run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='example.txt',
        help='Input file to use for testing (default: example.txt)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print('Available tests:')
        print('  1. Extract Recipes')
        print('  2. Check Meal Plan Structure')
        print('  3. Check Sections')
        print('  4. Check Ingredients')
        print('  5. Check Allergens')
        print('  6. Parse Ingredients')
        print('  7. Check Nutrition')
        print('  8. Check Budget')
        print('  9. Full Validation')
        print('  10. Quality Evaluation')
        return
    
    validator, example_content, mock_args = setup(args.file)
    recipes = validator.extract_recipes(example_content)
    
    if args.test:
        # Map test numbers to their test functions
        test_functions = {
            1: lambda: test_extract_recipes(validator, example_content)[0],  # Returns tuple, take first element
            2: lambda: test_check_meal_plan_structure(validator, example_content),
            3: lambda: test_check_sections(validator, recipes),
            4: lambda: test_check_ingredients(validator, example_content),
            5: lambda: test_check_allergens(validator, recipes),
            6: lambda: test_parse_ingredients(validator, recipes),
            7: lambda: test_check_nutrition(validator, example_content),
            8: lambda: test_check_budget(validator, example_content, mock_args),
            9: lambda: test_full_validation(validator, example_content),
            10: lambda: test_recipe_quality_evaluation(validator, example_content),
        }
        
        # Run the selected test
        if args.test in test_functions:
            result = test_functions[args.test]()
            status = '✓ PASS' if result else '✗ FAIL'
            print(f'\nTest {args.test} result: {status}')
        else:
            print(f'Error: Test {args.test} not found')
    else:
        # Run all tests
        run_all_tests(args.file)

if __name__ == '__main__':
    main()