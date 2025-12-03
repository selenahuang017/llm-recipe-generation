#!/usr/bin/env python3
"""
Test file to query the FDC API (Food Data Central) for ingredient information.
Takes a string input (ingredient name) and formats the request like _calculate_calories_for_recipe.
"""

import sys
import os
import requests
import json
from pathlib import Path

def load_secrets(secrets_file: str = '.secrets'):
    """
    Load secrets from a file into os.environ.
    Expected format: KEY=VALUE (one per line)
    """
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
        print(f"✓ Loaded secrets from {secrets_path}")
        return True
    else:
        print(f"✗ Secrets file not found at {secrets_path}")
        return False

def test_fdc_api(ingredient_name: str):
    """
    Test FDC API query for a given ingredient name.
    Formats the request exactly like _calculate_calories_for_recipe does.
    
    Args:
        ingredient_name: The name of the ingredient to search for
        
    Returns:
        dict with success status and response data
    """
    # Load API key from environment
    fdc_api_key = os.environ.get('FDC_API_KEY')
    
    if not fdc_api_key:
        return {
            'success': False,
            'error': 'FDC_API_KEY not found in environment. Make sure it is set in .secrets file.'
        }
    
    # Format params exactly like _calculate_calories_for_recipe
    params = {
        'query': ingredient_name,
        'pageSize': 1,
        'api_key': fdc_api_key
    }
    
    url = 'https://api.nal.usda.gov/fdc/v1/foods/search'
    
    print(f"\n{'='*80}")
    print(f"Testing FDC API for ingredient: '{ingredient_name}'")
    print(f"{'='*80}")
    print(f"\nRequest URL: {url}")
    print(f"Request params:")
    print(f"  - query: {params['query']}")
    print(f"  - pageSize: {params['pageSize']}")
    print(f"  - api_key: {params['api_key'][:10]}... (hidden)")
    
    try:
        # Make the request
        print(f"\nMaking request...")
        resp = requests.get(url, params=params)
        
        print(f"\nResponse Status Code: {resp.status_code}")
        
        if resp.status_code != 200:
            return {
                'success': False,
                'error': f'API request failed with status code {resp.status_code}',
                'response_text': resp.text[:500] if resp.text else 'No response text'
            }
        
        # Parse JSON response
        data = resp.json()
        foods = data.get('foods', [])
        
        if not foods:
            return {
                'success': False,
                'error': f'No foods found for ingredient "{ingredient_name}"',
                'response_data': data
            }
        
        # Get the first food result
        food = foods[0]
        
        # Extract relevant information
        fdc_id = food.get('fdcId')
        description = food.get('description', 'N/A')
        brand_owner = food.get('brandOwner', 'N/A')
        
        # Find calorie nutrient
        calories = None
        nutrients = food.get('foodNutrients', [])
        for nutrient in nutrients:
            if nutrient.get('nutrientName') == 'Energy' and nutrient.get('unitName') == 'KCAL':
                calories = nutrient.get('value')
                break
        
        result = {
            'success': True,
            'fdc_id': fdc_id,
            'description': description,
            'brand_owner': brand_owner,
            'calories_per_100g': calories,
            'total_nutrients': len(nutrients),
            'full_response': food
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f'Request exception: {str(e)}'
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f'JSON decode error: {str(e)}',
            'response_text': resp.text[:500] if 'resp' in locals() else 'No response'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }

def print_result(result):
    """Print the test result in a formatted way"""
    if result.get('success'):
        print(f"\n{'='*80}")
        print("✓ SUCCESS - Food found in FDC database")
        print(f"{'='*80}")
        print(f"\nFDC ID: {result.get('fdc_id')}")
        print(f"Description: {result.get('description')}")
        print(f"Brand Owner: {result.get('brand_owner')}")
        print(f"Calories per 100g: {result.get('calories_per_100g', 'N/A')}")
        print(f"Total Nutrients: {result.get('total_nutrients')}")
        
        # Show sample nutrients
        if 'full_response' in result:
            nutrients = result['full_response'].get('foodNutrients', [])
            if nutrients:
                print(f"\All Nutrients:")
                for i, nutrient in enumerate(nutrients):
                    name = nutrient.get('nutrientName', 'N/A')
                    value = nutrient.get('value', 'N/A')
                    unit = nutrient.get('unitName', '')
                    print(f"  {i+1}. {name}: {value} {unit}")
    else:
        print(f"\n{'='*80}")
        print("✗ FAILED")
        print(f"{'='*80}")
        print(f"\nError: {result.get('error', 'Unknown error')}")
        if 'response_text' in result:
            print(f"\nResponse text: {result['response_text']}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FDC API for ingredient lookup')
    parser.add_argument(
        'ingredient',
        nargs='?',
        default='white rice',
        help='Ingredient name to search for (default: white rice)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List some example ingredients to test'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Example ingredients to test:")
        print("  - white rice")
        print("  - chicken breast")
        print("  - olive oil")
        print("  - tomato")
        print("  - salmon")
        print("  - broccoli")
        return
    
    # Load secrets
    if not load_secrets():
        print("\nNote: Continuing anyway - API key might be set via environment variable")
    
    # Test the API
    result = test_fdc_api(args.ingredient)
    
    # Print results
    print_result(result)
    
    # Return exit code
    sys.exit(0 if result.get('success') else 1)

if __name__ == '__main__':
    main()

