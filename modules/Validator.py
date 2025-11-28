import os
import re
import requests
from .Ollama import OllamaChatSession


class Validator:
    '''
    Takes a string in as a response, and validates the response
    '''
    def __init__(self, args: dict,
        url: str = 'http://ollama.loweffort.meme/api/chat'):
        self.model =  OllamaChatSession(
            model=args.val_model,
            system_prompt='You will validate the correctness of the recipe against the given rules. These must be strict to ensure safety.',
            verbose=args.verbose,
            )
        self.url = url
        self.verbose = args.verbose
        self.args = args

        # Load API key from environment
        self.fdc_api_key = os.environ.get('FDC_API_KEY')
        if not self.fdc_api_key:
            raise ValueError('FDC_API_KEY environment variable is not set.')

    def check_sections(self, response: str):
        format = '1. Title, 2. Serving size, 3. Ingredients and amounts, 4. Instructions, 5. Nutritional information, 6. Anything else.'
        prompt = f'''Check if this recipe {response} is in this format: {format}.\nReturn only the string 'True' if correct, else return only the missing sections.'''
        
        response = self.model.ask(prompt)
        if self.verbose:
            print(f'Check sections: {response}')
        
        if response.strip().lower() == 'true':
            return True
        
        return response.strip()
    
    def check_ingredients(self, response: str):
        ingredients = self.args.ingredients
        prompt = f'''Check if this recipe {response} includes the ingredients: {ingredients} in both the Ingredients and Instructions sections.\nReturn only the string 'True' if correct, else return only the missing sections.'''
        
        model_response = self.model.ask(prompt)
        if self.verbose:
            print(f'Check ingredients: {model_response}')
        
        if model_response.strip().lower() == 'true':
            return True
        
        return model_response.strip()

    def check_allergens(self, response: str):
        '''
        Simple allergen check: scan the Ingredients and Instructions sections
        for any allergen keywords from args.allergens_to_avoid.
        '''
        allergens = getattr(self.args, 'allergens_to_avoid', [])
        if not allergens:
            return True

        allergens_found = []
        lower = response.lower()
        for allergen in allergens:
            if allergen.lower() in lower:
                allergens_found.append(allergen)

        if allergens_found:
            allergen_output = 'Allergen(s) present): ' + ', '.join(allergens_found)
            return allergen_output
        return True
    
    # add more checks as needed
    def check_budget(self, response: str):
        pass

    def check_calories(self, response: str):
        '''
        Attempt to compute total calories by summing calories of each ingredient via FDC API.
        Compare against max_calories if provided in args.max_calories.
        Requires that Ingredients list includes amounts with units in a parseable way.
        '''

        lines = response.splitlines()
        # Naive parse: find lines under 'Ingredients' section
        in_ingredients = False
        ingredient_lines = []
        for line in lines:
            # Whatever header part we need here:
            if line.lower().startswith('3. ingredients'):
                in_ingredients = True
                continue
            if in_ingredients:
                if re.match(r'^\d+\.', line):
                    break
                ingredient_lines.append(line.strip())

        if not ingredient_lines:
            return 'Could not parse ingredient list to compute calories'

        total_cal = 0.0
        missing_ingredients = []
        for ing in ingredient_lines:
            # very naive splitting: '2 cups flour' --> qty=2, unit=cups, name=flour
            m = re.match(r'([0-9/\.]+)\s+(\w+)\s+(.*)', ing)
            if not m:
                missing_ingredients.append(ing)
                continue
            qty, unit, name = m.groups()
            # convert fraction if needed
            try:
                qty = float(eval(qty))
            except Exception:
                missing_ingredients.append(ing)
                continue

            # Query FDC for this ingredient
            params = {
                'query': name,
                'pageSize': 1,
                'api_key': self.fdc_api_key
            }
            resp = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search', params=params)
            if resp.status_code != 200:
                missing_ingredients.append(name)
                continue
            data = resp.json()
            foods = data.get('foods')
            if not foods:
                missing_ingredients.append(name)
                continue
            food = foods[0]
            # find calorie nutrient
            calories = None
            for nutrient in food.get('foodNutrients', []):
                if nutrient.get('nutrientName') == 'Energy' and nutrient.get('unitName') == 'KCAL':
                    calories = nutrient.get('value')
                    break
            if calories is None:
                missing_ingredients.append(name)
                continue

            if unit.lower() in ('g', 'gram', 'grams'):
                total_cal += (calories / 100.0) * qty
            else:
                # unable to convert non-gram units safely
                missing_ingredients.append(f'{name} (unit {unit})')

        if missing_ingredients:
            return 'Could not compute calories for: ' + ', '.join(missing_ingredients)

        max_cal = getattr(self.args, 'max_calories', None)
        if max_cal is not None:
            if total_cal > max_cal:
                return f'Total calories {total_cal:.1f} exceeds limit {max_cal}'

        return True

    def validate(self, response: str):
        '''
        validation pipeline, does all the checks. 
        Returns True if all pass, or else a natural language response of all issues
        '''

        checks = [
            self.check_sections,
            self.check_ingredients,
            self.check_allergens,
            self.check_calories,
            self.check_budget,
            ]
        
        errors = []
        
        for check in checks:
            result = check(response)
            if result is not True: 
                errors.append(result)
        
        validation = True
        if errors:
            validation = ' | '.join(errors)  # Combine all msg
        else:
            validation = True
        
        if self.verbose:
            print(f'Validation: {validation}')

        return validation