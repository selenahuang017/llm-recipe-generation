import os
import re
import requests
from .Ollama import OllamaChatSession
from typing import Dict, List, Tuple


class Validator:
    '''
    Takes a string in as a response, and validates the response
    '''
    def __init__(self, args: dict,
        url: str = 'https://ollama.loweffort.meme/api/chat'):
        self.model =  OllamaChatSession(
            model=args.val_model,
            system_prompt='You will validate the correctness of the recipe against the given rules. These must be strict to ensure safety.',
            verbose=args.verbose,
            )
        self.url = url
        self.verbose = args.verbose
        self.args = args

        # Load API key from environment (required for budget/calorie checks)
        self.fdc_api_key = os.environ.get('FDC_API_KEY')
        # Only raise error if budget checking is enabled (calorie check can fail gracefully)
        check_budget = getattr(args, 'check_budget', False)
        if check_budget and not self.fdc_api_key:
            raise ValueError('FDC_API_KEY environment variable is not set. Required for budget checking.')

    def extract_recipes(self, meal_plan: str) -> Dict[str, str]:
        '''
        Extremely robust extractor for BREAKFAST, LUNCH, DINNER from messy LLM output.
        Handles markdown, colons, multiple hashes, bold, spacing, dashes, etc.
        '''
        recipes = {'breakfast': '', 'lunch': '', 'dinner': ''}

        # Light normalization (do NOT strip # entirely)
        cleaned = re.sub(r'[*_`]', '', meal_plan)

        # Tolerant section matcher:
        pattern = r'''
            (?m)                    # multiline mode
            (^|\n)                  # anchor
            \s*                     # leading spaces
            #{0,3}                  # optional markdown ### headers
            \s*                     
            (BREAKFAST|LUNCH|DINNER)   # section name
            \s*                     # optional spaces
            [:–\-]*                 # optional punctuation
            \s*
            \n                      # newline ends header
            (.*?)                   # capture body (lazy)
            (?=                     # --- LOOKAHEAD START ---
                \n\s*#{0,3}\s*(?:BREAKFAST|LUNCH|DINNER)\b   # next header
                | \Z                # OR end of string
            )                       # --- LOOKAHEAD END ---
        '''

        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL | re.VERBOSE)

        # Store sections
        for header, body, _ in matches:
            recipes[header.lower()] = body.strip()

        # Fallback heuristic if some sections are still empty
        if any(not v for v in recipes.values()):
            for meal in ['breakfast', 'lunch', 'dinner']:
                if recipes[meal]:
                    continue
                # fallback search for line containing the word
                simple = re.search(
                    meal + r'.*?\n(.*?)(?=\n[A-Za-z ]*?:|\Z)',
                    cleaned,
                    re.IGNORECASE | re.DOTALL
                )
                if simple:
                    recipes[meal] = simple.group(1).strip()

        return recipes

    def check_meal_plan_structure(self, meal_plan: str) -> Dict[str, any]:
        '''
        Check if meal plan has 3 recipes following breakfast/lunch/dinner pattern.
        Returns dict with success and message.
        '''
        recipes = self.extract_recipes(meal_plan)
        
        missing = []
        if not recipes['breakfast']:
            missing.append('breakfast')
        if not recipes['lunch']:
            missing.append('lunch')
        if not recipes['dinner']:
            missing.append('dinner')
        
        if missing:
            return {
                'success': False,
                'message': f'Missing meal(s): {", ".join(missing)}'
            }
        
        return {
            'success': True,
            'message': 'Meal plan structure is correct: breakfast, lunch, and dinner recipes found'
        }

    def check_sections(self, recipe: str, meal_type: str = '') -> Dict[str, any]:
        format = '1. Title, 2. Serving size, 3. Ingredients and amounts, 4. Instructions, 5. Nutritional information, 6. Anything else.'
        prompt = f'''Check if this recipe {recipe} is in this format: {format}.\nReturn only the string "True" if correct, else return only the missing sections.'''
        
        response = self.model.ask(prompt)
        if self.verbose:
            print(f'Check sections ({meal_type}): {response}')
        
        if response.strip().lower() == 'true':
            return {
                'success': True,
                'message': f'{meal_type.capitalize()}: All sections present' if meal_type else 'All sections present'
            }
        
        return {
            'success': False,
            'message': f'{meal_type.capitalize()}: Missing sections - {response.strip()}' if meal_type else f'Missing sections - {response.strip()}'
        }
    
    def check_ingredients(self, meal_plan: str) -> Dict[str, any]:
        '''
        Check if all ingredients are used at least once across the three meals.
        Each ingredient must appear in at least one of the three recipes (breakfast, lunch, or dinner).
        Returns dict with success and message.
        '''
        ingredients = self.args.ingredients
        
        # Check if all ingredients appear at least once across the meal plan
        prompt = f'''Check if this meal plan includes ALL of these ingredients at least once: {ingredients}. Each ingredient must appear in at least one of the three recipes (breakfast, lunch, or dinner). An ingredient does not need to appear in all three recipes, but it must appear in at least one recipe somewhere in the meal plan.\nReturn only the string "True" if all ingredients are present at least once somewhere in the meal plan, else return only the missing ingredients.'''
        
        model_response = self.model.ask(prompt + ' Meal plan: ' + meal_plan)
        if self.verbose:
            print(f'Check ingredients: {model_response}')
        
        if model_response.strip().lower() == 'true':
            return {
                'success': True,
                'message': 'All ingredients are used across the meal plan'
            }
        
        return {
            'success': False,
            'message': f'Missing ingredients: {model_response.strip()}'
        }

    def check_allergens(self, recipe: str, meal_type: str = '') -> Dict[str, any]:
        '''
        Use model to check if any ingredients in the recipe contain allergens.
        Uses args.allergens (not allergens_to_avoid).
        '''
        allergens = getattr(self.args, 'allergens', None)
        if not allergens:
            return {
                'success': True,
                'message': f'{meal_type.capitalize()}: No allergens specified' if meal_type else 'No allergens specified'
            }

        prompt = f'''Check if any of the ingredients in this recipe contain or are related to these allergens: {allergens}. Analyze the ingredients list carefully.\nReturn only the string "True" if no allergens are present, else return the allergen(s) found.'''
        
        model_response = self.model.ask(prompt + ' Recipe: ' + recipe)
        if self.verbose:
            print(f'Check allergens ({meal_type}): {model_response}')
        
        if model_response.strip().lower() == 'true':
            return {
                'success': True,
                'message': f'{meal_type.capitalize()}: No allergens detected' if meal_type else 'No allergens detected'
            }
        
        return {
            'success': False,
            'message': f'{meal_type.capitalize()}: Allergen(s) found - {model_response.strip()}' if meal_type else f'Allergen(s) found - {model_response.strip()}'
        }
    
    def _parse_ingredients_from_recipe(self, recipe: str) -> List[Tuple[float, str, str]]:
        '''
        Extremely robust ingredient parser for LLM recipes.

        Returns list of (qty: float, unit: str, ingredient_name: str).

        Handles:
        - '1 cup chopped onion'
        - '½ cup rice'
        - '200 g potatoes'
        - '1 tbsp of olive oil'
        - '2 eggs'
        - '1 medium potato'
        - 'about 2 tbsp sugar'
        - '(optional)', '(approx 200 g)'
        - Markdown bullets
        '''

        # --- 1. Extract section 3 cleanly (stop before section 4 header) ---
        normalized = recipe.replace('\r\n', '\n')

        section_match = re.search(
            r'''(?imsx)
            ^\s*3\s*[\.\)\-]?\s*(?:ingredients?[^:\n]*:?)\s*\n+   # section 3 header
            (.*?)                                               # capture body lazily
            (?=^\s*4\s*[\.\)\-]?\s*)                            # stop exactly at section 4 header
            ''',
            normalized
        )

        if section_match:
            ingredients_text = section_match.group(1).strip()
        else:
            # fallback: start at an "ingredients" header and stop at the next instructions/directions header (even if unnumbered)
            fallback_match = re.search(
                r'''(?imsx)
                (?:ingredients?[^:\n]*:?)\s*\n+   # ingredients heading
                (.*?)                             # capture body lazily
                (?=^\s*[*_#>\-]*\s*(?:4\s*[\.\)\-]?\s*)?(?:instructions?|directions?|steps?)\b|$)  # stop when instructions begin
                ''',
                normalized
            )
            ingredients_text = fallback_match.group(1).strip() if fallback_match else recipe

        # --- 2. Normalize Unicode fractions ---
        FRACTIONS = {
            '½': '1/2', '¼': '1/4', '¾': '3/4',
            '⅓': '1/3', '⅔': '2/3',
            '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8'
        }
        for sym, frac in FRACTIONS.items():
            ingredients_text = ingredients_text.replace(sym, frac)

        # --- 3. Get candidate lines ---
        raw_lines = ingredients_text.splitlines()
        lines = []

        for line in raw_lines:
            clean = line.strip().lstrip('-•—–').strip()
            if len(clean.split()) >= 2:
                lines.append(clean)

        parsed = []

        # --- 4. Regex patterns for ingredient lines ---
        patterns = [
            # 1) Quantity + unit + name
            r'^([0-9./]+)\s+([a-zA-Z]+)\s+(.+)$',

            # 2) 'about 2 tbsp sugar'
            r'^(?:about|approx\.?|~)\s*([0-9./]+)\s+([a-zA-Z]+)\s+(.+)$',

            # 3) '200 g potatoes'
            r'^([0-9./]+)\s*(g|kg|oz|lb|ml|l|tsp|tbsp)\s+(.+)$',

            # 4) '1 cup of chopped onion'
            r'^([0-9./]+)\s+([a-zA-Z]+)\s+of\s+(.+)$',

            # 5) '2 eggs'
            r'^([0-9./]+)\s+(egg[s]?)$',

            # 6) '1 potato'
            r'^([0-9./]+)\s+([A-Za-z][A-Za-z ]+?)$',
        ]

        def parse_qty(q):
            if '/' in q:
                try:
                    a, b = q.split('/')
                    return float(a) / float(b)
                except:
                    return None
            try:
                return float(q)
            except:
                return None

        # --- 5. Try parsing each ingredient line ---
        for line in lines:
            cleaned = re.sub(r'\([^)]*\)', '', line)  # remove parentheses
            cleaned = cleaned.replace(',', ' ').strip()

            for pat in patterns:
                m = re.match(pat, cleaned, flags=re.IGNORECASE)
                if not m:
                    continue

                qty_str = m.group(1)
                unit = m.group(2) if len(m.groups()) >= 2 else ''
                name = m.group(3) if len(m.groups()) >= 3 else m.group(2)

                qty = parse_qty(qty_str)
                if qty is None:
                    continue

                unit = unit.lower().strip()
                name = name.strip()
                name = re.sub(r'(optional|to taste|as needed)$', '', name, flags=re.IGNORECASE).strip()

                parsed.append((qty, unit, name))
                break

        return parsed

    
    def _parse_ingredients_fallback(self, recipe: str) -> List[Tuple[float, str, str]]:
        '''
        Very forgiving fallback parser.
        Catches ANYTHING resembling '<qty> <unit> <name>'.
        '''

        text = recipe.lower()
        lines = [l.strip(' -•\t') for l in text.splitlines()]

        parsed = []

        for line in lines:
            # Skip empty or irrelevant lines
            if not line or len(line.split()) < 2:
                continue

            # Generic pattern: 'qty unit name'
            m = re.match(
                r'^([0-9./]+)\s+([a-zA-Z]+)\s+(.+)$',
                line,
                flags=re.IGNORECASE
            )
            if m:
                qty = self._parse_quantity(m.group(1))
                unit = m.group(2)
                name = m.group(3)
                if qty is not None:
                    parsed.append((qty, unit, name))
                continue

            # 'qty name' (2 eggs, 1 potato)
            m = re.match(
                r'^([0-9./]+)\s+([a-zA-Z][a-zA-Z ]+)$',
                line
            )
            if m:
                qty = self._parse_quantity(m.group(1))
                unit = 'piece'
                name = m.group(2)
                if qty is not None:
                    parsed.append((qty, unit, name))
                continue

        return parsed

    
    def _parse_quantity(self, qty_str: str) -> float:
        '''
        Parse quantity string to float.
        Handles fractions, decimals, and common fraction symbols.
        '''
        qty_str = qty_str.strip()
        
        # Handle common fraction symbols
        if '½' in qty_str or '1/2' in qty_str:
            return 0.5
        elif '¼' in qty_str or '1/4' in qty_str:
            return 0.25
        elif '¾' in qty_str or '3/4' in qty_str:
            return 0.75
        elif '⅓' in qty_str or '1/3' in qty_str:
            return 1/3
        elif '⅔' in qty_str or '2/3' in qty_str:
            return 2/3
        elif '⅛' in qty_str or '1/8' in qty_str:
            return 1/8
        elif '⅜' in qty_str or '3/8' in qty_str:
            return 3/8
        elif '⅝' in qty_str or '5/8' in qty_str:
            return 5/8
        elif '⅞' in qty_str or '7/8' in qty_str:
            return 7/8
        
        # Handle fraction format like '1/2' or '3/4'
        if '/' in qty_str:
            try:
                num, den = qty_str.split('/')
                return float(num) / float(den)
            except Exception:
                pass
        
        # Handle decimal format
        try:
            return float(qty_str)
        except ValueError:
            return None

    def _get_standard_conversion(self, unit: str, ingredient_name: str) -> float:
        '''
        Get standard conversion from unit to grams.
        Returns grams per unit, or None if conversion is not available/ambiguous.
        '''
        unit_lower = unit.lower().strip()
        ingredient_lower = ingredient_name.lower()
        
        # Direct gram conversions
        if unit_lower in ('g', 'gram', 'grams'):
            return 1.0
        
        # Volume to gram conversions (approximate, ingredient-specific)
        # Common ingredient densities
        if unit_lower in ('cup', 'cups'):
            # Ingredient-specific conversions
            if 'rice' in ingredient_lower:
                return 200.0  # 1 cup rice ≈ 200g
            elif 'flour' in ingredient_lower:
                return 120.0  # 1 cup flour ≈ 120g
            elif 'sugar' in ingredient_lower:
                return 200.0  # 1 cup sugar ≈ 200g
            elif 'water' in ingredient_lower or 'milk' in ingredient_lower:
                return 240.0  # 1 cup liquid ≈ 240g
            elif 'oil' in ingredient_lower:
                return 220.0  # 1 cup oil ≈ 220g
            elif 'tomato' in ingredient_lower or 'sauce' in ingredient_lower:
                return 240.0  # 1 cup tomato/sauce ≈ 240g
            else:
                # Generic cup conversion (may need model for accuracy)
                return None
        
        elif unit_lower in ('tbsp', 'tablespoon', 'tablespoons'):
            if 'oil' in ingredient_lower or 'butter' in ingredient_lower:
                return 14.0  # 1 tbsp oil/butter ≈ 14g
            elif 'syrup' in ingredient_lower or 'honey' in ingredient_lower:
                return 21.0  # 1 tbsp syrup/honey ≈ 21g
            elif 'sesame' in ingredient_lower and 'seed' in ingredient_lower:
                return 9.0  # 1 tbsp sesame seeds ≈ 9g
            else:
                return 15.0  # Generic 1 tbsp ≈ 15g (for liquids/dense items)
        
        elif unit_lower in ('tsp', 'teaspoon', 'teaspoons'):
            if 'oil' in ingredient_lower or 'butter' in ingredient_lower:
                return 4.7  # 1 tsp oil/butter ≈ 4.7g
            elif 'syrup' in ingredient_lower or 'honey' in ingredient_lower:
                return 7.0  # 1 tsp syrup/honey ≈ 7g
            else:
                return 5.0  # Generic 1 tsp ≈ 5g
        
        # Piece/unit conversions
        elif unit_lower in ('piece', 'pieces', 'medium', 'large', 'small'):
            if 'egg' in ingredient_lower:
                return 50.0  # 1 medium egg ≈ 50g
            elif 'oreo' in ingredient_lower or 'cookie' in ingredient_lower:
                return 11.0  # 1 cookie ≈ 11g
            else:
                # Piece conversions are very ingredient-specific
                return None
        
        # Weight conversions
        elif unit_lower in ('oz', 'ounce', 'ounces'):
            return 28.35  # 1 oz = 28.35g
        elif unit_lower in ('lb', 'pound', 'pounds'):
            return 453.59  # 1 lb = 453.59g
        
        # Special cases
        elif unit_lower == 'pinch':
            return 0.5  # 1 pinch ≈ 0.5g (very approximate)
        
        # Unknown unit - needs model assistance
        return None

    def _get_nutrients_from_fdc(self, ingredient_name: str) -> Dict[str, float]:
        '''
        Query FDC API safely using Foundation/SR data only.
        Prevents 2000–5000 kcal/100g branded-item anomalies.
        Returns None for any missing nutrient.
        '''

        if not self.fdc_api_key:
            return {k: None for k in ['calories', 'protein', 'fat', 'carbs', 'fiber']}

        params = {
            'query': ingredient_name,
            'pageSize': 1,
            'dataType': ['Foundation', 'SR Legacy'],  # << SAFE
            'api_key': self.fdc_api_key,
        }

        try:
            resp = requests.get(
                'https://api.nal.usda.gov/fdc/v1/foods/search', params=params
            )
            data = resp.json()
            foods = data.get('foods', [])
            if not foods:
                return {k: None for k in ['calories', 'protein', 'fat', 'carbs', 'fiber']}

            nutrients = foods[0].get('foodNutrients', [])

            out = {'calories': None, 'protein': None, 'fat': None, 'carbs': None, 'fiber': None}

            for n in nutrients:
                name = n.get('nutrientName', '')
                val = n.get('value', None)

                if name == 'Energy' and n.get('unitName') == 'KCAL':
                    out['calories'] = val
                elif name == 'Protein':
                    out['protein'] = val
                elif name == 'Total lipid (fat)':
                    out['fat'] = val
                elif name == 'Carbohydrate, by difference':
                    out['carbs'] = val
                elif name == 'Fiber, total dietary':
                    out['fiber'] = val

            return out

        except Exception:
            return {k: None for k in ['calories', 'protein', 'fat', 'carbs', 'fiber']}


    def _lookup_density(self, ingredient_name: str):
        '''
        Returns grams per 1 cup for the ingredient.
        Densities are APPROXIMATE but stable and prevent wild errors.
        '''
        name = ingredient_name.lower()

        DENSITIES = {
            # Common cooking ingredients
            'water': 240,
            'milk': 245,
            'broth': 240,
            'oil': 220,
            'olive oil': 220,
            'butter': 227,
            'flour': 120,
            'oat flour': 92,
            'rice': 195,
            'sugar': 200,
            'honey': 340,
            'maple syrup': 322,
            'salt': 292,
            'quinoa': 185,
            'potato': 150,
            'carrot': 128,
            'duck': 145,
            'egg': 50,       # per medium egg
            'avocado': 150,
            'spinach': 30,
            'kale': 70,
            'onion': 160,
            'pepper': 120,
            'tomato': 180
        }

        # Match by substring (so 'diced onion' works)
        for key, density in DENSITIES.items():
            if key in name:
                return density

        return None


    def _convert_to_grams(self, qty: float, unit: str, ingredient_name: str) -> float:
        '''
        Convert qty+unit into grams safely.
        Returns None if conversion cannot be trusted.
        '''

        unit = unit.lower().strip()

        # Direct grams
        if unit in ('g', 'gram', 'grams'):
            return qty

        if unit in ('kg', 'kilogram', 'kilograms'):
            return qty * 1000

        # Volume measures — need density OR fallback density table
        VOLUME_UNITS_ML = {
            'ml': 1,
            'milliliter': 1,
            'milliliters': 1,
            'l': 1000,
            'liter': 1000,
            'liters': 1000,
            'tsp': 5,
            'teaspoon': 5,
            'tbsp': 15,
            'tablespoon': 15,
            'cup': 240,
            'cups': 240,
        }

        # If unit is a volume measure
        if unit in VOLUME_UNITS_ML:
            ml = qty * VOLUME_UNITS_ML[unit]

            # Lookup ingredient density
            density_g_per_cup = self._lookup_density(ingredient_name)
            if density_g_per_cup:
                g_per_ml = density_g_per_cup / 240
                return ml * g_per_ml

            # Unknown density → refuse to estimate
            return None

        # Pieces (eggs, shrimp, potatoes)
        if unit in ('piece', 'pieces', 'item', 'items'):
            density = self._lookup_density(ingredient_name)
            if density:
                # treat '1 piece' as roughly 1/2 cup unless known
                return qty * (density * 0.5)
            return None

        # Fallback: unit not recognized
        return None


    def _extract_nutritional_info(self, recipe: str) -> Dict[str, float]:
        '''
        Extract nutritional information from section 5 of recipe.
        Hybrid approach: tries regex first, then model if needed.
        Returns dict with keys: 'calories', 'protein', 'fat', 'carbs', 'fiber'
        '''
        result = {
            'calories': None,
            'protein': None,
            'fat': None,
            'carbs': None,
            'fiber': None
        }
        
        # Find section 5
        lines = recipe.splitlines()
        in_section_5 = False
        section_5_lines = []
        
        for line in lines:
            # Check for section 5 header
            if re.search(r'5\.\s*[Nn]utritional', line) or re.search(r'\*\*5\.\s*[Nn]utritional', line):
                in_section_5 = True
                continue
            if in_section_5:
                # Stop at next numbered section
                if re.match(r'^\s*\*?\*?\d+\.', line) and 'nutritional' not in line.lower():
                    break
                if line.strip():
                    section_5_lines.append(line.strip())
        
        section_5_text = '\n'.join(section_5_lines)
        
        # Try regex patterns first
        # Calories patterns
        cal_patterns = [
            r'[Cc]alories?:\s*~?\s*([\d.]+)\s*kcal',
            r'[Cc]alories?:\s*~?\s*([\d.]+)',
            r'\|\s*[Cc]alories?\s*\|\s*~?\s*([\d.]+)\s*kcal',
        ]
        for pattern in cal_patterns:
            match = re.search(pattern, section_5_text)
            if match:
                result['calories'] = float(match.group(1))
                break
        
        # Protein patterns
        protein_patterns = [
            r'[Pp]rotein:\s*([\d.]+)\s*g',
            r'\|\s*[Pp]rotein\s*\|\s*([\d.]+)\s*g',
        ]
        for pattern in protein_patterns:
            match = re.search(pattern, section_5_text)
            if match:
                result['protein'] = float(match.group(1))
                break
        
        # Fat patterns
        fat_patterns = [
            r'[Ff]at:\s*([\d.]+)\s*g',
            r'[Tt]otal\s+[Ff]at:\s*([\d.]+)\s*g',
            r'\|\s*[Ff]at\s*\|\s*([\d.]+)\s*g',
        ]
        for pattern in fat_patterns:
            match = re.search(pattern, section_5_text)
            if match:
                result['fat'] = float(match.group(1))
                break
        
        # Carbs patterns
        carbs_patterns = [
            r'[Cc]arbohydrates?:\s*([\d.]+)\s*g',
            r'[Cc]arbs?:\s*([\d.]+)\s*g',
            r'\|\s*[Cc]arbohydrates?\s*\|\s*([\d.]+)\s*g',
        ]
        for pattern in carbs_patterns:
            match = re.search(pattern, section_5_text)
            if match:
                result['carbs'] = float(match.group(1))
                break
        
        # Fiber patterns
        fiber_patterns = [
            r'[Ff]iber:\s*([\d.]+)\s*g',
            r'\|\s*[Ff]iber\s*\|\s*([\d.]+)\s*g',
        ]
        for pattern in fiber_patterns:
            match = re.search(pattern, section_5_text)
            if match:
                result['fiber'] = float(match.group(1))
                break
        
        # If regex failed for any values, use model
        missing_values = [k for k, v in result.items() if v is None]
        if missing_values and section_5_text:
            prompt = f'''Extract nutritional information from this recipe section:
            {section_5_text}

            Extract the following values (if present):
            - Calories (in kcal)
            - Protein (in g)
            - Fat (in g)
            - Carbohydrates (in g)
            - Fiber (in g)

            Format your response as:
            Calories: XX kcal
            Protein: XX g
            Fat: XX g
            Carbohydrates: XX g
            Fiber: XX g

            If a value is not found, write "N/A" for that line.'''
            
            try:
                model_response = self.model.ask(prompt)
                
                # Parse model response
                for line in model_response.split('\n'):
                    line = line.strip()
                    if 'calories' in line.lower() and 'kcal' in line.lower():
                        match = re.search(r'([\d.]+)', line)
                        if match:
                            result['calories'] = float(match.group(1))
                    elif 'protein' in line.lower() and 'g' in line.lower():
                        match = re.search(r'([\d.]+)', line)
                        if match:
                            result['protein'] = float(match.group(1))
                    elif 'fat' in line.lower() and 'g' in line.lower():
                        match = re.search(r'([\d.]+)', line)
                        if match:
                            result['fat'] = float(match.group(1))
                    elif 'carbohydrate' in line.lower() and 'g' in line.lower():
                        match = re.search(r'([\d.]+)', line)
                        if match:
                            result['carbs'] = float(match.group(1))
                    elif 'fiber' in line.lower() and 'g' in line.lower():
                        match = re.search(r'([\d.]+)', line)
                        if match:
                            result['fiber'] = float(match.group(1))
            except Exception as e:
                if self.verbose:
                    print(f'Error in model extraction: {e}')
        
        return result

    def _calculate_nutrition_for_recipe(self, recipe: str):
        '''
        Compute nutrition safely.
        Skips ingredients with unknown units or densities.
        Meal is only considered valid if >=70% of ingredients were computable.
        '''

        ingredients = self._parse_ingredients_from_recipe(recipe)

        totals = {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0, 'fiber': 0}
        missing = []
        processed_count = 0

        for qty, unit, name in ingredients:

            # Get USDA nutrients
            nutrients = self._get_nutrients_from_fdc(name)
            if nutrients['calories'] is None:
                missing.append(f'{name} (no USDA match)')
                continue

            # Convert to grams
            grams = self._convert_to_grams(qty, unit, name)
            if grams is None:
                missing.append(f'{name} (unit "{unit}" not convertible)')
                continue

            processed_count += 1

            # Add nutrient totals
            factor = grams / 100.0
            for key in totals:
                if nutrients[key] is not None:
                    totals[key] += nutrients[key] * factor

        # Require at least 70% of ingredients to be computable
        if processed_count < max(1, int(len(ingredients) * 0.7)):
            return ({'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0, 'fiber': 0}, missing)

        return (totals, missing)


    def check_nutrition(self, meal_plan: str) -> Dict[str, any]:
        '''
        Check nutrition (calories, protein, fat, carbs, fiber) per recipe AND total across all meals.
        Only validates calories against max limits. Other nutrients are informational only.
        Returns dict with success and message.
        '''
        recipes = self.extract_recipes(meal_plan)
        meal_nutrition = {}
        total_nutrition = {
            'calories': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'carbs': 0.0,
            'fiber': 0.0
        }
        all_missing = []
        calorie_issues = []
        caloric_mismatches = []
        tolerance = 0.1  # 10% tolerance
        
        for meal_type, recipe in recipes.items():
            if not recipe:
                continue
            
            # Calculate nutrition from ingredients
            calculated_nutrition, missing = self._calculate_nutrition_for_recipe(recipe)
            meal_nutrition[meal_type] = calculated_nutrition
            
            # Add to totals
            for key in total_nutrition:
                total_nutrition[key] += calculated_nutrition[key]
            
            if missing:
                all_missing.extend([f'{meal_type}: {m}' for m in missing])
            
            # Extract stated nutrition from section 5
            stated_nutrition = self._extract_nutritional_info(recipe)
            
            # Compare calculated vs stated - check calories for mismatches, but report all as info
            # First, check calories for tolerance mismatches
            calculated_calories = calculated_nutrition.get('calories', 0.0)
            stated_calories = stated_nutrition.get('calories')
            
            if stated_calories is not None:
                # Calculate tolerance range for calories only
                lower_bound = stated_calories * (1 - tolerance)
                upper_bound = stated_calories * (1 + tolerance)
                
                # Calculate difference percentage (used in both cases)
                diff_percent = abs((calculated_calories - stated_calories) / stated_calories * 100) if stated_calories > 0 else 0
                
                # Check if calculated calories are within tolerance
                is_within_tolerance = lower_bound <= calculated_calories <= upper_bound
                
                # Build message - always report, but indicate if it exceeds tolerance
                if is_within_tolerance:
                    message = f'{meal_type.capitalize()}: Calories calculated {calculated_calories:.1f} kcal, stated {stated_calories:.1f} kcal ({diff_percent:.1f}% difference)'
                else:
                    message = f'{meal_type.capitalize()}: Calculated calories {calculated_calories:.1f} kcal, stated {stated_calories:.1f} kcal ({diff_percent:.1f}% difference EXCEEDS {tolerance*100:.0f}% tolerance)'
                
                caloric_mismatches.append(message)
        
        # Check if we were able to calculate nutrition for at least one meal
        # If all meals have missing ingredients or no meals were processed, that's a failure
        if all_missing and not meal_nutrition:
            return {
                'success': False,
                'message': f'Could not compute nutrition for any meals. Missing ingredients: {", ".join(all_missing)}'
            }
        
        # If some meals have missing ingredients but we calculated others, continue
        # but include the missing ingredients in the message

        # Check TOTAL calories limits (both min and max)
        # If --calories is set, calculate min/max with ±10% tolerance
        target_calories = getattr(self.args, 'calories', None)
        if target_calories is not None:
            min_total_cal = target_calories * 0.9  # 10% below
            max_total_cal = target_calories * 1.1   # 10% above
        else:
            min_total_cal = None
            max_total_cal = None
        
        if min_total_cal is not None:
            if total_nutrition['calories'] < min_total_cal:
                calorie_issues.append(f'Total calories {total_nutrition["calories"]:.1f} kcal below minimum of {min_total_cal:.1f} kcal')
        if max_total_cal is not None:
            if total_nutrition['calories'] > max_total_cal:
                calorie_issues.append(f'Total calories {total_nutrition["calories"]:.1f} kcal exceeds maximum of {max_total_cal:.1f} kcal')
        
        # Create nutritional summary (informational)
        # Process in consistent order: breakfast, lunch, dinner
        per_meal_summary = []
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            if meal_type in meal_nutrition:
                nutrition = meal_nutrition[meal_type]
                per_meal_summary.append(
                    f'{meal_type.capitalize()}: {nutrition["calories"]:.1f} kcal, '
                    f'{nutrition["protein"]:.1f}g protein, {nutrition["fat"]:.1f}g fat, '
                    f'{nutrition["carbs"]:.1f}g carbs, {nutrition["fiber"]:.1f}g fiber'
                )
        
        total_summary = (
            f'Total: {total_nutrition["calories"]:.1f} kcal, '
            f'{total_nutrition["protein"]:.1f}g protein, {total_nutrition["fat"]:.1f}g fat, '
            f'{total_nutrition["carbs"]:.1f}g carbs, {total_nutrition["fiber"]:.1f}g fiber'
        )
        
        nutrition_info = ' | '.join(per_meal_summary) + ' | ' + total_summary
        
        # Build message with nutritional info and any issues
        message_parts = [f'Nutritional information: {nutrition_info}']
        
        # Add missing ingredients information (if any)
        if all_missing:
            message_parts.append(f'Warning: Could not compute nutrition for some ingredients: {", ".join(all_missing)}. Calculated values may be incomplete.')
        
        # Add nutrition mismatches (to ask model to update stated values)
        if caloric_mismatches:
            message_parts.append(f'Please update the recipe because the calories are not near the target: {" | ".join(caloric_mismatches)}')
        
        # Add calorie issues (to ask model to adjust recipes)
        if calorie_issues:
            message_parts.append(f'Please adjust the recipes to meet calorie requirements: {" | ".join(calorie_issues)}')
        
        # Success is based on:
        # 1. Calories being in range (if target specified)
        # 2. Being able to calculate nutrition for at least some meals
        has_calorie_issues = len(calorie_issues) > 0
        # If we have missing ingredients but calculated some nutrition, it's a partial success
        # Only fail if we have critical missing ingredients that prevent all calculations
        has_critical_missing = all_missing and not meal_nutrition
        
        return {
            'success': not has_calorie_issues and not has_critical_missing,
            'message': '. '.join(message_parts)
        }
    
    def _get_nutrient_unit(self, nutrient_key: str) -> str:
        '''Get unit for a nutrient key'''
        if nutrient_key == 'calories':
            return 'kcal'
        return 'g'

    def check_budget(self, meal_plan: str) -> Dict[str, any]:
        '''
        Check total cost across all three recipes against budget.
        Uses model estimation for cost (FDC API doesn't have price data).
        '''
        budget = getattr(self.args, 'budget', None)
        if not budget:
            return {
                'success': True,
                'message': 'No budget specified'
            }
        
        recipes = self.extract_recipes(meal_plan)
        all_ingredients = []
        
        # Collect all ingredients from all recipes
        for meal_type, recipe in recipes.items():
            if not recipe:
                continue
            ingredients = self._parse_ingredients_from_recipe(recipe)
            for qty, unit, name in ingredients:
                all_ingredients.append((qty, unit, name, meal_type))
        
        if not all_ingredients:
            return {
                'success': False,
                'message': 'Could not parse ingredients for budget estimation'
            }
        
        # Use model to estimate cost (FDC API doesn't have price data)
        # Include ALL ingredients from all generated recipes (not just the provided ingredients)
        ingredient_summary = []
        for qty, unit, name, meal_type in all_ingredients:
            ingredient_summary.append(f'{qty} {unit} {name} ({meal_type})')
        
        prompt = f'''Estimate the approximate total cost in USD for ALL these ingredients used across the three recipes: {", ".join(ingredient_summary)}. This includes all ingredients listed in the recipes, not just the main ingredients. Consider typical grocery store prices. Return only a number representing the total cost in USD, no other text.'''
        
        model_response = self.model.ask(prompt)
        if self.verbose:
            print(f'Budget check model response: {model_response}')
        
        # Try to extract number from response
        cost_match = re.search(r'[\d.]+', model_response)
        if cost_match:
            total_cost = float(cost_match.group())
        else:
            return {
                'success': False,
                'message': 'Could not estimate cost from model response'
            }
        
        if total_cost > budget:
            return {
                'success': False,
                'message': f'Total cost ${total_cost:.2f} exceeds budget ${budget:.2f}'
            }
        
        return {
            'success': True,
            'message': f'Total cost ${total_cost:.2f} is within budget ${budget:.2f}'
        }

    def validate(self, meal_plan: str) -> Dict[str, any]:
        '''
        Validation pipeline, does all the checks on the meal plan.
        Returns dict with success and message.
        '''
        all_results = []
        
        # First check: meal plan structure
        structure_check = self.check_meal_plan_structure(meal_plan)
        all_results.append(structure_check)
        
        if not structure_check['success']:
            # If structure is wrong, can't proceed with other checks
            if self.verbose:
                print(f'Validation error: {structure_check["message"]}')
            return {
                'success': False,
                'message': structure_check['message']
            }
        
        # Extract recipes
        recipes = self.extract_recipes(meal_plan)
        
        # Check ingredients across meal plan (not per-recipe)
        ingredients_check = self.check_ingredients(meal_plan)
        all_results.append(ingredients_check)
        
        # Per-recipe checks
        for meal_type, recipe in recipes.items():
            if not recipe:
                continue
            
            # Check sections for each recipe
            sections_check = self.check_sections(recipe, meal_type)
            all_results.append(sections_check)
            
            # Check allergens for each recipe
            allergens_check = self.check_allergens(recipe, meal_type)
            all_results.append(allergens_check)
        
        # Check nutrition (calories, protein, fat, carbs, fiber) per-recipe and total
        nutrition_check = self.check_nutrition(meal_plan)
        all_results.append(nutrition_check)
        
        # Check budget if flag is set
        if getattr(self.args, 'check_budget', False):
            budget_check = self.check_budget(meal_plan)
            all_results.append(budget_check)
        
        # Aggregate results
        all_success = all(r['success'] for r in all_results)
        all_messages = [r['message'] for r in all_results]
        
        if self.verbose:
            print('=== Validation Results ===')
            for result in all_results:
                status = '✓' if result['success'] else '✗'
                print(f'{status} {result["message"]}')
            print('=========================')
        
        final_message = ' | '.join(all_messages) if all_messages else 'All checks passed'
        
        return {
            'success': all_success,
            'message': final_message
        }

    def evaluate_recipe_quality(self, meal_plan: str) -> Dict[str, any]:
        '''
        Final evaluation of recipe quality using the model.
        Evaluates creativity, uniqueness, difficulty, and tastiness with reasoning.
        Only run on the final meal plan.
        Returns dict with scores and reasoning for each metric.
        '''
        prompt = f'''Evaluate this meal plan on the following criteria. For each criterion, provide:
1. A score from 1-10
2. A brief reasoning (2-3 sentences)

Meal Plan:
{meal_plan}

Criteria:
1. CREATIVITY: How creative and innovative are the recipes? Do they use ingredients in unexpected or interesting ways?
2. RECIPE UNIQUENESS: How unique are these recipes compared to typical meal plans? Are they distinctive from common recipes?
3. DIFFICULTY: What is the overall difficulty level of preparing these recipes? Consider complexity of techniques, number of steps, and time required.
4. TASTINESS: Estimate how tasty and appealing these recipes would be. Consider flavor combinations, balance, and overall appeal.

Format your response as:
CREATIVITY: [score]/10 - [reasoning]
UNIQUENESS: [score]/10 - [reasoning]
DIFFICULTY: [score]/10 - [reasoning]
TASTINESS: [score]/10 - [reasoning]'''

        model_response = self.model.ask(prompt)
        if self.verbose:
            print(f'Recipe quality evaluation: {model_response}')
        
        # Parse the response to extract scores and reasoning
        evaluation = {
            'creativity': {'score': None, 'reasoning': ''},
            'uniqueness': {'score': None, 'reasoning': ''},
            'difficulty': {'score': None, 'reasoning': ''},
            'tastiness': {'score': None, 'reasoning': ''}
        }
        
        # Try to parse structured response
        lines = model_response.split('\n')
        i = 0
        current_metric = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line contains a metric header
            if 'CREATIVITY:' in line.upper():
                current_metric = 'creativity'
                match = re.search(r'(\d+(?:\.\d+)?)/10', line)
                if match:
                    evaluation['creativity']['score'] = float(match.group(1))
                # Check for reasoning on same line after dash
                reasoning_parts = re.split(r'\d+/10\s*[-–]\s*', line, maxsplit=1)
                if len(reasoning_parts) > 1:
                    evaluation['creativity']['reasoning'] = reasoning_parts[1].strip()
                # Check for reasoning on next lines
                elif i + 1 < len(lines):
                    reasoning_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not any(keyword in lines[j].upper() for keyword in ['UNIQUENESS:', 'DIFFICULTY:', 'TASTINESS:', 'CREATIVITY:']):
                        reasoning_lines.append(lines[j].strip())
                        j += 1
                    if reasoning_lines:
                        evaluation['creativity']['reasoning'] = ' '.join(reasoning_lines)
                        
            elif 'UNIQUENESS:' in line.upper():
                current_metric = 'uniqueness'
                match = re.search(r'(\d+(?:\.\d+)?)/10', line)
                if match:
                    evaluation['uniqueness']['score'] = float(match.group(1))
                reasoning_parts = re.split(r'\d+/10\s*[-–]\s*', line, maxsplit=1)
                if len(reasoning_parts) > 1:
                    evaluation['uniqueness']['reasoning'] = reasoning_parts[1].strip()
                elif i + 1 < len(lines):
                    reasoning_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not any(keyword in lines[j].upper() for keyword in ['UNIQUENESS:', 'DIFFICULTY:', 'TASTINESS:', 'CREATIVITY:']):
                        reasoning_lines.append(lines[j].strip())
                        j += 1
                    if reasoning_lines:
                        evaluation['uniqueness']['reasoning'] = ' '.join(reasoning_lines)
                        
            elif 'DIFFICULTY:' in line.upper():
                current_metric = 'difficulty'
                match = re.search(r'(\d+(?:\.\d+)?)/10', line)
                if match:
                    evaluation['difficulty']['score'] = float(match.group(1))
                reasoning_parts = re.split(r'\d+/10\s*[-–]\s*', line, maxsplit=1)
                if len(reasoning_parts) > 1:
                    evaluation['difficulty']['reasoning'] = reasoning_parts[1].strip()
                elif i + 1 < len(lines):
                    reasoning_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not any(keyword in lines[j].upper() for keyword in ['UNIQUENESS:', 'DIFFICULTY:', 'TASTINESS:', 'CREATIVITY:']):
                        reasoning_lines.append(lines[j].strip())
                        j += 1
                    if reasoning_lines:
                        evaluation['difficulty']['reasoning'] = ' '.join(reasoning_lines)
                        
            elif 'TASTINESS:' in line.upper():
                current_metric = 'tastiness'
                match = re.search(r'(\d+(?:\.\d+)?)/10', line)
                if match:
                    evaluation['tastiness']['score'] = float(match.group(1))
                reasoning_parts = re.split(r'\d+/10\s*[-–]\s*', line, maxsplit=1)
                if len(reasoning_parts) > 1:
                    evaluation['tastiness']['reasoning'] = reasoning_parts[1].strip()
                elif i + 1 < len(lines):
                    reasoning_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not any(keyword in lines[j].upper() for keyword in ['UNIQUENESS:', 'DIFFICULTY:', 'TASTINESS:', 'CREATIVITY:']):
                        reasoning_lines.append(lines[j].strip())
                        j += 1
                    if reasoning_lines:
                        evaluation['tastiness']['reasoning'] = ' '.join(reasoning_lines)
            
            i += 1
        
        # If parsing failed, store raw response
        if all(v['score'] is None for v in evaluation.values()):
            evaluation['raw_response'] = model_response
        
        return evaluation