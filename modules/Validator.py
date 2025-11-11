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
            verbose=args.verbose
        )
        self.url = url
        self.verbose = args.verbose
        self.args = args

    def check_sections(self, response:str):
        format = '1. Title, 2. Serving size, 3. Ingredients and amounts, 4. Instructions, 5. Nutritional information, 6. Anything else.'
        prompt = f'''Check if this recipe {response} is in this format: {format}. 
            Return only the string "True" if correct, else return only the missing sections.'''
        response = self.model.ask(prompt)
        if self.verbose: 
            print(f'Check sections: {response}')
        if response == 'True':
            return True
        return response
    
    def check_ingredients(self, response:str):
        ingredients = self.args.ingredients
        prompt = f'''Check if this recipe {response} includes the ingredients: {ingredients} in both the Ingredients and Instructions sections.
            Return only the string "True" if correct, else return only the missing sections.'''
        response = self.model.ask(prompt)
        if self.verbose: 
            print(f'Check sections: {response}')
        if response == 'True':
            return True
        return response

    def check_allergens(self, response:str):
        # if valid, return true
        # else return missing ingredients formatted
        return True 
    
    # add more checks as needed
    def check_budget(self, response:str):
        pass

    def check_calories(self, response:str):
        pass

    def validate(self, response: str):
        '''
        validation pipeline, does all the checks. 
        Returns True if all pass, or else a natural language response of all issues
        '''

        checks = [
            self.check_sections,
            self.check_ingredients,
            self.check_allergens,
            self.check_budget  # can add more
        ]
        
        errors = []
        
        for check in checks:
            result = check(response)
            if result is not True: 
                errors.append(result)
        
        validation = True
        if errors:
            validation = " | ".join(errors)  # Combine all msg
        
        if self.verbose:
            print(f'Validation: {validation}')

        return validation
