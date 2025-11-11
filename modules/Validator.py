from .Ollama import OllamaChatSession

class Validator:
    '''
    Takes a string in as a response, and validates the response
    '''
    def __init__(self, args: dict,
        url: str = 'http://ollama.loweffort.meme/api/chat'):
        self.model = args.val_model # can be different from regular request model
        self.url = url
        self.verbosity = args.verbose
        self.args = args

    def check_sections(self, response:str):
        # if valid, return true
        # else return missing sections formatted
        # 1. Title, 2. serving size, 3. Ingredients, 4. Instructions, 5. Nutritional information, 6. Anything else
        return True
    
    def check_ingredients(self, response:str):
        # if valid, return true
        # else return missing ingredients formatted
        return True 

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
        
        if self.verbosity:
            print(f'Validation: {validation}')

        return validation
