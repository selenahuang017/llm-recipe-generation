class Validator:
    '''
    Takes a string in as a response, and validates the response
    '''
    def __init__(self, args: dict,
        model: str = 'qwen3:4b',
        url: str = 'http://ollama.loweffort.meme/api/chat', 
        verbose: bool = False):
        self.model = model
        self.url = url
        self.verbosity = verbose
        self.args = args

    def check_sections(response:str):
        # if valid, return true
        # else return missing sections formatted
        return True
    
    def check_ingredients(response:str):
        # if valid, return true
        # else return missing ingredients formatted
        return True 

    def check_allergens(response:str):
        # if valid, return true
        # else return missing ingredients formatted
        return True 
    
    # add more checks as needed
    def check_budget(response:str):
        pass

    def check_calories(response:str):
        pass

    def validate(response: str):
        '''
        Takes in the response as a string
        returns true if everything is valid, else returns a string with new changes'''

        return True
