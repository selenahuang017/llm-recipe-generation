import requests
import json
from typing import List, Dict, Optional

class OllamaChatSession:
    '''
    Manages a conversational session with an Ollama model using /api/chat.
    Maintains memory (chat history) across turns and supports streaming output.
    '''

    def __init__(self,
                 model: str = 'gpt-oss:20b',
                 url: str = 'https://ollama.loweffort.meme/api/chat',
                 system_prompt: Optional[str] = None,
                 stream: bool = True,
                 verbose: bool = False):
        self.model = model
        self.url = url
        self.stream = stream
        self.messages: List[Dict[str, str]] = []
        self.verbosity = verbose
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})

    def get_initial_user_message(self):
        for msg in self.messages:
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def initial_request(self, args: dict) -> str:
        """
        Generate the initial meal plan request with a strict template
        so the model always outputs BREAKFAST, LUNCH, and DINNER clearly.
        """

        ingredients = (
            f"You must create a meal plan using ALL of these ingredients "
            f"at least once across the three meals: {args.ingredients}."
        )

        allergens = (
            f"These allergens must be strictly avoided: {args.allergens}."
            if args.allergens
            else "There are no allergens to avoid."
        )

        calories = (
            f"The entire meal plan should total approximately {args.calories} calories"
            f"(±10% tolerance)."
            if getattr(args, "calories", None)
            else ""
        )

        budget = (
            f"The total ingredient cost must stay under ${args.budget}."
            if getattr(args, "budget", None)
            else ""
        )

        # 🔥 STRICT STRUCTURE TEMPLATE — models follow this reliably
        structure = """
    You MUST output your response using EXACTLY the following structure:

    ## BREAKFAST
    1. Title
    2. Serving size
    3. Ingredients and amounts
    4. Instructions
    5. Nutritional information
    6. Anything else

    ## LUNCH
    1. Title
    2. Serving size
    3. Ingredients and amounts
    4. Instructions
    5. Nutritional information
    6. Anything else

    ## DINNER
    1. Title
    2. Serving size
    3. Ingredients and amounts
    4. Instructions
    5. Nutritional information
    6. Anything else

    Do NOT add extra sections.
    Do NOT change the section numbers.
    Do NOT change the section titles.
    """

        prompt = f"""
    Your task is to generate a full 3-meal daily meal plan.

    {ingredients}
    {allergens}
    {calories}
    {budget}

    Each meal must follow this recipe schema:
    1. Title
    2. Serving size
    3. Ingredients and amounts (measure solid ingredients in grams and liquid ingredients in ml)
    4. Instructions
    5. Nutritional information
    6. Anything else

    {structure}

    Use ALL required ingredients across the three meals.
    Return ONLY the structured meal plan — no commentary before or after.

    In the ingredients section (#3), format every ingredient as:
    <quantity> <unit> <ingredient name>

    Examples:
    150 g eggs
    100 g spinach
    10 g olive oil
    100 ml orange juice

    Do NOT put the ingredient name before the quantity.
    Do NOT use colons.
    Do NOT include parentheses.
    Do NOT include approximations.
    """

        if self.verbosity:
            print(f"Prompt: {prompt}")

        return self.ask(prompt)

    
    def request(self, validation:str) -> str: 
        '''request from validator, has additional information'''
        # format here for return request.
        prompt = f'Update the previous meal plan, but note the following: {validation}'

        return self.ask(prompt)

    def ask(self, prompt: str) -> str:
        '''Send a new user message and receive the assistant's response.'''
        self.messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model,
            'messages': self.messages,
            'stream': self.stream,
            }

        output = []
        with requests.post(self.url, json=data, stream=self.stream) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                j = json.loads(line)
                msg = j.get('message', {}).get('content', '')
                if msg:
                    output.append(msg)
        #print()
        full_response = ''.join(output)

        # Store assistant reply in memory
        self.messages.append({'role': 'assistant', 'content': full_response})
        return full_response

    def save_memory(self, path: str = 'ollama_memory.json'):
        '''Persist chat memory to disk.'''
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2)

    def load_memory(self, path: str = 'ollama_memory.json'):
        '''Load previous memory from disk.'''
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
        except FileNotFoundError:
            print(f'No memory file found at {path}')

    def clear_memory(self):
        '''Reset chat memory except for any system message.'''
        sys_msgs = [m for m in self.messages if m['role'] == 'system']
        self.messages = sys_msgs