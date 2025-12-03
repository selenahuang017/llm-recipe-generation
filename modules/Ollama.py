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

    def initial_request(self, args:dict) -> str: 
        ingredients = f'Create a meal plan with these ingredients: {args.ingredients}. All ingredients must be used across the three meals (breakfast, lunch, and dinner), but the recipes should be diverse and different from each other.'
        allergens = f'Note to avoid these allergens: {args.allergens}.' if args.allergens else ''
        format = 'Put each recipe in this format: 1. Title, 2. Serving size, 3. Ingredients and amounts, 4. Instructions, 5. Nutritional information, 6. Anything else.'
        meal_plan_format = 'Structure the meal plan as follows: BREAKFAST: [recipe format] LUNCH: [recipe format] DINNER: [recipe format]'
        budget = f'Make sure to stay under the budget of ${args.budget} if possible. If not, indicate so.' if getattr(args, 'budget', None) else ''
        calories = f'Return the projected calories and aim for {args.calories}. Indicate if over or under the projection.' if getattr(args, 'calories', None) else ''

        prompt = f"""{ingredients} {meal_plan_format} {format} {allergens} {budget} {calories}"""
        if self.verbosity:
            print(f'Prompt: {prompt}')
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