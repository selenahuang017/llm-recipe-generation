class OllamaChatSession:
    '''
    Manages a conversational session with an Ollama model using /api/chat.
    Maintains memory (chat history) across turns and supports streaming output.
    '''

    def __init__(self,
                 model: str = 'gpt-oss:20b',
                 url: str = 'http://ollama.loweffort.meme/api/chat',
                 system_prompt: Optional[str] = None,
                 stream: bool = True):
        self.model = model
        self.url = url
        self.stream = stream
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})

    def initial_request(self, args:dict) -> str: 
        #TODO : create format

        # format here for initial request: currently simple
        prompt = "Create a recipe with these ingredients:" + args.ingredients

        return self.ask(prompt, verbose=args.verbose)
    
    def request(self, validation:str) -> str: 
        '''request from validator, has additional information'''
        # format here for return request.
        prompt = "Update the previous recipe, but note the following:" + validation

        return self.ask(prompt)

    def ask(self, prompt: str, verbose: bool = True) -> str:
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
                    if verbose:
                        print(msg, end='', flush=True)
                    output.append(msg)
        print()
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