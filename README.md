# llm-recipe-generation
For Class


model evaluation:
- creativity
- recipe uniqueness
- difficulty
- random estimate of tastiness ( + reasoning lol )


you need to do pip install . 
then you can run main with python -m modules.main



Example run: 
python -m modules.main --model gpt-oss:20b --max_iterations 3 --ingredients 'rubarb, rice, duck, tomatoes, sesame seeds, persimmon, and oreos' --allergens gluten --budget 50 --check_budget --verbose > output.log 2>&1