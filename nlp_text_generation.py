# NLP text generation
#
# Uses GPT-2 model from HuggingFace: https://huggingface.co/gpt2
#
# Last updated: Sep 08th 2021

# Setup
from transformers import pipeline, set_seed

# Settings
random_seed = 42
max_length = 30
num_return_sequences = 5
initial_sentence = "Hello, I'm a language model,"

# Create generator that uses GPT-2
generator = pipeline('text-generation', model='gpt2')

# Random seed for text generation
set_seed(random_seed)

# Run the generator
output = generator(initial_sentence, max_length = max_length, num_return_sequences = num_return_sequences)

# Write the output to a file
with open('output.txt', 'w') as f:
    ival = 1
    for val in output:
        print('---\nOutput {} of {}\n---\n'.format(ival, num_return_sequences), file=f)
        print(val['generated_text'], file=f)
        if ival < num_return_sequences: print(file=f)
        ival += 1

print('Done')
