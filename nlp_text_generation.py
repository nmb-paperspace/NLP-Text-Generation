# NLP text generation
#
# Uses GPT-2 model from HuggingFace: https://huggingface.co/gpt2
#
# Last updated: Sep 08th 2021

# Setup
from transformers import pipeline, set_seed

# Create text generator
generator = pipeline('text-generation', model='gpt2')

# Random seed for text generation
set_seed(42)

# Generate text and output to file
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(output)

print('Done')
