# NLP-Text-Generation

**NOTE**: Final copy of this is to be put in https://github.com/gradient-ai/NLP-Text-Generation ; this one in nmb-paperspace is not the public one to be run as a tutorial.

Last updated: Sep 13th 2021

This is our GitHub repository for the Paperspace Gradient NLP Text Generation Tutorial example (**TODO**: Link). It runs the GPT-2 model from HuggingFace: https://huggingface.co/gpt2 .

The example shows:

 - Text generation from a modern deep-learning-based natural language processing model, GPT-2
 - Gradient Projects linked to GitHub repositories
 - Gradient Workflows
 - Triggering a Workflow to rerun based upon a change in the repository, as needed in many production systems
 - Versioned Gradient Datasets as output

The repo contains 2 files: `nlp_text_generation.py` in the main directory, and `nlp_text_generation.yaml`in the `.gradient/workflows` directory. The YAML file contains the Gradient Workflow which in turn calls the Python script.

The Workflow is triggered to run when the YAML file is present in the `.gradient/workflows/` directory, and the repo is linked to the user's Gradient project. The Workflow clones this repo and then in turn calls the Python script. The script outputs the generated text to the file `outputs.txt` in a Gradient-managed Dataset, which the user can then view.

The Workflow runs on the Paperspace HuggingFace NLP container (`paperspace/transformers-gpu:0.4.0`).

## Steps to run this tutorial

**TODO**

### Altering the model settings and triggering a Workflow rerun

The ability to trigger Workflow reruns is useful in several situations, especially more production-oriented ones such as CI/CD, where the state of the collection of code, data, deployments, models, and so on should be consistent.

Here, changing the model settings can be used to trigger a rerun of the model. The 4 values under "Settings" in the `nlp_text_generation.py` script (random seed, maximum text length, number of returned text sequences, and the initial text sentence) can be altered to generate different text.

If the resulting updated version of the `nlp_text_generation.py` file is uploaded to the repo main directory to replace the one present, and the project remains linked to the repo, the Workflow will be rerun. A new `output.txt` file is generated, and placed in a new version of the output Gradient-managed Dataset.

## Next Steps

See the documentation page for this tutorial (**TODO**: Link) for some suggested next steps (e.g., you can run the newer+larger GPT-Neo instead of GPT-2).
