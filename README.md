# responsible-ai-audit

# High level goals

## Bias in Language Models

Language models, such as the GPT family, pose a great challenge in bias evaluation. In class, we focused on the simplest setting: binary classifier and binary sensitive attributes. How would you assess bias when these assumptions are changed, such as text as output or multiple values for sensitive attributes within the input? 

https://ai.facebook.com/blog/measure-fairness-and-mitigate-ai-bias/
https://arxiv.org/pdf/2205.12586.pdf
https://huggingface.co/blog/evaluating-llm-bias

Using statistical methods to numerically estimate the bias of a model.

## Attacking LLMs
Large Language Models (LLM) such as GPT demonstrate impressive capabilities. Recently, these LLMs are integrated with other systems and connected to the internet (such as ChatGPT plugins or Google Bard). With the current design, external input (e.g., search results, calendar invitation) is passed to the LLM on the same channel as the instructions within a single prompt. The external input, which should be considered untrusted, could contain instructions to the LLM. This poses a concrete and real vulnerability.

https://medium.com/@danieldkang/attacking-chatgpt-with-standard-program-attacks-7eafdd5c409e
https://twitter.com/random_walker/status/1636925702011252737?s=61&t=fFScuUf755hLYLFoVM8Gvw
https://til.simonwillison.net/llms/python-react-pattern
https://arxiv.org/abs/2210.03629
https://csrc.nist.gov/publications/detail/white-paper/2023/03/08/adversarial-machine-learning-taxonomy-and-terminology/draft
https://learnprompting.org/

## Adversarial Examples as Privacy or Copyright Defenses
Adversarial examples are especially crafted inputs to neural networks that aims to confuse the network but seems benign or unnoticeable to humans. One of the uses of neural networks is for surveillance and face recognition. How could adversarial examples empower individuals and groups to protect their privacy?

https://sandlab.cs.uchicago.edu/fawkes/
https://arxiv.org/pdf/2107.10302.pdf
https://arxiv.org/pdf/2112.04558.pdf
https://nightshade.cs.uchicago.edu/whatis.html


# Steps for project

1. Load LLM from Huggingface
2. Create a chat interface for inputting user prompts (including system prompts), changing hyperparameters
3. Find adversarial prompts from online that jailbreak the LLM and note down prompts that are particularly effective
4. Use uncertainty estimation techniques (https://sites.google.com/view/uncertainty-nlp) for these prompts and visualize the output

sidequest: https://github.com/facebookresearch/ResponsibleNLP/tree/main?tab=readme-ov-file

https://arxiv.org/abs/2306.10193
