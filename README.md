# TAUPro

This repository includes the code and data used in our project.

the submitted paper

the "code" folder includes:
- phase_1_translation_script.py - a code for receiving a dataframe and translating the questions and answers to English, via the OpenAI API.
- phase_2_main_llama.py - code that reads a dataframe with translated questions and produces a dataframe with the Llama's responses, as well as a log file.
- phase_2_main_mistralcode.py - similar to the previous one, but adjusted to the Mistral model.

the "data_phase1" and "data_phase2" folders include data necessary for running the code scripts.
