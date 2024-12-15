# %% [markdown]
# # imports

# %%
import pandas as pd
import os
import pickle
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM 
import torch

device = torch.device("cuda")


# %% [markdown]
# # set up model

# %%
# init model here:
model_folder = r"/home/dcor/andalrhmana/project1/mistral-mathstral-7b"
os.chdir(r"/home/dcor/andalrhmana/project1")

tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForCausalLM.from_pretrained(
    model_folder,
    torch_dtype= torch.float16    
).to("cuda")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer, 
    device=0
)

print("pipe device is... ",pipe.device)





# %%
# ================================
MODEL = 'mistralai'
MODEL_VERSION = 'mistralai-Mathstral-7B-v0.1'



# ================================


def call_model(prompt, seed):

    # set torch seed
    torch.manual_seed(seed)

    # generate response
    messages = [
    {"role": "user", "content": prompt},
        ]
    
    # determine max_length
    max_length = len(tokenizer(prompt)['input_ids']) + 2048
    response = pipe(messages, max_length=max_length)[0]['generated_text'][-1]['content']

    return response


# %% [markdown]
# # helper functions

# %%

# ================================
PROMPT_1 = "Here is a multiple-choice/true-false question from the Data Structures course. Solve the question.\n"
COT_SUFFIX = "Solve it step by step.\n"
PROMPT_2 = "write the final answer as a single digit or letter, without additional text.\n"

# ================================
def create_prompt_1(question, cot):
    prompt = PROMPT_1 + question + '\n'
    if cot:
        prompt += COT_SUFFIX
    return prompt

# ================================
def create_prompt_2(prompt_1, answer_1):
    return prompt_1 + answer_1 + "\n" + PROMPT_2

# ================================
def format_answer(answer):
    return str(answer).lower().replace(".", "").replace(",", "").replace(" ", "")

# ================================

def digit_to_letter(digit):
    if digit in "12345678":
        number = int(digit)
        return chr(number + 96)
    return digit

def check_answer(answer, correct_answer):
    return digit_to_letter(answer) == digit_to_letter(correct_answer)

# %% [markdown]
# # create the data (for types c, d)

# %%
# set up current working directory

os.chdir(r"/home/dcor/andalrhmana/project1")

# set up dataframes
df_read = pd.read_pickle('phase_2_df_read.pkl')
df_write = pd.read_pickle('phase_2_df_write.pkl')

# set up logs from pickle
logs = pickle.load(open('logs.pkl', 'rb'))

#================================================================================================

# main loop
for index, row in df_read.iterrows():

    # skip irrelevant rows
    if row['question_type'].lower() not in ['c', 'd']:
        continue
    if row['dataset'] != 'tested':
        continue
    if row['finished'] != False:
        continue
    if row['multiple_choice_answer'] == "na" or row['multiple_choice_answer'] == False:
        continue
    
    # 2 rows - with and without cot
    for cot in [False, True]:

        # get row data from df_read
        question = row['question_translation_latex']
        prompt_1 = create_prompt_1(question, cot)
        target_answer = format_answer(row['multiple_choice_answer'])

        # add new row to df_write
        new_index = len(df_write)
        #df_write[new_index] = [None] * len(df_write.columns)

        # fill in the new row in df_write
        df_write.at[new_index, 'question_id'] = index
        df_write.at[new_index, 'model'] = MODEL
        df_write.at[new_index, 'model_version'] = MODEL_VERSION
        df_write.at[new_index, 'question_prompt'] = prompt_1
        df_write.at[new_index, 'prompt_engineering_method'] = "cot" if cot else "none"
        co = "cot" if cot else "none"

        # add answer data for each seed
        for seed in range(5):
            df_write.at[new_index, f"seed_{seed}"] = seed
            model_answer_1 = call_model(prompt_1, seed)
            prompt_2 = create_prompt_2(prompt_1, model_answer_1)
            model_final_answer = format_answer(call_model(prompt_2, seed))
            df_write.at[new_index, f"answer_{seed}"] = model_final_answer
            check = check_answer(model_final_answer, target_answer)
            df_write.at[new_index, f"was_correct_{seed}"] = check
            # add log data
            log = {
		'id': index,
                'model': MODEL, 
                'model_version': MODEL_VERSION,
                'prompt_engineering_method': co,
                'query_1': prompt_1,
                'answer_1': model_answer_1,
                'query_2': prompt_2,
                'seed': seed,
                'final_answer': model_final_answer,
		'target_answer': target_answer,
		'was_correct': check
            }
            logs.append(log)

    # mark row as finished
    df_read.at[index, 'finished'] = True

    # save dataframes  
    df_read.to_pickle('phase_2_df_read.pkl')
    df_write.to_pickle('phase_2_df_write.pkl')
    pickle.dump(logs, open('logs.pkl', 'wb'))

    # print progress
    print(f"finished row {index} out of {len(df_read)}")



