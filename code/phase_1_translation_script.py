# imports

import openai
from openai import OpenAI
import os
import pandas as pd
import random
import pickle

# change to the directory where the data is
# os.chdir(r"C:\Users\USER\Desktop\vscode_env\archive\misc\github") 

#===========================================================================#

# OPEN AI

model = "gpt-4o-2024-08-06"
key = "your-key-goes-here"
prompt_header = r"""
translate the following to english. your response should contain only the latex code for the translation - nothing else. mathematical expressions should be properly formated. assume the document has already started with the following code:
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}

\title{Title}

\begin{document}

\maketitle

\section{Translated question no. 1}
% your response here

the code should compile cleanly without errors. your code will be pasted into the latex document, so don't include anything that is not the latex code for the translation.

question to translate:
"""
client = OpenAI(api_key=key)


def translate_with_api(question):
    prompt = prompt_header + question
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

#===========================================================================#

# HELPER FUNCTIONS

def translate(s):
    return translate_with_api(s)

def get_filename(s):
    s = str(s)
    # 1
    if s == "https://docs.google.com/document/d/1m9QlOxr3Vv3z0TqwfYVvZtR-cgSUOuuc":
        return "ds01ab"
    # 2
    if s == "https://docs.google.com/document/d/1oTX1kHKgw8iBufPGBn9d1Tzmll5dgOIz":
        return "ds01ba"
    # 3
    if s == "https://docs.google.com/document/d/1WrQhJhuDshuffTfuKpQvOh2PtYD6TLs9":
        return "ds01bb-semi-resemble-ab-ba"
    # 4
    if s == "https://docs.google.com/document/d/1ey-5vr7nIIagweUCwpKNy3tIrvxAFsz7":
        return "ds02aa-ans"
    # 5
    if s == "https://docs.google.com/document/d/1UoJdnerEFxl6OfCgPoKIRk-N_cj2xNo1":
        return "ds02aa"
    # 6
    if s == "https://docs.google.com/document/d/1ARR0v5jx5FzYIqxWEbptDtL2r0-6A1B9":
        return "ds02ba"
    # 7
    if s == "https://docs.google.com/document/d/1uYfTozVQ7fEkuWRJr2g7R7ZqkkAKm4Jd":
        return "ds03aa-Stud-sol"
    # 8
    if s == "https://docs.google.com/document/d/1uYfTozVQ7fEkuWRJr2g7R7ZqkkAKm4Jd":
        return "ds03aa"
    # 9
    if s == "https://docs.google.com/document/d/1s1iM0bGmR04O6J6zz7iGmjYLOXhp4LqZ":
        return "ds03abbb"
    # 10
    if s == "https://docs.google.com/document/d/1DsdNOK2cTVO4bctkBjFCBCsjpJEvM3qR":
        return "ds03ba-SOL"
    # 11
    if s == "https://docs.google.com/document/d/1XDB0oWMfvAxlym7XTAow0iF4PbwLN3qE":
        return "ds04aa-ans"
    # 12
    if s == "https://docs.google.com/document/d/1LSxoxH2sEItqkTRlrCstcMk02LN676Jz":
        return "ds04ab+bb+ca-ans"
    # 13
    if s == "https://docs.google.com/document/d/1Ze9Vu1t-M8eGfSRY5m6GdpbIz9KUxnzN":
        return "ds05aa-ans"
    # 14
    if s == "https://docs.google.com/document/d/1v9YBu71MdYUn2k2hxj7fgF-DeJsqsoZM":
        return "ds05ab+bb+answer at end"
    # 15
    if s == "https://drive.google.com/file/d/1fFT6cjw9-7q6nuBj8uY37qEtxNLrGkw0":
        return "ds05ab+bb-ans"
    # 16
    if s == "https://docs.google.com/document/d/18ncdSZ5zZsryYGUoPJwpZDAVetmCCUf-":
        return "ds05ab+bb"
    # 17
    if s == "https://drive.google.com/file/d/14YLvZq3ucViky7WwmaP1rnPJprOZcKx0":
        return "ds05ba-sol-from-Vaad"
    # 18
    if s == "https://drive.google.com/file/d/1QtmyoyTbfXNOvd6YwpSuFRVBcGBsZW5W":
        return "ds05ba-sol-from-Vaad"
    # 19
    if s == "https://docs.google.com/document/d/1altMUQBVybKQpoFz-rQsDmI_e0FWhzbu":
        return "ds06aa-ans"
    # 20
    if s == "https://drive.google.com/file/d/1KmfglCiZ4iBUbm4wItEeGLSeehWRl5dD":
        return "ds06ab-Onlyans"
    # 21
    if s == "https://docs.google.com/document/d/1N_8TPis-QeSok6nc6qH27CNnIWn_kOpH":
        return "ds06ab+ans-at-end"
    # 22
    if s == "https://docs.google.com/document/d/1F0vu6OlXCIefyxtoWYLLiJN2bNi8bL6i":
        return "ds06b_moed_a_solution"
    # 23
    if s == "https://docs.google.com/document/d/1KYPVPcemr20Z7bG9QDBCTCrVsMuMWnyH":
        return "ds06b_moed_b_solution"
    # 24
    if s == "https://docs.google.com/document/d/15av3BYO1sip8j7xbHE3cwGQOEe2WyO9U":
        return "ds07a-a-ans"
    else: 
        raise ValueError("no such link")
    
def make_title(row_number):
    title = "Filename: " + get_filename(df.at[row_number, "Url of the exam file in our shared directory"]) + ", "
    title = title.replace("_", "\_")
    title += "Year: " + str(df.at[row_number, "Exam year"]) + ", "
    title += "Semester: " + str(df.at[row_number, "Semester"]) + ", "
    title += "Moed: " + str(df.at[row_number, "Moed"]) + ", "
    title += "Question Number: " + str(df.at[row_number, "Question number in the exam"])
    return title


def make_latex_item(row_number):
    s = "\section{" + make_title(row_number) + "}\n"
    s += str(df.at[row_number, "Translation of the question to english in Latex format"]) + "\n"
    return s

def make_latex_answer_item(row_number):
    s = "\section{" + make_title(row_number) + "}\n"
    s += str(df.at[row_number, "Translation of open answer or short explanation"]) + "\n"
    return s

def make_latex_file(df):

    s = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}

\title{T1 Translated Questions}
\author{Submitted by Abed and Orr}
\date{}

\begin{document}

\maketitle

"""

    for i in range(len(df)):
        if df.at[i, "Translation of the question to english in Latex format"] != "N\A":
            s += make_latex_item(i)

    s += r"\end{document}" + "\n"

    return s

def make_latex_answers_file(df):

    s = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}

\title{T1 Translated Questions}
\author{Submitted by Abed and Orr}
\date{}

\begin{document}

\maketitle

"""

    for i in range(len(df)):
        if df.at[i, "Translation of open answer or short explanation"] != "N\A":
            s += make_latex_answer_item(i)

    s += r"\end{document}" + "\n"

    return s

#===========================================================================#

# load dataframe
df = pd.read_excel("question_data1a.xlsx")

# creating a list questions that are of type B, C, or D
BCD_questions = []
for i, row in df.iterrows():
    if row["Type of question"] in ["B", "C", "D"]:
        BCD_questions.append(i)

# choosing a 2/3 of the questions in random
random.shuffle(BCD_questions)
question_to_translate = BCD_questions[:int(len(BCD_questions) * 2 / 3)]

# translating the questions
for i in question_to_translate:
    df.at[i, "Translation of the question to english in Latex format"] = translate(df.at[i, "the question in hebrew "])
    df.at[i, "the request for the translation included both the question and answer"] = True
    print("translated question number", i)

# translating the answers
for i in question_to_translate:
    if df.at[i, "Has solution?"] != True:
        print("no solution")
        continue
    if df.at[i, "Open answer or short explanation"] == "N\A":
        print("no open answer")
        continue
    df.at[i, "Translation of open answer or short explanation"] = translate(df.at[i, "Open answer or short explanation"])
    print("translated question number", i)

# replacing nan with "N\A" in all columns
df = df.fillna("N\A")

# removing the latex header from data
for i in question_to_translate:
    if df.at[i, "Translation of the question to english in Latex format"] != "N\A":
        data = df.at[i, "Translation of the question to english in Latex format"]
        # clean the latex header
        data = data.replace("```latex", "")
        data = data.replace("```", "")
        data = data.replace(r"\end{document}", "")
        df.at[i, "Translation of the question to english in Latex format"] = data
    if df.at[i, "Translation of open answer or short explanation"] != "N\A":
        data = df.at[i, "Translation of open answer or short explanation"]
        # clean the latex header
        data = data.replace("```latex", "")
        data = data.replace("```", "")
        data = data.replace(r"\end{document}", "")
        df.at[i, "Translation of open answer or short explanation"] = data

# saving the dataframe as a pickle
with open("question_data1a_processed.pkl", "wb") as f:
    pickle.dump(df, f)

# saving the dataframe as an excel file
df.to_excel("question_data1a_processed.xlsx", index=False)

#===========================================================================#

