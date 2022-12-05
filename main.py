######################## META-VARIABLES ########################
COHERE_API_KEY = '' # TODO
RESUMES_PATH = 'resumes'

############################ IMPORTS ############################

# TODO NOTE: You may need to download these modules
# pip install textract & pip install spacy & pip install nltk & pip install cohere

# For determining which action to complete
import sys
import argparse

# For reading resume files
from os import listdir, rename
from os.path import abspath, isfile, join
# from PyPDF2 import PdfReader
from textract import process

# For NLP
import spacy
from spacy.cli.download import download
try: # make sure 'en_core_web_lg' is downloaded from spacy
    spacy.load('en_core_web_lg')
except OSError:
    download(model='en_core_web_lg')
    
# For NER and parts of speech recognition
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords', 'wordnet', 'omw-1.4'], quiet=True)

# For generating recommendations
import cohere

# Data manipulation
import pandas as pd
import numpy as np

###################### IMPORT LIBRARY STATE ######################

# Prepare data for resume language processing; need access in multiple functions
nlp = spacy.load('en_core_web_lg')
skill_pattern_path = 'jz_skill_patterns.jsonl'
ruler = nlp.add_pipe('entity_ruler')
ruler.from_disk(skill_pattern_path)

co = cohere.Client(COHERE_API_KEY)

######################## READING RESUMES ########################

def extract_text_from_doc(doc_name):
    """
    Extracts all of the raw text data from a file and returns it.
        doc_name: string => path of the file to read from
    """
    return process(abspath(doc_name)).decode('utf-8')

######################## PARSING RESUMES ########################

def get_skills(text):
    """
    Extracts a list of the hard skills found in a text body.
        nlp: [object Language] => from spacy.load
        text: string => the text body to parse from
    """
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            subset.append(ent.text)
    myset.append(subset)
    return subset

################### REMOVE NOISE FROM RESUMES ###################

def unique_set(x):
    """
    Removes duplicate values from a list and returns the modified list.
        x: list<T> => the list to filter
    """
    return list(set(x))

def identify_text(resume_data):
    """
    Splits words, removes trailing whitespace, hyperlinks, etc., and attempts to lemmatize
    words found in resume_data['doc_text']. Returns a list of the lemmatized words.
        resume_data: Pandas.DataFrame<T extends { doc_text: string }> => from main()
    """
    clean = []
    for i in range(resume_data.shape[0]):
        review = re.sub( # remove whitespace, hyperlinks, etc.
            '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
            ' ',
            resume_data['doc_text'].iloc[i],
        )
        review = review.lower()
        review = review.split()
        lm = WordNetLemmatizer()
        review = [
            lm.lemmatize(word) # determine part of speech for the word
            for word in review
            if not word in set(stopwords.words('english')) # ignores "a", "the", "in", etc.
        ]
        review = ' '.join(review)
        clean.append(review)
    return clean

##################### GENERATE SUGGESTIONS #####################

def generate_suggestion(skills, skill_level):
    """
    Makes a call to the Co:here API, providing it with the skills list and skill level of a user.
    Returns a string representation of the suggested resume item for the user whose skills and skill level were provided.
        skills: string[] => list of skills from the resume_data record
        skill_level: 'Beginner' | 'Intermediate' | 'Advanced' => level of industry knowledge this record user has
    """
    with open('prompt.txt', 'r') as file:
        return co.generate(
            prompt=f"""{file.read()}
Skills: {', '.join(skills)}
Level: {skill_level}
Description: """.strip(),
            model='xlarge',
            max_tokens=50,
            temperature=0.8,
            p=1,
            k=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["---"],
            return_likelihoods='NONE',
        ).generations[0].text.split('\n')[0]

############################# MAIN #############################

def handle_all_files():
    # Get a list of all resume file names in the "resumes" folder
    all_resumes = [f for f in listdir(RESUMES_PATH) if isfile(join(RESUMES_PATH, f))]

    # Sanitize resume file names for parser tools
    for f in all_resumes:
        old_name = f
        new_name = f.replace(' ', '_').lower()
        rename(join(RESUMES_PATH, old_name), join(RESUMES_PATH, new_name))
        index_of_old = all_resumes.index(old_name)
        all_resumes[index_of_old] = new_name


    # Grab all of the text from the resumes
    resume_data = pd.DataFrame({'doc_name': f, 'doc_text': extract_text_from_doc(join(RESUMES_PATH, f))} for f in all_resumes)
    resume_data = resume_data.reindex(np.random.permutation(resume_data.index))

    # Parse the resumes
    resume_data['cleaned_text'] = identify_text(resume_data)
    resume_data['skills'] = resume_data['cleaned_text'].str.lower().apply(get_skills)
    resume_data['skills'] = resume_data['skills'].apply(unique_set)

    # Make recommendations for each resume
    for i in range(len(resume_data)):
        print(f'\033[1;3mFor the document {resume_data.iloc[i].doc_name} consider adding:\033[0m')
        print(generate_suggestion(resume_data.iloc[0]['skills'], 'Beginner') + '\n')

def handle_one_file(file_name):
    # Sanitize resume file names for parser tools
    old_name = file_name
    new_name = file_name.replace(' ', '_').lower()
    rename(join(RESUMES_PATH, old_name), join(RESUMES_PATH, new_name))

    # Grab all of the text from the resumes
    resume_data = pd.DataFrame({'doc_name': new_name, 'doc_text': extract_text_from_doc(join(RESUMES_PATH, new_name))}, index=[0])

    # Parse the resumes
    resume_data['cleaned_text'] = identify_text(resume_data)
    resume_data['skills'] = resume_data['cleaned_text'].str.lower().apply(get_skills)
    resume_data['skills'] = resume_data['skills'].apply(unique_set)

    # Make recommendations for resume
    print(f'\033[1;3mFor the document {resume_data.iloc[0].doc_name} consider adding:\033[0m')
    for i in range(5):
        print(generate_suggestion(resume_data.iloc[0]['skills'], 'Beginner') + '\n')

if __name__ == '__main__':
    # Determine CLI flags
    parser = argparse.ArgumentParser(description='Provides resume suggestions by scanning your resume for your skills and by leveraging NLP AI models.')
    parser.add_argument('-f', '--files', help="Specify how many files to generate suggestions for.", required=True, choices=['single', 'all'])
    parser.add_argument('-r', '--resume', help="If -f flag set to 'single', the resume name to generate suggestions for. File is expected to be found in the /resumes directory.", metavar="file_name")
    args = parser.parse_args()

    # Require "-r" flag if "-f" flag is set to "single"
    if args.files == 'single' and not args.resume:
        print('usage: main.py [-h] -f {single,all} [-r file_name]')
        print('main.py: error: the following arguments are required with the "-f single" flag: -r/--resume')
        sys.exit(1)
    # Expect no "-r" flag when "-f" flag is set to "all"
    if args.files == 'all' and args.resume:
        print('usage: main.py [-h] -f {single,all} [-r file_name]')
        print('main.py: error: unexpected flag "-r" with "-f" flag set to "all"')
        sys.exit(2)
    
    # Route to requested commands
    if args.files == 'single':
        handle_one_file(args.resume)
    else:
        handle_all_files()