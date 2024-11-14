import os, re, shutil
import pandas as pd
from datetime import datetime
from fastapi import status, HTTPException
from langdetect import detect
import asyncio
from aiofiles import open as aio_open
from langchain_community.llms.ollama import Ollama
import aiofiles


ollamaModel = Ollama(model="qwen2.5:7b", keep_alive=-1)

agents = ["langchain", "llamaindex"]
llm_models = ["mistral", "llama2", "llama3", "llama3.1", "stablelm2", "aya", "qwen2.5:7b"]
DIRECTORY_PATH = "data"
language_code = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'gl': 'Galician',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'id': 'Indonesian',
    'ga': 'Irish',
    'it': 'Italian',
    'ja': 'Japanese',
    'kn': 'Kannada',
    'kk': 'Kazakh',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mk': 'Macedonian',
    'ms': 'Malay',
    'mt': 'Maltese',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sr': 'Serbian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'es': 'Spanish',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'cy': 'Welsh',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'zu': 'Zulu'
}





# this function is use for verify agent name, is agent not match then it assign langchain as agent
def verify_agent(agent: str):
    if agent in agents:
        return agent
    else:
        agent = "langchain"
        return agent
        

# this function is use for verify llm name, is llm not match then it assign mistral as agent
def verify_llm(model: str):
    if model in llm_models:
        return model
    else:
        model = "qwen2.5:7b"
        return model

    
# this function is handel error and log/print error in terminal/server
def handel_exception(error):
    print("---------------ERROR--------------")
    print(error)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{error}")


# this function convert excel file to json file
def excel_to_json(data_path, filename):
    df = pd.read_excel(f"{data_path}/{filename}", engine='openpyxl')
    df.to_json(f"{data_path}/{filename.split('.')[0]}.json", orient='records', indent=4)


# create collection name valid
def sanitize_collection_name(name):
    # Replace all non-alphanumeric characters with underscores
    sanitized_name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    
    # Ensure it starts and ends with an alphanumeric character
    if not sanitized_name[0].isalnum():
        sanitized_name = 'a' + sanitized_name[1:]
    if not sanitized_name[-1].isalnum():
        sanitized_name = sanitized_name[:-1] + 'a'
    
    # Ensure length is between 3 and 63 characters
    if len(sanitized_name) < 3:
        sanitized_name = sanitized_name.ljust(3, 'a')
    elif len(sanitized_name) > 63:
        sanitized_name = sanitized_name[:63]
    
    return sanitized_name


# change document name
async def rename_and_save_file(file_obj, document_name: str = None, version_id: str = None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_extension = os.path.splitext(file_obj.filename)[1]

    if document_name and version_id:
        new_file_name = f"{current_time}_{document_name}_v{version_id}{file_extension}"
    else:
        new_file_name = f"{current_time}{file_extension}"

    #new_file_name = os.path.basename(new_file_name)
    new_file_path = os.path.join(DIRECTORY_PATH, new_file_name)
    if not os.path.exists(DIRECTORY_PATH):
        os.makedirs(DIRECTORY_PATH)

    async with aiofiles.open(new_file_path, "wb") as f:
        content = await file_obj.read()
        await f.write(content)
    return new_file_path, new_file_name


# regex for user query to find metadata
def find_word_in_sentence(word, sentence):
    # Use re.IGNORECASE to make the search case-insensitive
    pattern = re.compile(re.escape(word), re.IGNORECASE)
    match = pattern.search(sentence)
    return match is not None


# Delete document/file
def delete_document(path, filename):
    try:
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"File {filename} has been deleted successfully.")
            return True
        else:
            print(f"File {filename} does not exist in the directory {path}.")
            return False

    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")
        return False


# Delete Directory
def delete_directory(directory_path):
    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Delete the directory and its contents
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' has been deleted successfully.")
            return True
        else:
            print(f"Directory '{directory_path}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred while deleting the directory: {e}")
        return False



def matching_percentage(str1, str2):
    # Normalize the strings
    str1 = str1.lower().replace(" ", "")
    str2 = str2.lower().replace(" ", "")
    
    # Find common characters
    common_chars = set(str1) & set(str2)
    common_count = sum(min(str1.count(char), str2.count(char)) for char in common_chars)
    
    # Calculate total characters in the longest string
    max_length = max(len(str1), len(str2))
    
    # Calculate the percentage
    percentage = (common_count / max_length) * 100
    percentage = int(percentage)
    
    return percentage



def language_detaction(text: str):
    try:
        res_langcode = detect(text)
        res_language = language_code.get(res_langcode, "Unknown Language")
    except Exception as e:
        res_langcode = "und"
        res_language = "Can't Detected"
        print(e)

    return res_langcode, res_language


