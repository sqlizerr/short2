import json
import os
import re
from time import sleep, time

from google.cloud import vertexai
import yaml

from shortGPT.config.api_db import ApiKeyManager
project_id = "gen-lang-client-0261891613"
location = "asia-south1-a"

client = vertexai.Client(project=project_id, location=location)

model_name = "gemini-1.5-flash-001"
# def num_tokens_from_messages(texts, model="gemini-1.5-flash"):
#     """Returns the number of tokens used by a list of messages."""
#     try:
#         encoding = tiktoken.encoding_for_model(model)
#     except KeyError:
#         encoding = tiktoken.get_encoding("cl100k_base")
#     if model == "gemini-1.5-flash":  # note: future models may deviate from this
#         if isinstance(texts, str):
#             texts = [texts]
#         score = 0
#         for text in texts:
#             score += 4 + len(encoding.encode(text))
#         return score
#     else:
#         raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
#         See https://github.com/openai/openai-python/blob/main/chatml.md for information""")

def num_tokens_from_messages(texts, model_name):
  """
  Returns the number of tokens used by a list of messages using a Gemini-compatible tokenizer 
  in the context of Vertex AI.

  Args:
      texts (str or list of str): Input text(s) to tokenize.
      model_name (str): The name of the Gemini model on Vertex AI.

  Returns:
      int: Total number of tokens used by the input messages.
  """

  if isinstance(texts, str):
    texts = [texts]

  total_tokens = 0
  

  for text in texts:
    # Add 4 tokens per message as overhead (if needed)
    total_tokens += 4 

    # Use Vertex AI client to count tokens
    token_count_response = +client.count_tokens(
        model_name=model_name,
        content=text
    )
    total_tokens += token_count_response.token_count

  return total_tokens

def extract_biggest_json(string):
    json_regex = r"\{(?:[^{}]|(?R))*\}"
    json_objects = re.findall(json_regex, string)
    if json_objects:
        return max(json_objects, key=len)
    return None


def get_first_number(string):
    pattern = r'\b(0|[1-9]|10)\b'
    match = re.search(pattern, string)
    if match:
        return int(match.group())
    else:
        return None


def load_yaml_file(file_path: str) -> dict:
    """Reads and returns the contents of a YAML file as dictionary"""
    return yaml.safe_load(open_file(file_path))


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

from pathlib import Path

def load_local_yaml_prompt(file_path):
    _here = Path(__file__).parent
    _absolute_path = (_here / '..' / file_path).resolve()
    json_template = load_yaml_file(str(_absolute_path))
    return json_template['chat_prompt'], json_template['system_prompt']


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# def gpt3Turbo_completion(chat_prompt="", system="You are an AI that can give the answer to anything", temp=0.7, model=model, max_tokens=1000, remove_nl=True, conversation=None):
#     # openai.api_key = ApiKeyManager.get_api_key("OPENAI")
#     genai.configure(api_key= ApiKeyManager.get_api_key("OPENAI"))
#     max_retry = 5
#     retry = 0
#     while True:
#         try:
#             if conversation:
#                 messages = conversation
#             else:
#                 messages = [
#                     {"role": "system", "content": system},
#                     {"role": "user", "content": chat_prompt}
#                 ]
#             response = model.generate_content(
#                 messages,
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=temp,
#                     max_output_tokens=max_tokens))
#             text = response.choices[0].message.content.strip()
#             if remove_nl:
#                 text = re.sub('\s+', ' ', text)
#             filename = '%s_gpt3.txt' % time()
#             if not os.path.exists('.logs/gpt_logs'):
#                 os.makedirs('.logs/gpt_logs')
#             with open('.logs/gpt_logs/%s' % filename, 'w', encoding='utf-8') as outfile:
#                 outfile.write(f"System prompt: ===\n{system}\n===\n"+f"Chat prompt: ===\n{chat_prompt}\n===\n" + f'RESPONSE:\n====\n{text}\n===\n')
#             return text
#         except Exception as oops:
#             retry += 1
#             if retry >= max_retry:
#                 raise Exception("Gemini error: %s" % oops)
#             print('Error communicating with Gemini API:', oops)
#             sleep(1)

def gpt3Turbo_completion(chat_prompt="", system="You are an AI that can give the answer to anything", temp=0.7, model_name=model_name, max_tokens=1000, remove_nl=True, conversation=None):
  """
  Performs a chat completion using the specified Vertex AI Gemini model.

  Args:
      chat_prompt (str, optional): The user's input for the chat. Defaults to "".
      system (str, optional): The system instruction for the model. Defaults to "You are an AI that can give the answer to anything".
      temp (float, optional): Temperature for the model's creativity. Defaults to 0.7.
      model_name (str, optional): The name of the Vertex AI Gemini model. Defaults to "text-bison".
      max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 1000.
      remove_nl (bool, optional): Whether to remove extra newline characters. Defaults to True.
      conversation (list, optional): A list of messages representing the conversation history. Defaults to None.

  Returns:
      str: The generated response from the model.
  """

  max_retry = 5
  retry = 0
  

  while True:
    try:
      if conversation:
        # Handle conversation history (if provided)
        # This part needs to be adapted based on the specific format 
        # required by the Vertex AI predict API for conversation history
        # 
        # Example (assuming conversation is a list of dicts with "role" and "content"):
        # instances = [{"conversation": conversation}] 

        raise NotImplementedError("Handling conversation history is not yet implemented.")

      else:
        instances = [{"content": chat_prompt}]

      response = client.predict(
          model_name=model_name,
          instances=instances,
          parameters={
              "temperature": temp,
              "max_output_tokens": max_tokens,
              "system_prompt": system  # Include system prompt in parameters
          }
      )

      text = response.predictions[0]["content"].strip()
      if remove_nl:
        text = re.sub('\s+', ' ', text)

      filename = '%s_vertexai.txt' % time()
      if not os.path.exists('.logs/vertexai_logs'):
        os.makedirs('.logs/vertexai_logs')
      with open('.logs/vertexai_logs/%s' % filename, 'w', encoding='utf-8') as outfile:
        outfile.write(f"System prompt: ===\n{system}\n===\n" + f"Chat prompt: ===\n{chat_prompt}\n===\n" + f'RESPONSE:\n====\n{text}\n===\n')

      return text

    except Exception as oops:
      retry += 1
      if retry >= max_retry:
        raise Exception("Vertex AI error: %s" % oops)
      print('Error communicating with Vertex AI API:', oops)
      time.sleep(1)
