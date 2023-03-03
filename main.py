# from fastapi import FastAPI, Form

# app = FastAPI()

from fastapi import FastAPI, Form, File, UploadFile
import uvicorn
import replicate
import os
import openai
import requests
import json
# import redis

# r = redis.from_url(
#   "redis://default:A38GqqOQuZEk7bgPzgbsN4Il3DBUl68j@redis-12925.c252.ap-southeast-1-1.ec2.cloud.redislabs.com:12925"
# )
# from walrus import Database

# db = Database.from_url(
#   "redis://default:A38GqqOQuZEk7bgPzgbsN4Il3DBUl68j@redis-12925.c252.ap-southeast-1-1.ec2.cloud.redislabs.com:12925"
# )

import redis

r = redis.Redis(
  host='redis-12925.c252.ap-southeast-1-1.ec2.cloud.redislabs.com',
  port=12925,
  username="default",
  password="A38GqqOQuZEk7bgPzgbsN4Il3DBUl68j")

openai.api_key = "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"

app = FastAPI()
from io import BytesIO


def process_audio(inputs: dict):

  business = r.json().get(f"business:shaoye2")
  whisper_corrections = business["whisper_corrections"]
  whisper_substitute = ""
  for i in business["whisper_substitute"]:
    whisper_substitute += f"{i['from']}={i['to']} "
  whisper_input_sample = business["whisper_input_sample"]
  whisper_output_sample = business["whisper_output_sample"]
  prompt = business["prompt"]
  lang = 'zh'

  model = replicate.models.get("junyaoc/whisper-hallu")
  version = model.versions.get(
    "7dd741561d3ec0583f608f7796843e680919d915fd42c0c3370eb1939edef59e")

  # inputs = {
  #   'voice': BytesIO(response.content),
  #   'prompt': prompt,
  #   'lang': lang,
  # }

  print("started prediction")
  output = version.predict(**inputs)
  print('output', output)

  if (output):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=
      "Due to local dialects and mispronouciation, some words had been wrongly transcribed. Here's a list of commonly mispelt phrase and the correct value:\n\n"
      + whisper_corrections +
      "\n\nPlease make equivalent corrections, when the word is actually numbers, replace with number. e.g. 二十rewrite into 20.Substitute:"
      + whisper_substitute + "\n\nExample:\nInput: " + whisper_input_sample +
      "\nOutput: " + whisper_output_sample + "\n\nInput:\n" + output +
      "\n\nOutput(you must return a valid python JSON, which is able to be loaded using JSON.loads()):\n",
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

  # if os.path.exists(file_location):
  #   os.remove(file_location)
  # else:
  #   print("The file does not exist")

  print(response)

  # return {
  #   "lang": lang,
  #   "prompt": prompt,
  #   "raw": output,
  #   "processed": response["choices"][0]["text"]
  # }
  url = "https://aiyu-parse-text.junyaoc.repl.co"
  payload = json.dumps({
    "id": "shaoye2",
    "input": response["choices"][0]["text"]
  })
  headers = {'Content-Type': 'application/json'}

  response = requests.request("POST", url, headers=headers, data=payload)
  print(json.loads(response.text))
  return json.loads(response.text)


@app.post("/")
async def infer(data: dict):

  print(data)
  iamge_source = data['url']

  payload = {}
  headers = {
    'Authorization':
    'Bearer EAArIErwJuXABAKOjZAh19nSTaEIJNCOclvihV9ZCqnVPPHrfwhWZA5nxkDZBU1Ye9U5IHld1lm9KzfZCXISq7f1evVbEYTAEukh2P1SYWLeW3af9HkFXjNkR6DPkuQOxsCGkbZCvErc0ROZCS7IZANOOyVqH3tgor3ZCCXPBAs4o7ygfcZBahkqb5kGvkfTZAZBwilCOZASgMz5ZAURAZDZD'
  }
  response = requests.request("GET",
                              iamge_source,
                              headers=headers,
                              data=payload)

  inputs = {
    'voice': BytesIO(response.content),
    # 'prompt': prompt,
    'lang': 'zh',
  }
  return process_audio(inputs)


@app.post("/audio")
async def audio(audio: UploadFile, prompt: str = Form()):

  print('prompt', prompt)

  file_location = f"/tmp/{audio.filename}"
  with open(file_location, "wb+") as file_object:
    file_object.write(audio.file.read())

  model = replicate.models.get("junyaoc/whisper-hallu")
  version = model.versions.get(
    "7dd741561d3ec0583f608f7796843e680919d915fd42c0c3370eb1939edef59e")

  inputs = {
    'voice': open(file_location, "rb"),
    'prompt': prompt,
    'lang': 'zh',
  }
  return process_audio(inputs)


uvicorn.run(app, host="0.0.0.0", port=8080)

#https://github.com/tiangolo/fastapi
#https://tryolabs.com/blog/2019/12/10/top-10-python-libraries-of-2019/
