from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
import json
print("creating data loader...")
data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
# test_loader = DataLoader(testing_data, batch_size=2, shuffle=True, num_workers=8)
iterator = iter(data)
namelist = []
for i in range(len(data)):
    source, model_kwargs = next(iterator)
    namelist.append(model_kwargs['y']['key_id'])

import codecs as cs
from os.path import join as pjoin

textlist = []
for name in namelist:
    with cs.open(pjoin('/mnt/disk_1/jinpeng/AMASS/HumanML3D/HumanML3D/texts/', name + '.txt')) as f:
        textlist.append({name: f.readlines()[0]})
import time
import os
import openai
import numpy as np

openai.api_type = "azure"
openai.api_base = "https://gcrgpt4aoai9c.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "65ad84f76d2347d3a4517a100d5ff629"
system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 4096
token_limit = 4096
conversation = [system_message]
in_context = open("/mnt/disk_1/jinpeng/motion-diffusion-model/T2M/prompt.txt").readlines()

retries = 10000
b = 0
c = b
while retries > 0:
    try:
        for i in range(b, len(textlist)):
            user_input = str(in_context)[2:-2] + list(textlist[i].values())[0].split('#')[0]
            conversation.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
                messages=conversation,
                temperature=0.7,
                max_tokens=max_response_tokens,
            )
            del conversation[1]
            retries = 10000
            mess = json.dumps(response['choices'][0]['message']['content'])
            f2 = open('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_0905/' + list(textlist[i].keys())[0] + '.json', 'w')
            f2.write(mess)
            f2.close()
            print("number" + str(i) + ': ' + list(textlist[i].keys())[0] + 'json')
            c = i
    except Exception as e:
        if e:
            b = c
            del conversation[1]
            print('try: number ' + str(b))
            print(e)
            print('Timeout error, retrying...')
            retries -= 1
            time.sleep(1)
        else:
            del conversation[1]
            raise e
