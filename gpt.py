from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader

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
in_context = "Given a user prompt, envision a motion scene and create eight distinct brief pose descriptions at a " \
             "frame rate of 1 fps. Ensure that each description is self-contained. The difference between two " \
             "adjacent descriptions must be small, considering the small interval. Use the following format: [F1: "", " \
             "F2: "", ..., F8: ""]. Before you write each description, you must follow these instructions. These are " \
             "of primary importance: 1. Posture and Position: Observe and describe the overall position and " \
             "orientation of the body. This includes the three-dimensional spatial location of the body, whether it " \
             "is upright, sitting, or standing. 2. Body Curves: Observe the curves and contours of the body. Pay " \
             "attention to the curves and postures of the head, neck, back, waist, hips, shoulders, arms, " \
             "and legs. 3. Limbs Angles: Pay attention to the angles of each joint. Observe the angles of the " \
             "shoulders, elbows, wrists, hips, knees, ankles, and other areas, and describe whether they are bent, " \
             "extended, or flexed. 4. Center of Gravity: Observe the position of the body's center of gravity. Take " \
             "note of the balance of the head, torso, and legs, as well as whether the body's center of gravity is " \
             "leaning forward, backward, or to one side. 5. Gestures and Postures: Pay attention to the movements and " \
             "positions of the hands. The posture of the hands, the degree of finger flexion, and the orientation of " \
             "the palms can provide information about the posture. 6. Don't include words in the scenes such as " \
             "chairs, table or hamburger. Don't use the words like mouth or stomach, just include the body parts that " \
             "could reflect human posture. Do not use the relative state of the frames before and after to describe " \
             "the pose. Do not use verb. Some sample descriptions are as follows: The person is striding forward with " \
             "the right leg in front of the left. The right heel is on the ground with the toes pointing up. The left " \
             "knee is bent. The upper body is hunched forward slightly. Both arms are bent, with the left arm " \
             "reaching in front of the upper body." "The person is in a crouching pose and is touching the ground. " \
             "The left hand is backwards, spread apart from the right hand. The right hand is beside the right foot, " \
             "below the right hip, then the left elbow is bent at right angle, the left upper arm and the right thigh " \
             "are parallel to the floor and the right arm is in front of the left arm, both knees are almost " \
             "completely bent. The person is kneeling on their left leg and is bent forward." "The figure is doing " \
             "backwards movements and is in a inclined pose. The right knee is forming a L shape and the left foot is " \
             "stretched forwards, the right elbow is barely bent, then the left shoulder is further down than the " \
             "right. The subject is inclined backward and to the left of the pelvis. The left hand is further down " \
             "than the left hip and behind the right hand and wide apart from the right hand, the right leg is behind " \
             "the other. The right upper arm is parallel to the ground." "The right knee is unbent with the right leg " \
             "next to the left while both hands are apart wider than shoulder width. The right upper arm, " \
             "the left leg, the torso and the right thigh are straightened up while the right elbow is bent a bit. " \
             "Prompt: "

retries = 3
b = 1087
c = b
while retries > 0:
    try:
        for i in range(b, len(textlist)):
            user_input = in_context + list(textlist[i].values())[0].split('#')[0]
            conversation.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
                messages=conversation,
                temperature=0.7,
                max_tokens=max_response_tokens,
            )
            del conversation[1]
            retries = 3
            np.save(
                '/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_0822/' + list(textlist[i].keys())[0] + '.npy',
                response['choices'][0]['message']['content'])
            print("number" + str(i) + ': ' + list(textlist[i].keys())[0] + 'npy')
            c = i
    except Exception as e:
        if e:
            b = c
            print('try: number ' + str(b))
            print(e)
            print('Timeout error, retrying...')
            retries -= 1
            time.sleep(30)
        else:
            raise e
