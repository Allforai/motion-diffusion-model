import torch
import os
import numpy as np
from text2pose.generative.evaluate_generative import load_model
from text2pose.generative.model_generative import CondTextPoser
from text2pose.vocab import Vocabulary  # needed
from T2M.text_to_pose.tools import compute_text2poses_similarity, search_optimal_path, generate_random_string
from data_loaders.p2m.tools import axis_angle_to
from utils.parser_util import generate_args
from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.p2m.tools import inverse
from body_models.smplh import SMPLH
import random
import string
from T2M.text_to_pose import utils_visu
from PIL import Image, ImageDraw, ImageFont
from human_body_prior.body_model.body_model import BodyModel
import math

# GPT
print("==========GPT is working==========")
import openai

openai.api_type = "azure"
openai.api_base = "https://gcrgpt4aoai9c.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "65ad84f76d2347d3a4517a100d5ff629"
system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 4096
token_limit = 4096
conversation = [system_message]
in_context = open("/mnt/disk_1/jinpeng/motion-diffusion-model/T2M/prompt.txt").readlines()
prompt = input('Please enter your order\n')
user_input = str(in_context)[2:-2] + prompt
conversation.append({"role": "user", "content": user_input})
response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    messages=conversation,
    temperature=0.7,
    max_tokens=max_response_tokens,
)
Fs = response['choices'][0]['message']['content']
print("==========GPT response==========")
print(Fs)


print("==========Sampling Pose==========")
# pose
pose_model_path = '/mnt/disk_1/jinpeng/motion-diffusion-model/text2pose/experiments/eccv22_posescript_models/CondTextPoser_textencoder-glovebigru_vocA1H1_latentD32/train-posescript-H1/wloss_kld0.2_v2v4.0_rot2.0_jts2.0_kldnpmul0.02_kldntmul0.0/B32_Adam_lr1e-05_wd0.0001_pretrained_gen_glovebigru_vocA1H1_dataA1/seed0/checkpoint_1999.pth'
n_generate = 32

pose_model, _ = load_model(pose_model_path, 'cuda')
body_model = BodyModel(bm_fname='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/SMPLH_NEUTRAL.npz', num_betas=16)
body_model.eval()
body_model.to('cuda')
pose_dists = []
text_dists = []
poses = []
with torch.no_grad():
    for f in Fs.split(':')[1:]:
        pose = pose_model.sample_str_nposes(f, n=n_generate)['pose_body'][0].view(n_generate, -1)
        text_dist = pose_model.text_encoder(pose_model.tokenizer(f).to('cuda').view(1, -1),
                                            torch.tensor([len(pose_model.tokenizer(f).to('cuda'))]))
        pose_dist = pose_model.pose_encoder(pose)
        poses.append(pose)
        pose_dists.append(pose_dist)
        text_dists.append(text_dist)
text2poses_similarity = compute_text2poses_similarity(text_dists, pose_dists)
path = search_optimal_path(pose_dists, text2poses_similarity, 'cuda', n_generate, 'all')
print(path)

random_string = generate_random_string(5)

out_path = os.path.join('/mnt/disk_1/jinpeng/motion-diffusion-model/0907_unseen', prompt.split(' ')[-1], random_string)
print(f"Images saved in {out_path}")
os.makedirs(f'{out_path}', exist_ok=True)
print("==========Generating Pose Image==========")
margin_img = 320
nb_rows = 4
nb_cols = 8
for pose_i, pose in enumerate(poses):
    imgs = utils_visu.image_from_pose_data(pose, body_model)
    imgs = [img[margin_img:-margin_img, margin_img:-margin_img] for img in imgs]
    image_width, image_height = imgs[0].shape[1], imgs[0].shape[0]

    merged_width = nb_cols * image_width
    merged_height = nb_rows * image_height
    merged_image = Image.new('RGB', (merged_width, merged_height))

    font = ImageFont.truetype("/usr/share/fonts/truetype/Sarai/Sarai.ttf", size=int(1/5 * image_width))
    draw = ImageDraw.Draw(merged_image)

    max_indexes = torch.argsort(text2poses_similarity[pose_i])[:5].tolist()  # rank top-5

    #
    for i in range(len(imgs)):
        row = i // nb_cols
        col = i % nb_cols

        kl = text2poses_similarity[pose_i][i].item()

        x = col * image_width
        y = row * image_height

        image = Image.fromarray(imgs[i])

        merged_image.paste(image, (x, y))

        color = (0, 0, 255)
        if i in max_indexes:
            color = (255, 0, 0)

        draw.text((col * image_width + 2/3 * image_width, row * image_height + 1/7 * image_height), f"{kl}", fill=color, font=font)

    merged_image.save(f'{out_path}/merged_image_frame{pose_i+1}.jpg')


draw = ImageDraw.Draw(merged_image)

print("==========Generating Condition==========")
imgs_selected = []
raw_paths = ''
cond = []
for i, p in enumerate(path):
    imgs = utils_visu.image_from_pose_data(poses[i][p, :][None], body_model)
    imgs_selected.append(imgs[0])
    raw_paths += ' ' + str(p + 1)
    pose_i = poses[i][p, :].reshape(-1, 3)[1:22][None]
    cond.append(pose_i)
cond = torch.concatenate(cond, axis=0)
assert cond.shape[0] == 8
cond = axis_angle_to("rot6d", cond)
print("==========Generating Selected Poses==========")
imgs_selected = [img[margin_img:-margin_img, margin_img:-margin_img] for img in imgs_selected]

image_width, image_height = imgs_selected[0].shape[1], imgs_selected[0].shape[0]
merged_width = nb_cols * image_width
merged_height = math.floor(len(poses) / nb_cols) * image_height
merged_image = Image.new('RGB', (merged_width, merged_height))

for i in range(len(imgs_selected)):
    row = i // nb_cols  #
    col = i % nb_cols
    x = col * image_width
    y = row * image_height
    image = Image.fromarray(imgs_selected[i])
    merged_image.paste(image, (x, y))
merged_image.save(f'{out_path}/selected_image_frame_{raw_paths}.jpg')


print("==========Generating SMPLH Meshes==========")
args = generate_args()
data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
total_num_samples = args.num_samples * args.num_repetitions
model, diffusion = create_model_and_diffusion(args, train_loader)
print(f"Loading checkpoints from [{args.model_path}]...")
state_dict = torch.load(args.model_path, map_location='cpu')
load_model_wo_clip(model, state_dict)
model.to('cuda')
model.eval()  # disable random masking

cond = cond.unsqueeze(0).to('cuda')
model_kwargs = {'y': {'pose_feature': cond, 'lengths': 64 * torch.ones(1).type(torch.IntTensor),
                      'mask': torch.ones(64, dtype=bool)}}
# add CFG scale to batch
model_kwargs['y']['scale'] = torch.ones(cond.shape[0]).to('cuda') * 2.5
model_kwargs['y'] = {key: val.to('cuda') if torch.is_tensor(val) else val for key, val in
                     model_kwargs['y'].items()}

smplh = SMPLH(
    path='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/',
    input_pose_rep='rot6d',
    batch_size=1,
    gender='neutral').to('cuda').eval()

all_motions = []
for rep_i in range(3):
    print(f'### Sampling [repetitions #{rep_i}]')
    sample_fn = diffusion.p_sample_loop
    final = sample_fn(
        model,
        (cond.shape[0], model.njoints, model.nfeats, 64),  # BUG FIX
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sample = final
    trans = sample[:, 0:3].permute(0, 3, 2, 1)
    trans = inverse(trans)
    pose = sample[:, 3:].permute(0, 3, 2, 1).reshape(1, 64, -1, 6)
    vertices = smplh(1, pose,
                     trans).cpu().numpy()
    all_motions.append(vertices)
all_motions = np.concatenate(all_motions, axis=0)

npy_path = os.path.join(out_path, prompt.split(' ')[-1] + '.npy')
prompt_path = os.path.join(out_path, prompt.split(' ')[-1] + '_prompt.npy')
print(f"saving results file to [{npy_path}]")
np.save(npy_path, all_motions)
np.save(prompt_path, prompt)
