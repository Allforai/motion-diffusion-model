import os
import math
import argparse
import torch
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.distributions import kl_divergence, Normal
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.vocab import Vocabulary  # needed
from text2pose.generative.evaluate_generative import load_model
import text2pose.utils_visu as utils_visu

parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str,
                    default='/mnt/disk_1/jinpeng/motion-diffusion-model/text_to_pose/experiments/eccv22_posescript_models/CondTextPoser_textencoder-glovebigru_vocA1H1_latentD32/train-posescript-H1/wloss_kld0.2_v2v4.0_rot2.0_jts2.0_kldnpmul0.02_kldntmul0.0/B32_Adam_lr1e-05_wd0.0001_pretrained_gen_glovebigru_vocA1H1_dataA1/seed0/checkpoint_1999.pth',
                    help='Path to the model.')
parser.add_argument('--n_generate', type=int, default=32, help="Number of poses to generate.")
parser.add_argument('--op', type=str, default='all', help="Number of poses to generate.")
parser.add_argument('--nb_cols', type=int, default=8, help="Number of poses to generate.")
args = parser.parse_args()

model_path = args.model_path
n_generate = args.n_generate
nb_cols = args.nb_cols

nb_rows = n_generate // nb_cols
margin_img = 320

device = 'cpu'

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def setup(model_path):
    model, _ = load_model(model_path, device)

    body_model = BodyModel(bm_fname=config.SMPLH_NEUTRAL_BM, num_betas=config.n_betas)
    body_model.eval()
    body_model.to(device)

    return model, body_model


model, body_model = setup(model_path)

# A person walks like a crab.
Fs = [
    "The person is in a crouched position, resembling a crab walk. Both hands and feet are on the ground, and the back is slightly arched.",
    "The left hand and foot are moved forward simultaneously, while the right hand and foot remain in place. The body leans to the left side.",
    "The right hand and foot are brought forward, aligning with the left hand and foot. The body is level, and both knees are bent.",
    "The person shuffles sideways to the left. The left hand and foot move together, followed by the right hand and foot. The body maintains a crouched position.",
    "Continuing the lateral movement, the person shifts to the right. The right hand and foot move first, then the left hand and foot. The body remains crouched.",
    "The left hand and foot are advanced once more to the left, creating a smooth crab walk motion. The right hand and foot follow suit.",
    "Shifting to the right, the person moves their right hand and foot forward. The left hand and foot mirror the movement, maintaining the crouched posture.",
    "The crab walk continues as the person repeats the lateral movement. Both hands and feet work in sync, allowing for smooth sideways motion."
]

# A person is dancing waltz.
Fs = [
    "The person is in a waltz starting position. Their right foot is pointed forward, and the left foot is placed slightly back. The left hand is holding the right hand, extended forward and slightly to the side. The elbows are slightly bent, and the person is standing tall.",
    "The dancer takes a step forward with their left foot. The right foot remains in place, and the arms are gracefully extended to the sides. The left knee is slightly bent, and the right leg is straight with the toes pointed.",
    "Continuing the waltz, the person now steps to the side with their right foot. The left foot follows, bringing the feet together. The arms are brought closer to the body, creating a flowing motion.",
    "The dancer executes a graceful turn to the right. The right foot is brought behind the left foot, and they rise onto their toes. The arms are elegantly curved, and the head turns gently to the right.",
    "In a fluid movement, the dancer transitions to a reverse turn. The left foot crosses behind the right, and they pivot smoothly. The arms stretch outward and away from each other.",
    "The person continues the reverse turn. The right foot now crosses behind the left, and they pivot further. The arms are still extended gracefully, following the movement.",
    "Continuing the waltz, the person steps to the side with their right foot, crossing it slightly in front of the left foot. The right arm extends gracefully to the side, and the left arm is brought closer to the body, forming a soft curve at the elbow.",
    "The waltz concludes with a final twirl. The person spins elegantly, raising their right arm upward and extending the left arm outward. The legs are in a poised position, and the expression exudes grace and charm."
]

pose_dists = []
text_dists = []
poses = []

with torch.no_grad():
    for f in Fs:
        pose = model.sample_str_nposes(f, n=n_generate)['pose_body'][0].view(n_generate, -1)

        pose_dist = model.pose_encoder(pose)
        text_dist = model.text_encoder(model.tokenizer(f).to(device).view(1, -1),
                                       torch.tensor([len(model.tokenizer(f).to(device))]))

        poses.append(pose)
        pose_dists.append(pose_dist)
        text_dists.append(text_dist)


def compute_text2poses_similarity(tds, pds):
    kl_divs = []
    for td, pd in zip(tds, pds):  # (1x32), (n_generate, 32)
        kl = kl_divergence(pd, td).sum(-1, keepdims=True)  # (n_generate x 32) -> (n_generate x 1)
        kl_divs.append(kl)
    return kl_divs


# [(n_generate x 1), (n_generate x 1), ...]
text2poses_similarity = compute_text2poses_similarity(text_dists, pose_dists)


def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


random_string = generate_random_string(5)
print(f"Images saved in vis/{random_string}")
os.makedirs(f'vis/{random_string}', exist_ok=True)

for pose_i, pose in enumerate(poses):
    imgs = utils_visu.image_from_pose_data(pose, body_model)
    imgs = [img[margin_img:-margin_img, margin_img:-margin_img] for img in imgs]
    image_width, image_height = imgs[0].shape[1], imgs[0].shape[0]

    merged_width = nb_cols * image_width
    merged_height = nb_rows * image_height
    merged_image = Image.new('RGB', (merged_width, merged_height))

    font = ImageFont.truetype("/usr/share/fonts/truetype/Sarai/Sarai.ttf", size=int(1 / 5 * image_width))
    draw = ImageDraw.Draw(merged_image)

    max_indexes = torch.argsort(text2poses_similarity[pose_i])[:5].tolist()  # rank top-5

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

        draw.text((col * image_width + 2 / 3 * image_width, row * image_height + 1 / 7 * image_height), f"{kl}",
                  fill=color, font=font)

    merged_image.save(f'vis/{random_string}/merged_image_frame{pose_i + 1}.jpg')


def viterbi_algorithm(energies):
    num_nodes = len(energies)
    num_states = len(energies[0])

    # Initialize the Viterbi matrix and the backpointer matrix
    viterbi = [[0.0] * num_states for _ in range(num_nodes)]
    backpointers = [[0] * num_states for _ in range(num_nodes)]

    # Initialize the first column of the Viterbi matrix with initial probabilities
    for state in range(num_states):
        viterbi[0][state] = energies[0][state][0]

    # Forward pass: fill in the Viterbi matrix and backpointer matrix
    for node in range(1, num_nodes):
        for state in range(num_states):
            min_prob = float('inf')
            min_prev_state = -1
            for prev_state in range(num_states):
                prob = viterbi[node - 1][prev_state] + energies[node][state][prev_state]
                if prob < min_prob:
                    min_prob = prob
                    min_prev_state = prev_state
            viterbi[node][state] = min_prob
            backpointers[node][state] = min_prev_state

    # Backward pass: find the optimal path
    path = [0] * num_nodes
    min_final_prob = min(viterbi[-1])
    last_state = viterbi[-1].index(min_final_prob)

    path[-1] = last_state
    for node in range(num_nodes - 2, -1, -1):
        path[node] = backpointers[node + 1][path[node + 1]]

    return path


def search_optimal_path(pose_dists, text_kl_divs, op='all'):
    kl_divs = [torch.zeros(n_generate, n_generate)]  # make len(kl_divs) == len(text_kl_divs)
    for i in range(len(pose_dists) - 1):
        pose_dist1 = pose_dists[i + 1]  # (n_generate x 32)
        pose_dist2 = pose_dists[i]  # (n_generate x 32)

        # (n_generate**2 x 32)
        pose_dist1_proxy = Normal(
            pose_dist1.loc.unsqueeze(1).repeat(1, n_generate, 1).view(-1, pose_dist1.loc.shape[-1]),
            pose_dist1.scale.unsqueeze(1).repeat(1, n_generate, 1).view(-1, pose_dist1.scale.shape[-1]))
        # (n_generate**2 x 32)
        pose_dist2_proxy = Normal(pose_dist2.loc.repeat(n_generate, 1),
                                  pose_dist2.scale.repeat(n_generate, 1))

        # (n_generate**2 x 32) -> (n_generate**2) -> (n_generate x n_generate)
        kl_div = kl_divergence(pose_dist1_proxy, pose_dist2_proxy).sum(-1).view(n_generate, n_generate)
        kl_divs.append(kl_div)

    pose_kl_divs = torch.stack(kl_divs)  # (8 x n_generate x n_generate)
    text_kl_divs = torch.stack(text_kl_divs).expand(-1, -1,
                                                    n_generate) * 100.0  # (8 x n_generate x 1) -> (8 x n_generate x n_generate)

    if op == 'all':
        energies = pose_kl_divs + text_kl_divs
    elif op == 'text':
        energies = text_kl_divs
    elif op == 'pose':
        energies = pose_kl_divs
    else:
        raise ValueError(f'Unknown {op}')

    path = viterbi_algorithm(energies.tolist())
    return path


path = search_optimal_path(pose_dists, text2poses_similarity, args.op)

imgs_selected = []
raw_paths = ''
poses_selected = []

for i, p in enumerate(path):
    imgs = utils_visu.image_from_pose_data(poses[i][p, :][None], body_model)
    imgs_selected.append(imgs[0])
    poses_selected.append({'text': Fs[i], 'pose': poses[i][p, :].cpu().numpy()})
    raw_paths += ' ' + str(p + 1)

import pickle

with open(f'vis/{random_string}/poses_selected.pkl', 'wb') as f:
    pickle.dump(poses_selected, f)

imgs_selected = [img[margin_img:-margin_img, margin_img:-margin_img] for img in imgs_selected]

image_width, image_height = imgs_selected[0].shape[1], imgs_selected[0].shape[0]
merged_width = nb_cols * image_width
merged_height = math.floor(len(Fs) / nb_cols) * image_height
merged_image = Image.new('RGB', (merged_width, merged_height))

draw = ImageDraw.Draw(merged_image)

for i in range(len(imgs_selected)):
    row = i // nb_cols
    col = i % nb_cols

    x = col * image_width
    y = row * image_height

    image = Image.fromarray(imgs_selected[i])

    merged_image.paste(image, (x, y))

merged_image.save(f'vis/{random_string}/selected_image_frame_{raw_paths}.jpg')
