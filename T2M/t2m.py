import argparse
import torch
from utils.parser_util import generate_args
from utils.fixseed import fixseed
import os
from utils import dist_util
from data_loaders.p2m.dataset import HumanML3D
from text2pose.generative.evaluate_generative import load_model
from T2M.text_to_pose.tools import compute_text2poses_similarity, search_optimal_path
from text2pose.vocab import Vocabulary  # needed
from data_loaders.p2m.tools import axis_angle_to
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from torch.utils.data import DataLoader
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.p2m.tools import inverse
from body_models.smplh import SMPLH
import numpy as np


def main():
    args = generate_args()
    fixseed(args.seed)

    dist_util.setup_dist(args.device)

    # Text Data
    # A person walks like a crab.
    # Fs = [
    #     "The person is in a crouched position, resembling a crab walk. Both hands and feet are on the ground, "
    #     "and the back is slightly arched.",
    #     "The left hand and foot are moved forward simultaneously, while the right hand and foot remain in place. The "
    #     "body leans to the left side.",
    #     "The right hand and foot are brought forward, aligning with the left hand and foot. The body is level, "
    #     "and both knees are bent.",
    #     "The person shuffles sideways to the left. The left hand and foot move together, followed by the right hand "
    #     "and foot. The body maintains a crouched position.",
    #     "Continuing the lateral movement, the person shifts to the right. The right hand and foot move first, "
    #     "then the left hand and foot. The body remains crouched.",
    #     "The left hand and foot are advanced once more to the left, creating a smooth crab walk motion. The right "
    #     "hand and foot follow suit.",
    #     "Shifting to the right, the person moves their right hand and foot forward. The left hand and foot mirror the "
    #     "movement, maintaining the crouched posture.",
    #     "The crab walk continues as the person repeats the lateral movement. Both hands and feet work in sync, "
    #     "allowing for smooth sideways motion. "
    # ]
    # Fs = ["The person is standing upright with a relaxed posture. Both arms are hanging down by their sides.",
    #       "The person is starting to tilt forward at the waist. The upper body is leaning slightly, and the head is lowering.",
    #       "The person continues to bend over. The upper body is now at a 45-degree angle to the ground. Both arms are dangling freely.",
    #       "The person's back is parallel to the ground, creating a straight line from head to hips. Both arms are hanging down towards the ground.",
    #       "The person's upper body is almost parallel to the ground. Their hands are now touching their knees as they continue to bend forward.",
    #       "The person's upper body is perpendicular to the ground. Their hands are now reaching towards their ankles, and the head is facing downwards.",
    #       "The person's upper body has lowered further, and the hands are now touching the ground in front of their feet. The spine is fully curved in a forward bend.",
    #       "The person has reached their maximum bend. The head is near or between the knees. The arms are relaxed and hanging down towards the ground, and the back forms a deep arch."
    #       ]
    # Fs = ["The person is sitting upright on a stool. The torso is leaning slightly forward. Both arms are bent at the elbows, with the right hand holding a hamburger close to the mouth.",
    #       "The person's body remains in the same position. The left hand is holding the hamburger with a firm grip, while the right hand is reaching for a napkin on the table.",
    #       "The posture is unchanged. The person takes a bite from the hamburger, causing a slight tilt of the head downward. The jaws are slightly open, and the fingers of the left hand are gripping the bun.",
    #       "Still seated, the person has lowered the hamburger slightly, fingers adjusting their grip. The jaws are closing, and the expression seems focused on chewing. The left forearm is parallel to the ground.",
    #       "The body is still. The hamburger has been consumed halfway. The left hand now holds the remaining portion, while the right hand picks up a glass of drink. The person's gaze is directed downward.",
    #       "Unaltered in posture, the person takes a sip from the glass using the right hand. The left hand is holding the hamburger at chest level. The head is slightly tilted back as they swallow.",
    #       "The person lifts the hamburger again, taking another bite. The left hand is steady, maintaining its grip. The right arm is down by the side, holding the glass. The head is now tilted to the right slightly.",
    #       "The person is almost finished with the hamburger. Both hands are closer to the table now. The left hand is holding the last bite, and the right hand has put down the glass. The torso is leaning back slightly, indicating satisfaction."
    #       ]
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

    # loading model
    pose_model, _ = load_model(args.pose_model, dist_util.dev())

    # sampling pose  # batch sampling
    with torch.no_grad():
        for f in Fs:
            pose = pose_model.sample_str_nposes(f, n=args.n_generate)['pose_body'][0].view(args.n_generate, -1)
            text_dist = pose_model.text_encoder(pose_model.tokenizer(f).to(dist_util.dev()).view(1, -1),
                                                torch.tensor([len(pose_model.tokenizer(f).to(dist_util.dev()))]))
            pose_dist = pose_model.pose_encoder(pose)
            poses.append(pose)
            pose_dists.append(pose_dist)
            text_dists.append(text_dist)

    # [(n_generate x 1), (n_generate x 1), ...]
    text2poses_similarity = compute_text2poses_similarity(text_dists, pose_dists)

    path = search_optimal_path(pose_dists, text2poses_similarity, dist_util.dev(), args.n_generate, args.op)

    cond = []
    for i, p in enumerate(path):
        pose_i = poses[i][p, :].reshape(-1, 3)[1:22][None]
        cond.append(pose_i)
    cond = torch.concatenate(cond, axis=0)
    cond = axis_angle_to("rot6d", cond)
    print('==============Stage 1 Finished=============')
    print('==============Stage 2 Begin=============')
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Creating model and diffusion...")
    data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
    test_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model, diffusion = create_model_and_diffusion(args, test_loader)
    # body mody
    smplh = SMPLH(
        path='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/',
        input_pose_rep='rot6d',
        batch_size=1,
        gender='neutral').to(dist_util.dev()).eval()

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    all_motions = {}
    for i in range(args.num_repetitions):
        repeat_time = 'repeat_' + str(i)
        all_motions[repeat_time] = []
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        sample_fn = diffusion.p_sample_loop
        model_kwargs = {
            'y': {'pose_feature': cond.reshape(-1, 126)[None].type(torch.float32), 'lengths': 64 * torch.ones(1).type(torch.IntTensor),
                  'mask': torch.ones(64, dtype=bool), 'key_id': f}}
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                             model_kwargs['y'].items()}
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, 64),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        trans = sample[:, 0:3].permute(0, 3, 2, 1)
        trans = inverse(trans)
        pose = sample[:, 3:].permute(0, 3, 2, 1).reshape(1, 64, -1, 6)
        vertices = smplh(1, pose,
                         trans).cpu().numpy()
        all_motions['repeat_'+str(rep_i)].append(np.split(sample.cpu().numpy(), sample.shape[0]))
    for i in range(args.num_repetitions):
        all_motions['repeat_' + str(i)] = np.concatenate(all_motions['repeat_' + str(i)], axis=0)

    # npy_path = os.path.join(out_path, 'results.npy')
    # print(f"saving results file to [{npy_path}]")
    # results = {'sample': all_motions}
    # np.save(npy_path, results)
    np.save('/mnt/disk_1/jinpeng/motion-diffusion-model/compare/waltz2.npy', vertices)


if __name__ == "__main__":
    main()
