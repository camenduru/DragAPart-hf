from functools import partial
import os
from PIL import Image, ImageOps
import random

import cv2
from diffusers.models import AutoencoderKL
import gradio as gr
import numpy as np
from segment_anything import build_sam, SamPredictor
from tqdm import tqdm
from transformers import CLIPModel, AutoProcessor, CLIPVisionModel
import torch
from torchvision import transforms

from diffusion import create_diffusion
from model import UNet2DDragConditionModel

import spaces


TITLE = '''DragAPart: Learning a Part-Level Motion Prior for Articulated Objects'''
DESCRIPTION = """
<div>
Try <a href='https://arxiv.org/abs/24xx.xxxxx'><b>DragAPart</b></a> yourself to manipulate your favorite articulated objects in 2 seconds!
</div>
"""
INSTRUCTION = '''
2 steps to get started:
- Upload an image of an articulated object.
- Add one or more drags on the object to specify the part-level interactions.

How to add drags:
- To add a drag, first click on the starting point of the drag, then click on the ending point of the drag, on the Input Image (leftmost).
- You can add up to 10 drags, but we suggest one drag per part.
- After every click, the drags will be visualized on the Image with Drags (second from left).
- If the last drag is not completed (you specified the starting point but not the ending point), it will simply be ignored.
- Have fun dragging!

Then, you will be prompted to verify the object segmentation. Once you confirm that the segmentation is decent, the output image will be generated in seconds!
'''
PREPROCESS_INSTRUCTION = '''
Segmentation is needed if it is not already provided through an alpha channel in the input image.
You don't need to tick this box if you have chosen one of the example images.
If you have uploaded one of your own images, it is very likely that you will need to tick this box.
You should verify that the preprocessed image is object-centric (i.e., clearly contains a single object) and has white background.
'''

def center_and_square_image(pil_image_rgba, drags):
    image = pil_image_rgba
    alpha = np.array(image)[:, :, 3]  # Extract the alpha channel

    cy, cx = np.round(np.mean(np.nonzero(alpha), axis=1)).astype(int)
    side_length = max(image.width, image.height)
    padded_image = ImageOps.expand(
        image, 
        (side_length // 2, side_length // 2, side_length // 2, side_length // 2), 
        fill=(255, 255, 255, 255)
    )
    left, top = cx, cy
    new_drags = []
    for d in drags:
        x, y = d
        new_x, new_y = (x + side_length // 2 - cx) / side_length, (y + side_length // 2 - cy) / side_length
        new_drags.append((new_x, new_y))

    # Crop or pad the image as needed to make it centered around (cx, cy)
    image = padded_image.crop((left, top, left + side_length, top + side_length))
    # Resize the image to 256x256
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    return image, new_drags

def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "ckpts", "sam_vit_h_4b8939.pth")
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to("cuda"))
    return predictor

def model_init():
    model_checkpoint = os.path.join(os.path.dirname(__file__), "ckpts", "drag-a-part-final.pt")
    model = UNet2DDragConditionModel.from_pretrained_sd(
        os.path.join(os.path.dirname(__file__), "ckpts", "stable-diffusion-v1-5"),
        unet_additional_kwargs=dict(
            sample_size=32,
            flow_original_res=False,
            input_concat_dragging=False,
            attn_concat_dragging=True,
            use_drag_tokens=False,
            single_drag_token=False,
            one_sided_attn=True,
            flow_in_old_version=False,
        ),
        load=False,
    )
    model.load_state_dict(torch.load(model_checkpoint, map_location="cpu")["model"])
    model = model.to("cuda")
    return model

@spaces.GPU
def sam_segment(predictor, input_image, drags, foreground_points=None):
    image = np.asarray(input_image)
    predictor = predictor.to("cuda")
    predictor.set_image(image)

    with torch.no_grad():
        masks_bbox, _, _ = predictor.predict(
            point_coords=foreground_points if foreground_points is not None else None,
            point_labels=np.ones(len(foreground_points)) if foreground_points is not None else None,
            multimask_output=True
        )

    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    out_image, new_drags = center_and_square_image(Image.fromarray(out_image, mode="RGBA"), drags)

    return out_image, new_drags

def get_point(img, sel_pix, evt: gr.SelectData):
    sel_pix.append(evt.index)
    points = []
    img = np.array(img)
    height = img.shape[0]
    arrow_width_large = 7 * height // 256
    arrow_width_small = 3 * height // 256
    circle_size = 5 * height // 256

    with_alpha = img.shape[2] == 4
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 1:
            cv2.circle(img, tuple(point), circle_size, (0, 0, 255, 255) if with_alpha else (0, 0, 255), -1)
        else:
            cv2.circle(img, tuple(point), circle_size, (255, 0, 0, 255) if with_alpha else (255, 0, 0), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (0, 0, 0, 255) if with_alpha else (0, 0, 0), arrow_width_large)
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 0, 255) if with_alpha else (0, 0, 0), arrow_width_small)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

def clear_drag():
    return []

def preprocess_image(SAM_predictor, img, chk_group, drags):
    if img is None:
        gr.Warning("No image is specified. Please specify an image before preprocessing.")
        return None, drags

    if drags is None or len(drags) == 0:
        foreground_points = None
    else:
        foreground_points = np.array([drags[i] for i in range(0, len(drags), 2)])

    if len(drags) == 0:
        gr.Warning("No drags are specified. We recommend first specifying the drags before preprocessing.")

    new_drags = drags
    if "Preprocess with Segmentation" in chk_group:
        img_np = np.array(img)
        rgb_img = img_np[..., :3]
        img, new_drags = sam_segment(
            SAM_predictor,
            rgb_img,
            drags,
            foreground_points=foreground_points,
        )
    else:
        new_drags = [(d[0] / img.width, d[1] / img.height) for d in drags]

    img = np.array(img).astype(np.float32)
    processed_img = img[..., :3] * img[..., 3:] / 255. + 255. * (1 - img[..., 3:] / 255.)
    image_pil = Image.fromarray(processed_img.astype(np.uint8), mode="RGB")
    processed_img = image_pil.resize((256, 256), Image.LANCZOS)
    return processed_img, new_drags

@spaces.GPU
def single_image_sample(
    model,
    diffusion,
    x_cond,
    x_cond_clip,
    rel,
    cfg_scale,
    x_cond_extra,
    drags,
    hidden_cls,
    num_steps=50,
    vae=None,
):
    z = torch.randn(2, 4, 32, 32).to("cuda")
    if vae is not None:
        vae = vae.to("cuda")

    # Prepare input for classifer-free guidance
    rel = torch.cat([rel, rel], dim=0).to("cuda")
    x_cond = torch.cat([x_cond, x_cond], dim=0).to("cuda")
    x_cond_clip = torch.cat([x_cond_clip, x_cond_clip], dim=0).to("cuda")
    x_cond_extra = torch.cat([x_cond_extra, x_cond_extra], dim=0).to("cuda")
    drags = torch.cat([drags, drags], dim=0).to("cuda")
    hidden_cls = torch.cat([hidden_cls, hidden_cls], dim=0).to("cuda")

    model_kwargs = dict(
        x_cond=x_cond,
        x_cond_extra=x_cond_extra,
        cfg_scale=cfg_scale,
        hidden_cls=hidden_cls,
        drags=drags,
    )

    # Denoising
    step_delta = diffusion.num_timesteps // num_steps
    for i in tqdm(range(num_steps)):
        with torch.no_grad():
            samples = diffusion.p_sample(
                model.forward_with_cfg,
                z,
                torch.Tensor([diffusion.num_timesteps - 1 - step_delta * i]).long().to("cuda").repeat(z.shape[0]),
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["pred_xstart"]
            if i != num_steps - 1:
                z = diffusion.q_sample(
                    samples, 
                    torch.Tensor([diffusion.num_timesteps - 1 - step_delta * i]).long().to("cuda").repeat(z.shape[0])
                )

        samples, _ = samples.chunk(2, dim=0)

    with torch.no_grad():
        images = vae.decode(samples / 0.18215).sample
    return ((images + 1)[0].permute(1, 2, 0) * 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

@spaces.GPU
def generate_image(model, image_processor, vae, clip_model, clip_vit, diffusion, img_cond, seed, cfg_scale, drags_list):
    if img_cond is None:
        gr.Warning("Please preprocess the image first.")
        return None

    model = model.to("cuda")
    vae = vae.to("cuda")
    clip_model = clip_model.to("cuda")
    clip_vit = clip_vit.to("cuda")

    with torch.no_grad():
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        pixels_cond = transforms.ToTensor()(img_cond.astype(np.float32) / 127.5 - 1).unsqueeze(0).to("cuda")

        cond_pixel_preprocessed_for_clip = image_processor(
            images=Image.fromarray(img_cond), return_tensors="pt"
        ).pixel_values.to("cuda")
        with torch.no_grad():
            x_cond = vae.encode(pixels_cond).latent_dist.sample().mul_(0.18215)
            cond_clip_features = clip_model.get_image_features(cond_pixel_preprocessed_for_clip)
            cls_embedding = torch.stack(
                clip_vit(pixel_values=cond_pixel_preprocessed_for_clip, output_hidden_states=True).hidden_states,
                dim=1
            )[:, :, 0]

        # dummies
        rel = torch.zeros(1, 4).to("cuda")
        x_cond_extra = torch.zeros(1, 3, 32, 32).to("cuda")

        drags = torch.zeros(1, 10, 4).to("cuda")
        for i in range(0, len(drags_list), 2):
            if i + 1 == len(drags_list):
                gr.Warning("The ending point of the last drag is not specified. The last drag is ignored.")
                break

            idx = i // 2
            drags[0, idx, 0], drags[0, idx, 1], drags[0, idx, 2], drags[0, idx, 3] = \
                drags_list[i][0], drags_list[i][1], drags_list[i + 1][0], drags_list[i + 1][1]

            if idx == 9:
                break

        images = single_image_sample(
            model.to("cuda"),
            diffusion,
            x_cond,
            cond_clip_features,
            rel,
            cfg_scale,
            x_cond_extra,
            drags,
            cls_embedding,
            num_steps=50,
            vae=vae,
        )
        return images


sam_predictor = sam_init()
model = model_init()

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to('cuda')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
clip_vit = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
diffusion = create_diffusion(
    timestep_respacing="",
    learn_sigma=False,
)

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown("# " + DESCRIPTION)

    with gr.Row():
        gr.Markdown(INSTRUCTION)
    
    drags = gr.State(value=[])

    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_image = gr.Image(
                interactive=True,
                type='pil',
                image_mode="RGBA",
                width=256,
                show_label=True,
                label="Input Image",
            )

            example_folder = os.path.join(os.path.dirname(__file__), "./example_images")
            example_fns = [os.path.join(example_folder, example) for example in sorted(os.listdir(example_folder))]
            gr.Examples(
                examples=example_fns,
                inputs=[input_image],
                cache_examples=False,
                label='Feel free to use one of our provided examples!',
                examples_per_page=30
            )

            input_image.change(
                fn=clear_drag,
                outputs=[drags],
            )

        with gr.Column(scale=1):
            drag_image = gr.Image(
                type="numpy",
                label="Image with Drags",
                interactive=False,
                width=256,
                image_mode="RGB",
            )

            input_image.select(
                fn=get_point,
                inputs=[input_image, drags],
                outputs=[drag_image],
            )
        
        with gr.Column(scale=1):
            processed_image = gr.Image(
                type='numpy', 
                label="Processed Image", 
                interactive=False, 
                width=256,
                height=256,
                image_mode='RGB',
            )
            processed_image_highres = gr.Image(type='pil', image_mode='RGB', visible=False)

            with gr.Accordion('Advanced preprocessing options', open=True):
                with gr.Row():
                    with gr.Column():
                        preprocess_chk_group = gr.CheckboxGroup(
                            ['Preprocess with Segmentation'], 
                            label='Segment',
                            info=PREPROCESS_INSTRUCTION
                        )
            
            preprocess_button = gr.Button(
                value="Preprocess Input Image",
            )
            preprocess_button.click(
                fn=partial(preprocess_image, sam_predictor),
                inputs=[input_image, preprocess_chk_group, drags],
                outputs=[processed_image, drags],
                queue=True,
            )

        with gr.Column(scale=1):
            generated_image = gr.Image(
                type="numpy",
                label="Generated Image",
                interactive=False,
                height=256,
                width=256,
                image_mode="RGB",
            )

            with gr.Accordion('Advanced generation options', open=True):
                with gr.Row():
                    with gr.Column():
                        seed = gr.Slider(label="seed", value=0, minimum=0, maximum=10000, step=1, randomize=False)
                        cfg_scale = gr.Slider(
                            label="classifier-free guidance weight",
                            value=5, minimum=1, maximum=10, step=0.1
                        )

            generate_button = gr.Button(
                value="Generate Image",
            )
            generate_button.click(
                fn=partial(generate_image, model, image_processor, vae, clip_model, clip_vit, diffusion),
                inputs=[processed_image, seed, cfg_scale, drags],
                outputs=[generated_image],
            )

    demo.launch()
