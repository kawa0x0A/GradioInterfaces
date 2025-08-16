# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import gc
import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import random

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(
    0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'Wan2.2'))
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video

# Global Variables
prompt_expander = None
wan_i2v_a14b = None

#python generate.py --task i2v-a14B --size 1280*704 --ckpt_dir ./Wan2.2-I2V-a14B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

# Button Functions
def load_model():
    global wan_i2v_a14b
    try:
        print("Loading I2V-A14B model...", end='', flush=True)
        cfg = WAN_CONFIGS['i2v_A14B']
        wan_i2v_a14b = wan.WanI2V(
            config=cfg,
            checkpoint_dir='Wan2.2-I2V-A14B',
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=False,
        )
        print("done", flush=True)
        return "Model Loaded"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error: {str(e)}"


def ti2v_generation(prompt, image, size, frame_num, sample_steps, 
                   guide_scale, shift_scale, seed, n_prompt, sample_solver):
    global wan_i2v_a14b
    
    if wan_i2v_a14b is None:
        return None, "Error: Model not loaded. Please load the model first."

    if image is None:
        return None, "Error: Please upload an image for I2V generation."

    if prompt.strip() == "":
        return None, "Error: Please provide a prompt."

    try:
        print(f"Generating video with prompt: {prompt}")
        print(f"Parameters: size={size}, frame_num={frame_num}, steps={sample_steps}, guide_scale={guide_scale}, shift={shift_scale}, seed={seed}")
        
        # Generate video
        video = wan_i2v_a14b.generate(
            prompt,
            img=image,
            size=SIZE_CONFIGS[size],
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=shift_scale,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed if seed >= 0 else random.randint(0, 63),
            offload_model=True
        )

        # Save video
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        save_file = f"i2v_a14b_{size.replace('*','x')}_{formatted_prompt}_{formatted_time}.mp4"
        save_file = "result.mp4"
        
        cfg = WAN_CONFIGS['i2v_A14B']
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        return save_file, "Video generated successfully!"

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(error_msg)
        return None, error_msg


# Gradio Interface
def gradio_interface():
    with gr.Blocks(title="Wan2.2 I2V-A14B") as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.2 (I2V-A14B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Text and Image to Video Generation with Wan I2V-A14B Model
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                # Input image
                input_image = gr.Image(
                    type="pil",
                    label="Upload Input Image",
                    elem_id="image_upload",
                    height=600
                )

                # Prompt
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate from the image",
                    lines=3
                )

                # Generation parameters
                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        size = gr.Dropdown(
                            label="Video Size",
                            choices=list(SUPPORTED_SIZES['i2v-A14B']),
                            value="704*1280"
                        )
                        frame_num = gr.Slider(
                            label="Frame Number",
                            minimum=4,
                            maximum=120,
                            value=72,
                            step=4,
                            info="24 FPS is recommended for smooth video generation. Adjust based on your needs."
                        )
                    
                    with gr.Row():
                        sample_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1
                        )
                        guide_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.5
                        )
                    
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift Scale",
                            minimum=0.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1
                        )
                        seed = gr.Slider(
                            label="Seed (-1 for random)",
                            minimum=-1,
                            maximum=63,
                            step=1,
                            value=-1
                        )
                    
                    sample_solver = gr.Dropdown(
                        label="Sample Solver",
                        choices=["unipc", "dpm++"],
                        value="unipc"
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe what you don't want in the video",
                        lines=2
                    )

                # Generate button
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column():
                # Output
                output_video = gr.Video(
                    label='Generated Video', 
                    interactive=False, 
                    height=600
                )
                
                generation_status = gr.Textbox(
                    label="Generation Status",
                    interactive=False
                )

        generate_btn.click(
            fn=ti2v_generation,
            inputs=[
                prompt_input, input_image, size, frame_num, sample_steps,
                guide_scale, shift_scale, seed, negative_prompt, sample_solver
            ],
            outputs=[output_video, generation_status]
        )

    return demo


if __name__ == '__main__':
    load_model()  # Load model at startup

    # Launch Gradio interface
    demo = gradio_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=True
    )