import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import os

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

import textgrad as tg
from textgrad.optimizer import TextualGradientDescent

# ==================== Configurations ====================
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
# MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

EVAL_SYS_TEMPLATE = """Analyze the strengths and weaknesses of each generated video step by step, and explain why the video is not good or why it is good.

**Current Prompt**:
{prompt}

**Reasoning Task**:
{task}

**Note**:
- The videos were stitched together vertically to form a single video for comparison purposes;
- Your output should only include the analysis.
- There may be instances where both videos are subpar, necessitating strict adherence to the task definition.
"""

REASONING_TYPES = {
    "Graph Traversal": "Progressive traversal starting from a blue start node, gradually turning the entire graph blue by visiting all nodes, while following implicit traversal rules and avoiding forbidden edges.",
    "Maze Solving": "Navigate through a maze from an entry point to an exit, respecting maze boundaries and avoiding walls or obstacles.",
    "Sorting Numbers": "Process sequences containing numeric information (e.g., shapes or objects with numbers) by arranging or transforming them according to ascending or descending order rules.",
    "Temporal Ordering": "Determine the correct chronological sequence of events or actions based on temporal cues and dependencies.",
    "Rule Extrapolation": "Infer and apply hidden rules from observed patterns or sequences to predict subsequent steps or outcomes.",
    "Game Move Reasoning": "Analyze a game state and plan valid moves or actions according to the game's rules to achieve a goal.",
    "Shape Fitting": "Identify and fit geometric shapes into a spatial configuration, completing or extending partial shapes.",
    "Connecting Colors": "Detect color patterns and connect points or regions based on color similarity or continuity.",
    "Pattern Recognition": "Recognize recurring visual patterns or sequences and predict their continuation or completion.",
    "Odd-One-Out": "Identify regions that differ between two similar images shown side-by-side by locating distinct areas in each image.",
    "Counting Objects": "Accurately count the number of specific objects or features within a scene.",
    "Visual Analogy": "Solve analogy problems by understanding relationships between visual elements and applying similar transformations.",
    "Simple Sudoku Completion": "Complete a Sudoku puzzle by filling in missing numbers while adhering to Sudoku rules.",
    "Arithmetic Operations": "Perform basic arithmetic calculations and demonstrate the stepwise reasoning process visually.",
    "Symbolic Reasoning": "Manipulate and reason about abstract symbols or representations to solve logic puzzles.",
    "Visual Deduction": "Draw logical conclusions from visual evidence or clues presented in the video frames.",
    "Transitive Reasoning": "Apply transitive logic to infer relationships between elements based on given premises.",
    "Game Rule Reasoning": "Understand and apply complex game rules to predict outcomes or plan strategies.",
    "Tool Use": "Plan and execute the use of tools in a sequence of actions to achieve a specific physical goal, or identify the correct and appropriate tool for a given task without necessarily performing the action.",
    "Robot Navigation": "Guide a robot or agent through an environment to reach a target location while avoiding obstacles.",
    "Goal-Directed Planning": "Devise a multi-step plan to accomplish a defined goal, maintaining temporal coherence.",
    "Multi-Step Manipulation": "Perform a series of manipulations or transformations on objects to achieve a desired state.",
    "Visual Instruction Following": "Interpret and follow complex visual instructions to complete a task accurately.",
    "Game Strategy Planning": "Formulate and execute a strategy in a game scenario involving multiple sequential decisions."
}



# ==================== Utility Functions ====================

def setup_output_dir(output_dir, max_iterations):
    """Create output directory structure"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(max_iterations):
        Path(f"{output_dir}/step_{i}").mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")


def load_models():
    """Load Wan2.1 models"""
    print("Loading Wan2.1 models...")
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID, 
        subfolder="image_encoder", 
        torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        vae=vae, 
        image_encoder=image_encoder, 
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    print("✓ Models loaded successfully")
    return pipe


def prepare_image(image_path: str, pipe) -> Tuple[Image.Image, int, int]:
    """Prepare and adjust image size"""
    image = load_image(image_path)
    
    max_area = 720 * 1280
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    
    image = image.resize((width, height))
    print(f"✓ Image prepared: {width}x{height}")
    return image, width, height


def generate_video(
    pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    output_path: str,
    num_frames: int = 81,
    guidance_scale: float = 5.0
) -> List[Image.Image]:
    """
    Generate a single video
    Return video frames list
    """
    print(f"  Generating video with seed={seed}...")
    
    # Set random seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=generator
    ).frames[0]
    
    # Save video
    export_to_video(output, output_path, fps=16)
    print(f"  ✓ Video saved: {output_path}")
    
    return output


def stitch_frames_vertically(frames: List[Image.Image]) -> Image.Image:
    """Stitch the first and last frames of two videos vertically"""
    widths, heights = zip(*(f.size for f in frames))
    total_height = sum(heights)
    max_width = max(widths)
    
    stitched_img = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for f in frames:
        stitched_img.paste(f, (0, y_offset))
        y_offset += f.height
    
    return stitched_img


def extract_video_frames(video_path: str) -> List[Image.Image]:
    """Extract all frames from a video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)
    cap.release()
    return frames


def stitch_videos_vertically(
    video_path1: str,
    video_path2: str,
    output_path: str,
    fps: int = 16
) -> str:
    """
    stitch two videos vertically to form a single video
    """
    print(f"  Stitching videos vertically...")
    
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    # check if the videos are successfully opened
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Failed to open one or both video files")
    
    # get the video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    
    print(f"    Video 1: {width1}x{height1} @ {fps1} fps")
    print(f"    Video 2: {width2}x{height2} @ {fps2} fps")
    
    # output video size
    out_width = max(width1, width2)
    out_height = height1 + height2
    
    print(f"    Output: {out_width}x{out_height} @ {fps} fps")
    
    # use MJPEG encoding instead of mp4v (more stable)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    if not out.isOpened():
        raise ValueError(f"Failed to create VideoWriter for {output_path}")
    
    frame_count = 0
    
    # stitch frames frame by frame
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # ensure the dimensions of the two frames are consistent
        if frame1.shape[0] != height1 or frame1.shape[1] != width1:
            frame1 = cv2.resize(frame1, (width1, height1))
        if frame2.shape[0] != height2 or frame2.shape[1] != width2:
            frame2 = cv2.resize(frame2, (width2, height2))
        
        # adjust the frame width to the output width
        if frame1.shape[1] != out_width:
            frame1 = cv2.resize(frame1, (out_width, height1))
        if frame2.shape[1] != out_width:
            frame2 = cv2.resize(frame2, (out_width, height2))
        
        # vertically stitch the frames
        stitched_frame = np.vstack([frame1, frame2])
        
        # verify the dimensions of the stitched frames
        if stitched_frame.shape != (out_height, out_width, 3):
            print(f"    Warning: Frame shape mismatch at frame {frame_count}")
            stitched_frame = cv2.resize(stitched_frame, (out_width, out_height))
        
        success = out.write(stitched_frame)
        if not success:
            print(f"    Warning: Failed to write frame {frame_count}")
        
        frame_count += 1
    
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"  ✓ Stitched video saved: {output_path} ({frame_count} frames)")
    return output_path


def pil_image_to_bytes(img: Image.Image, format="JPEG", quality=50) -> bytes:
    """Convert PIL image to bytes"""
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def optimize_prompt(
    current_prompt: str,
    task_definition: str,
    stitched_video_path: str,
    llm_engine: str,
    max_iters: int = 1
) -> str:
    """
    Use TextGrad to optimize the prompt
    """
    print(f"  Optimizing prompt with TextGrad...")
    
    # Wrap prompt as an optimizable variable
    prompt_var = tg.Variable(
        current_prompt,
        requires_grad=True,
        role_description="a text prompt for video generation",
    )
    
    # Create optimizer
    optimizer = tg.TGD(
        engine=llm_engine,
        parameters=[prompt_var],
        constraints=["Only generate a valid prompt text within 300 tokens."],
    )

    eval_instruction = EVAL_SYS_TEMPLATE.format(prompt=current_prompt, task=task_definition)

    print(f"Evaluation instruction: {eval_instruction}")

    # Create multimodal evaluation loss
    loss_fn = tg.VideoPromptEvalLoss(
        evaluation_instruction=eval_instruction,
        engine=llm_engine
    )
    
    print(f"    Original prompt: {prompt_var.value[:100]}...")
    
    for i in range(max_iters):
        optimizer.zero_grad()
        
        # Extract first and last frames of the video
        video_frames = extract_video_frames(stitched_video_path)
        if len(video_frames) < 2:
            raise ValueError("Video frames less than 2")
        
        selected_frames = [video_frames[0], video_frames[-1]]
        stitched_frame = stitch_frames_vertically(selected_frames)
        stitched_frame = stitched_frame.resize((256, 256))
        
        # Convert to bytes
        stitched_frame_bytes = pil_image_to_bytes(stitched_frame, format="JPEG", quality=50)
        video_frame_vars = tg.Variable(
            stitched_frame_bytes,
            requires_grad=False,
            role_description="stitched first and last video frames"
        )
        
        # Calculate loss
        loss = loss_fn(prompt_var, video_frame_vars)
        loss.backward()
        optimizer.step()
    
    optimized_prompt = prompt_var.value
    print(f"    ✓ Optimized prompt: {optimized_prompt[:100]}...")
    
    return optimized_prompt


# ==================== Main Process ====================

def main(args):
    """Main program: video generation and optimization with multiple iterations"""
    
    # Get parameters from args
    OUTPUT_DIR = args.output_dir
    TEST_IMAGE_PATH = args.image_path
    MAX_ITERATIONS = args.max_iterations
    init_prompt = args.init_prompt
    task = args.task
    seed1 = args.seed1
    seed2 = args.seed2
    
    # Validate task
    if task not in REASONING_TYPES:
        raise ValueError(f"Invalid task '{task}'. Must be one of: {list(REASONING_TYPES.keys())}")
    
    task_definition = REASONING_TYPES[task]
    
    # Initialize
    setup_output_dir(OUTPUT_DIR, MAX_ITERATIONS)
    pipe = load_models()
    image, width, height = prepare_image(TEST_IMAGE_PATH, pipe)
    
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )

    # Initialize LLM engine
    llm_engine = "gpt-4o"
    tg.set_backward_engine(llm_engine)
    
    current_prompt = init_prompt
    
    # ==================== Multiple Iterations ====================
    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"STEP {iteration} - Video Generation and Optimization")
        print(f"{'='*60}")
        
        step_dir = f"{OUTPUT_DIR}/step_{iteration}"
        
        # Step 1: Generate two videos (different seeds)
        print(f"\n[Step {iteration}] Generating two videos with different seeds...")
        
        video1_path = f"{step_dir}/video_seed_{seed1}.mp4"
        video2_path = f"{step_dir}/video_seed_{seed2}.mp4"
        
        frames1 = generate_video(
            pipe, image, current_prompt, negative_prompt,
            width, height, seed1, video1_path
        )
        
        frames2 = generate_video(
            pipe, image, current_prompt, negative_prompt,
            width, height, seed2, video2_path
        )
        
        # Step 2: Stitch two videos vertically
        print(f"\n[Step {iteration}] Stitching videos vertically...")
        stitched_video_path = f"{step_dir}/stitched_videos.mp4"
        stitch_videos_vertically(video1_path, video2_path, stitched_video_path)
        
        # Step 3: Optimize prompt (if not the last iteration)
        if iteration < MAX_ITERATIONS - 1:
            print(f"\n[Step {iteration}] Optimizing prompt...")
            current_prompt = optimize_prompt(
                current_prompt=current_prompt,
                task_definition=task_definition,
                stitched_video_path=stitched_video_path,
                llm_engine=llm_engine,
                max_iters=1
            )
            
            # Save optimized prompt
            prompt_log_path = f"{step_dir}/optimized_prompt.txt"
            with open(prompt_log_path, 'w', encoding='utf-8') as f:
                f.write(current_prompt)
            print(f"  ✓ Prompt saved: {prompt_log_path}")
        else:
            print(f"\n[Step {iteration}] Final iteration - skipping optimization")
        
        print(f"\n✓ Step {iteration} completed!")
    
    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print("VIDEOTPO PIPELINE COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nDirectory structure:")
    for i in range(MAX_ITERATIONS):
        print(f"  step_{i}/")
        print(f"    ├── video_seed_*.mp4")
        print(f"    ├── stitched_videos.mp4")
        if i < MAX_ITERATIONS - 1:
            print(f"    └── optimized_prompt.txt")
    
    return OUTPUT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoTPO: Video Test-time Preference Optimization")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./videotpo_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None, # Required
        help="Path to the input image"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Number of iterations (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--init_prompt",
        type=str,
        default=None, # Required
        help="Initial prompt for video generation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Counting Objects",
        choices=list(REASONING_TYPES.keys()),
        help="Reasoning task type"
    )
    parser.add_argument(
        "--seed1",
        type=int,
        default=42,
        help="Random seed for first video"
    )
    parser.add_argument(
        "--seed2",
        type=int,
        default=123,
        help="Random seed for second video"
    )
    
    args = parser.parse_args()
    
    output_dir = main(args)
    print(f"\n✓ All results saved to: {output_dir}")