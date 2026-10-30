import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Edge", torch_dtype=torch.bfloat16, device_map="cuda"
)
video = pipe(prompt="A mobile robot navigates a warehouse aisle and stops at a shelf.").video
export_to_video(video, "first_video.mp4", fps=24)

