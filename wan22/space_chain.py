"""Chain four 5s Wan2.2-TI2V-5B clips through a Hugging Face Space and concat to 20s.

Clip 1 is text-to-video; clips 2-4 use the previous clip's last frame as the input
image. Per-clip wall time is written to clip_times.csv in the output directory.

Usage: python space_chain.py [start_clip]   (resumes from start_clip if earlier clips exist)
Env: SPACE, OUT_DIR, OUT, WIDTH, HEIGHT, STEPS, SEED, HF_TOKEN (optional, more ZeroGPU quota)
"""
import csv
import os
import shutil
import subprocess
import sys
import time

from gradio_client import Client, handle_file

SPACE = os.environ.get("SPACE", "https://upsampler-wan-2-2-5b-video.hf.space")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/clips")
OUT = os.environ.get("OUT", "/workspace/output_20s.mp4")
# the Space caps width/height at 1024: generate 16:9 at 1024x576, upscale+crop to 1280x704
W, H = int(os.environ.get("WIDTH", 1024)), int(os.environ.get("HEIGHT", 576))
STEPS = int(os.environ.get("STEPS", 4))
SEED = int(os.environ.get("SEED", 42))
PROMPT = (
    "A red fox walking along a quiet beach at sunset, golden hour light, gentle waves "
    "washing the shore, warm orange sky reflecting on wet sand, cinematic, smooth natural "
    "motion, detailed fur"
)


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def last_frame(video, png):
    ff("-sseof", "-1", "-i", video, "-update", "1", "-q:v", "1", png)


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    times_path = os.path.join(os.path.dirname(OUT), "clip_times.csv")
    # A token lifts the ZeroGPU quota; gradio sends it to the Space as x-hf-authorization.
    # SPACE may be a direct https://<sub>.hf.space URL so no huggingface.co call is made.
    tok = os.environ.get("HF_TOKEN")
    client = Client(SPACE, headers={"x-hf-authorization": f"Bearer {tok}"} if tok else None)

    for i in range(start, 5):
        clip = os.path.join(OUT_DIR, f"clip{i}.mp4")
        image = None if i == 1 else handle_file(os.path.join(OUT_DIR, f"last{i-1}.png"))
        t0 = time.time()
        video, seed = client.predict(
            image, PROMPT, H, W,
            duration_seconds=5.0, guidance_scale=0.0, steps=STEPS,
            seed=SEED, randomize_seed=False, api_name="/generate_video",
        )
        dt = time.time() - t0
        src = video["video"] if isinstance(video, dict) else video
        shutil.copy(src, clip)
        last_frame(clip, os.path.join(OUT_DIR, f"last{i}.png"))
        new = not os.path.exists(times_path) or i == 1
        with open(times_path, "w" if i == 1 else "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["clip", "mode", "seconds"])
            w.writerow([f"clip{i}", "t2v" if i == 1 else "i2v", f"{dt:.1f}"])
        print(f"clip{i} done in {dt:.1f}s -> {clip}", flush=True)

    # clips 2-4 begin on the previous clip's last frame: drop that duplicate frame
    lst = os.path.join(OUT_DIR, "list.txt")
    with open(lst, "w") as f:
        for i in range(1, 5):
            n = os.path.join(OUT_DIR, f"n{i}.mp4")
            pre = "" if i == 1 else "select='gte(n\\,1)',setpts=N/(24*TB),"
            ff("-i", os.path.join(OUT_DIR, f"clip{i}.mp4"),
               "-vf", f"{pre}fps=24,scale=1280:704:force_original_aspect_ratio=increase:flags=lanczos,crop=1280:704,format=yuv420p",
               "-an", "-c:v", "libx264", "-crf", "16", "-preset", "slow", n)
            f.write(f"file '{n}'\n")
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
