#!/usr/bin/env bash
# Wan2.2 TI2V-5B: 5s text->video, then 3x image(last frame)->video, concat to 20s.
# Run on the RTX 5090 server.  Usage: bash make_fox_20s.sh
set -euo pipefail

WORK=${WORK:-/workspace}
OUT=${OUT:-$WORK/output_20s.mp4}
REPO=$WORK/Wan2.2
CKPT=$WORK/Wan2.2-TI2V-5B
CLIPS=$WORK/clips
SIZE='1280*704'
FRAMES=121                # 121 frames @24fps ~= 5.04s
SEED=${SEED:-42}
PROMPT="A red fox walking along a quiet beach at sunset, golden hour light, gentle waves washing the shore, warm orange sky reflecting on wet sand, cinematic, smooth natural motion, detailed fur"

mkdir -p "$WORK" "$CLIPS"
cd "$WORK"

# ---- 1) install (skipped if already done) ----
if [ ! -d "$REPO" ]; then git clone https://github.com/Wan-Video/Wan2.2.git "$REPO"; fi
if [ ! -d "$WORK/venv" ]; then
  python3 -m venv "$WORK/venv"
  source "$WORK/venv/bin/activate"
  pip install -U pip
  # RTX 5090 (Blackwell, sm_120) needs CUDA 12.8+ builds
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  # flash_attn is built from source and often fails; Wan falls back to SDPA without it
  grep -vi '^flash_attn' "$REPO/requirements.txt" > "$WORK/req_noflash.txt"
  pip install -r "$WORK/req_noflash.txt"
  pip install "huggingface_hub[cli]"
else
  source "$WORK/venv/bin/activate"
fi
python -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
if [ ! -d "$CKPT" ]; then huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$CKPT"; fi

# ---- 2) generate ----
COMMON=(--task ti2v-5B --size "$SIZE" --frame_num "$FRAMES" --ckpt_dir "$CKPT"
        --offload_model True --convert_model_dtype --t5_cpu --prompt "$PROMPT")
TIMES="$WORK/clip_times.csv"; echo "clip,seconds" > "$TIMES"

for i in 1 2 3 4; do
  f="$CLIPS/clip$i.mp4"
  start=$(date +%s)
  if [ "$i" = 1 ]; then
    (cd "$REPO" && python generate.py "${COMMON[@]}" --base_seed "$SEED" --save_file "$f")
  else
    (cd "$REPO" && python generate.py "${COMMON[@]}" --base_seed "$SEED" --image "$CLIPS/last$((i-1)).png" --save_file "$f")
  fi
  echo "clip$i,$(( $(date +%s) - start ))" | tee -a "$TIMES"
  # last frame -> input image of the next clip
  ffmpeg -y -loglevel error -sseof -1 -i "$f" -update 1 -q:v 1 "$CLIPS/last$i.png"
done

# ---- 3) concat (clips 2-4 start on the previous last frame, so drop their 1st frame) ----
norm() { ffmpeg -y -loglevel error -i "$1" -vf "$2fps=24,scale=1280:704,format=yuv420p" -an -c:v libx264 -crf 14 -preset slow "$3"; }
norm "$CLIPS/clip1.mp4" ""                       "$CLIPS/n1.mp4"
for i in 2 3 4; do norm "$CLIPS/clip$i.mp4" "select='gte(n\,1)',setpts=N/(24*TB),"  "$CLIPS/n$i.mp4"; done
: > "$CLIPS/list.txt"; for i in 1 2 3 4; do echo "file 'n$i.mp4'" >> "$CLIPS/list.txt"; done
ffmpeg -y -loglevel error -f concat -safe 0 -i "$CLIPS/list.txt" -c copy "$OUT"
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=nw=1 "$OUT"
echo "== per-clip generation time =="; cat "$TIMES"
