#!/usr/bin/env bash
# 입력 데이터 준비. 원자료(aoa 체결·지갑)는 공개자가 올린 원본에서 직접 받고 해시를 확인한다(저장소에 재배포하지 않음).
# 사용: bash aoa_research/fetch_inputs.sh   (경로는 환경변수 AOA_RAW, MKT_DIR, BITSTAMP_REPO로 바꿀 수 있다)
set -euo pipefail
RAW=${AOA_RAW:-/home/user/aoa_raw/data}
MKT=${MKT_DIR:-/home/user/mktdata}
BITSTAMP_REPO=${BITSTAMP_REPO:-/home/user/alessiostomeo/btc-cycle-data}
REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
mkdir -p "$RAW" "$MKT/spot" "$MKT/daily"

# 1) aoa 공개 압축(Google Drive, DCInside 차트 마이너 갤러리 글 5051684에 링크)
ID=1XDwxbriz_kOq44iH-mHcjsYTBklMnMW3
ZIP="$(dirname "$RAW")/aoa_public_2021-12-31_with_letter.zip"
if [ ! -f "$ZIP" ]; then
  UA='Mozilla/5.0 (X11; Linux x86_64)'
  UUID=$(curl -sSL -A "$UA" "https://drive.google.com/uc?export=download&id=$ID" | grep -o 'name="uuid" value="[^"]*"' | sed 's/.*value="//;s/"//')
  curl -sSL -A "$UA" -e https://drive.google.com/ -o "$ZIP" \
    "https://drive.usercontent.google.com/download?id=$ID&export=download&authuser=0&confirm=t&uuid=$UUID"
fi
echo "b6f1dc7aadf8209bf6c99fd516a06c0cabdc77c5f16f9fc8df92fdf1d8d01b9a  $ZIP" | sha256sum -c -
python3 -I - "$ZIP" "$RAW" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for i in z.infolist():
    name = "letter_90days.txt" if i.filename.endswith(".txt") else i.filename
    assert "/" not in name and ".." not in name
    with z.open(i) as s, open(sys.argv[2] + "/" + name, "wb") as d:
        while True:
            b = s.read(1 << 20)
            if not b:
                break
            d.write(b)
PY
(cd "$RAW" && sha256sum -c - <<'SUMS'
8f2aa1af3caaeaeffff0375ce4d10b21e88ab22c624976d9031ddd0e715653b1  aoa-execution-2018-03-01-2018-12-31.csv
ebbd90935d8e609e6371fec69babbfea13c0458a94748b05f3f795cecd814b5a  aoa-execution-2019-01-01-2020-12-31.csv
a51f070c9e00aed72720b9029567d4adf32656d06594b87dff3b56360564ca37  aoa-execution-2021-01-01-2021-06-30.csv
3bd436edb68c60bd17eb3b0c307c68baf56ad79c1da9a21a0433031f4ea7a071  aoa-execution-2021-07-01-2021-12-31.csv
db0a8e6180faa093a7843ec09b1f673417252b2a7b6cac50ac96db426f2f3452  aoa-wallet-2018-03-01-2021-12-31.csv
SUMS
)

# 2) Bitstamp BTC/USD 1분봉
[ -d "$BITSTAMP_REPO/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/alessiostomeo/btc-cycle-data "$BITSTAMP_REPO"

# 3) 바이낸스 현물 1시간봉과 주식 일봉(같은 저장소의 claude/daily-market-data-download 브랜치), 삼성전자는 이 폴더의 data/
git -C "$REPO_ROOT" fetch -q origin claude/daily-market-data-download
for s in BTCUSDT ETHUSDT; do
  git -C "$REPO_ROOT" show "origin/claude/daily-market-data-download:data_spot/${s}_1h.csv" > "$MKT/spot/${s}_1h.csv"
done
for t in QQQ SPY 069500.KS; do
  git -C "$REPO_ROOT" show "origin/claude/daily-market-data-download:data_daily/yahoo/$t.csv" > "$MKT/daily/$t.csv"
done
cp "$REPO_ROOT/aoa_research/data/005930.KS.csv" "$MKT/daily/"
echo "준비 완료: AOA_RAW=$RAW MKT_DIR=$MKT BITSTAMP_REPO=$BITSTAMP_REPO"
