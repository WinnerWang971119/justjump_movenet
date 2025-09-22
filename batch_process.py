import os
import subprocess
from pathlib import Path

INPUT_DIR = Path("input_videos")
OUTPUT_DIR = Path("output_results")

VARIANT = "lightning"   # "lightning" | "thunder"
SAVE_PREVIEW = True     # 要不要存 preview.mp4
SKIP = 0                # 0 = 每格都跑, 1 = 每兩格跑一次

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    videos = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
    if not videos:
        print("[!] 沒有找到影片，請放到 input_videos/")
        return
    
    for vid in videos:
        outdir = OUTPUT_DIR / vid.stem
        cmd = [
            "python", "movenet_video_export.py",
            "--video", str(vid),
            "--outdir", str(outdir),
            "--variant", VARIANT,
            "--skip", str(SKIP)
        ]
        if SAVE_PREVIEW:
            cmd.append("--save-preview")
        
        print(f"[+] 處理 {vid.name} → {outdir}")
        subprocess.run(cmd, check=True)
    
    print("\n✅ 全部影片處理完成！結果在 output_results/ 裡。")

if __name__ == "__main__":
    main()
