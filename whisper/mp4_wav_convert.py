#!/usr/bin/env python3
"""
Manual video to WAV converter
Use this if you have already downloaded a video file
"""
# Zihan Lyu
# MP4 to WAV conversion
import os
import sys
from wavConvertor import convert_existing_video_to_wav

def main():
    if len(sys.argv) != 2:
        print("Usage: python mp4_wav_convert.py <video_file_path>")
        print("Example: python mp4_wav_convert.py downloaded_video.mp4")
        return
    
    video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        print(f"Error: File '{video_file}' not found!")
        return
    
    # Convert to WAV
    wav_file = convert_existing_video_to_wav(video_file, "output.wav")
    
    if wav_file:
        print(f"Success! WAV file created: {wav_file}")
    else:
        print("Conversion failed!")

if __name__ == "__main__":
    main()
