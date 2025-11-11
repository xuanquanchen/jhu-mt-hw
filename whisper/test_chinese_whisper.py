#!/usr/bin/env python3
"""
Chinese speech recognition test script
Test Whisper's Chinese speech recognition capabilities
"""

import whisper
import torch
import os
import sys

def test_chinese_whisper():
    """Test Whisper Chinese speech recognition"""
    
    print("Whisper Chinese Speech Recognition Test")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading Whisper model...")
    try:
        # Use tiny model, faster speed, suitable for testing
        model = whisper.load_model("tiny", device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Check for test audio files
    test_files = []
    
    # Check audio files in current directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            test_files.append(file)
    
    # Check tests directory
    tests_dir = 'tests'
    if os.path.exists(tests_dir):
        for file in os.listdir(tests_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                test_files.append(os.path.join(tests_dir, file))
    
    if not test_files:
        print("\nNo audio files found!")
        print("Please place audio files in current directory or tests directory")
        print("Supported formats: .wav, .mp3, .flac, .m4a, .ogg")
        
        # Provide some suggestions
        print("\nSuggestions:")
        print("1. Download some Chinese voice samples")
        print("2. Or record a Chinese voice with your phone")
        print("3. Rename file to test_chinese.wav and place in current directory")
        return
    
    print(f"\nFound {len(test_files)} audio files:")
    for i, file in enumerate(test_files, 1):
        print(f"  {i}. {file}")
    
    # Test each audio file
    for i, audio_file in enumerate(test_files, 1):
        print(f"\nTesting file {i}/{len(test_files)}: {audio_file}")
        print("-" * 40)
        
        try:
            # Transcribe audio
            print("Transcribing...")
            result = model.transcribe(
                audio_file,
                language="zh",  # Specify Chinese
                task="transcribe",  # Transcription task
                verbose=True
            )
            
            # Display results
            print(f"\nTranscription results:")
            print(f"Detected language: {result['language']}")
            print(f"Transcribed text: {result['text']}")
            
            # Display segment information
            if 'segments' in result and result['segments']:
                print(f"\nSegment information:")
                for j, segment in enumerate(result['segments'][:3], 1):  # Show only first 3 segments
                    print(f"  Segment {j}: [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
                if len(result['segments']) > 3:
                    print(f"  ... and {len(result['segments']) - 3} more segments")
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            continue
    
    print(f"\nTest completed!")

def show_usage():
    """Show usage instructions"""
    print("Whisper Chinese Speech Recognition Test Tool")
    print("=" * 50)
    print("Usage:")
    print("1. Place Chinese audio files in current directory")
    print("2. Run: python test_chinese_whisper.py")
    print("3. View transcription results")
    print("\nSupported audio formats: .wav, .mp3, .flac, .m4a, .ogg")
    print("Recommend using .wav format for best results")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
    else:
        test_chinese_whisper()
