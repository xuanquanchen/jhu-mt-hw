#!/usr/bin/env python3
"""
Test speech detection on a specific audio file
Based on the official Whisper test_transcribe.py
"""

import os
import sys
import torch
import whisper
from whisper.tokenizer import get_tokenizer

def test_audio_file(audio_path, model_name="tiny", language=None):
    """Test speech detection on an audio file"""
    
    print(f"Testing audio file: {audio_path}")
    print(f"Using model: {model_name}")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Whisper model...")
    try:
        model = whisper.load_model(model_name).to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Transcribe audio
    print("Transcribing audio...")
    try:
        result = model.transcribe(
            audio_path, 
            language=language, 
            temperature=0.0, 
            word_timestamps=True,
            verbose=True
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("TRANSCRIPTION RESULTS")
        print("=" * 50)
        print(f"Detected language: {result['language']}")
        print(f"Full text: {result['text']}")
        
        # Show segments with timestamps
        if 'segments' in result and result['segments']:
            print(f"\nSegments ({len(result['segments'])} total):")
            for i, segment in enumerate(result['segments'], 1):
                print(f"  {i:2d}. [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
        
        # Show word-level timestamps (first few words)
        if 'segments' in result and result['segments']:
            print(f"\nWord-level timestamps (first segment):")
            first_segment = result['segments'][0]
            if 'words' in first_segment and first_segment['words']:
                for word_info in first_segment['words'][:10]:  # Show first 10 words
                    print(f"  [{word_info['start']:.2f}s - {word_info['end']:.2f}s] '{word_info['word']}'")
                if len(first_segment['words']) > 10:
                    print(f"  ... and {len(first_segment['words']) - 10} more words")
        
        return result
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_audio_file.py <audio_file> [model_name] [language]")
        print("Example: python test_audio_file.py test_audio/english/english_01.wav tiny en")
        print("Available models: tiny, base, small, medium, large")
        return
    
    audio_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    language = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = test_audio_file(audio_path, model_name, language)
    
    if result:
        print("\nTranscription completed successfully!")
    else:
        print("\nTranscription failed!")

if __name__ == "__main__":
    main()


