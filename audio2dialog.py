### ushko/transcriber_elevenlabs.py
"""
ElevenLabs API Transcription Module

This module provides functions for transcribing audio files using the ElevenLabs API.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, NamedTuple
from dataclasses import dataclass
import time
from dotenv import load_dotenv
import argparse # Added import
import mutagen # Added main mutagen module import
from mutagen.mp3 import MP3 # Added import for audio duration
from mutagen import MutagenError # Added import for error handling

# Load environment variables
load_dotenv()
DEFAULT_API_KEY = os.getenv("ELEVENLABS_API_KEY")
API_PRICE_PER_HOUR = 0.48

class TranscriptionStats(NamedTuple):
    """Statistics about the transcription process."""
    api_duration: float # Renamed duration to api_duration for clarity
    file_size: int
    model: str
    audio_duration: Optional[float] # Added audio duration
    estimated_cost: Optional[float] # Added estimated cost

@dataclass
class TranscriptionResults:
    """Results of audio transcription."""
    dialog_file: str
    # Removed merged_file and channel_files

def transcribe_elevenlabs(
    entity_id: str,
    raw_audio_file: str,
    api_key: str = DEFAULT_API_KEY,
    language_code: str = "ru",
    num_speakers: int = 2,
    diarize: bool = True,
    channel_names: Optional[Dict[str, str]] = None, 
    metadata: Optional[Dict[str, Dict]] = None,
    prompt: str = "",
    timestamps_granularity: str = "word",
    output_folder: str = "transcriptions"  # Added parameter
) -> Tuple[Optional[Dict[str, Any]], Optional[TranscriptionStats]]:
    """
    Transcribe audio file using ElevenLabs API. Returns raw JSON response.
    
    Args:
        entity_id: Unique identifier for the entity
        raw_audio_file: Path to the audio file
        api_key: ElevenLabs API key
        language_code: Language code for transcription
        num_speakers: Number of speakers to detect
        diarize: Whether to perform speaker diarization
        channel_names: Dict mapping speaker IDs to names
        metadata: Additional metadata for speakers
        prompt: Transcription prompt
        timestamps_granularity: The granularity of the timestamps in the transcription. Allowed: 'none', 'word', 'character'.
        
    Returns:
        Tuple containing raw JSON transcription data (dict) and TranscriptionStats, or (None, None) on failure.
    """
    if not api_key:
        print("Error: No ElevenLabs API key provided")
        return None, None

    mp3_path = Path(raw_audio_file)
    if not mp3_path.exists():
        print(f"Error: Audio file not found: {raw_audio_file}")
        return None, None

    # Get audio duration before API call
    audio_duration = None
    try:
        # Attempt to load with MP3 specific, fallback to generic Mutagen
        try:
            audio_info = MP3(raw_audio_file)
        except MutagenError:
            audio_info = mutagen.File(raw_audio_file) 
            
        if audio_info and audio_info.info:
             audio_duration = audio_info.info.length
        else:
            print(f"Warning: Could not read audio duration for {raw_audio_file}")
            
    except MutagenError as e:
        print(f"Warning: Could not read audio metadata for {raw_audio_file}: {e}")
    except Exception as e: # Catch other potential file reading errors
        print(f"Warning: Error reading audio file {raw_audio_file} for duration: {e}")

    # Calculate estimated cost
    estimated_cost = None
    if audio_duration is not None:
        estimated_cost = (audio_duration / 3600) * API_PRICE_PER_HOUR

    start_time = time.time()
    
    # Create output directory structure
    output_dir = Path(output_folder) / str(entity_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = mp3_path.with_suffix('.json')
    
    # Perform transcription using ElevenLabs API
    curl_command = [
        'curl',
        '-X', 'POST',
        'https://api.elevenlabs.io/v1/speech-to-text',
        '-H', f'xi-api-key: {api_key}',
        '-H', 'Content-Type: multipart/form-data',
        '-F', 'model_id=scribe_v1',
        '-F', f'language_code={language_code}',
        '-F', f'num_speakers={num_speakers}',
        '-F', f'diarize={str(diarize).lower()}',
        '-F', f'timestamps_granularity={timestamps_granularity}',  # Added granularity param
    ]
    
    # Add prompt if provided
    if prompt:
        curl_command.extend(['-F', f'prompt={prompt}'])
        
    # Add audio file
    curl_command.extend(['-F', f'file=@{str(mp3_path)}'])
    
    try:
        print(f"Transcribing {mp3_path.name} with ElevenLabs API...")
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True) # Added check=True
        
        # Parse the JSON response
        transcription_json = json.loads(result.stdout)
        
        # Calculate statistics
        api_call_duration = time.time() - start_time
        file_size = mp3_path.stat().st_size
        
        stats = TranscriptionStats(
            api_duration=api_call_duration,
            file_size=file_size,
            model="elevenlabs_scribe_v1",
            audio_duration=audio_duration, # Include audio duration
            estimated_cost=estimated_cost # Include estimated cost
        )
        
        print(f"API call completed in {api_call_duration:.2f} seconds")
        return transcription_json, stats
        
    except subprocess.CalledProcessError as e:
        print(f"Error transcribing {mp3_path.name}: {e.stderr}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing ElevenLabs API response: {e}")
        print(f"Raw response: {result.stdout[:500]}...") # Log part of the response
        return None, None
    except Exception as e:
        print(f"Exception during transcription API call: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.ms string."""
    if seconds is None:
        return "??:??:??.???"
    milliseconds = int((seconds - int(seconds)) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def process_elevenlabs_transcription(
    transcription_json: Dict[str, Any],
    entity_id: str,
    channel_names: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Dict]] = None,
    output_folder: str = "transcriptions"
) -> Optional[TranscriptionResults]:
    """
    Processes the raw JSON transcription data from ElevenLabs API into text files.
    
    Args:
        transcription_json: Raw JSON data from the API.
        entity_id: Unique identifier for the entity.
        channel_names: Dict mapping speaker IDs to names.
        metadata: Additional metadata for speakers (currently unused in processing).
        
    Returns:
        TranscriptionResults object or None on failure.
    """
    try:
        # Ensure entity_id is a string to avoid PosixPath / ObjectId issue
        str_entity_id = str(entity_id)
        
        # Create output directory structure
        output_dir = Path(output_folder) / str_entity_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file prefix from entity_id (if it's a path, use stem)
        try:
            # Убедимся, что используем строку для создания объекта Path
            file_prefix = Path(str(entity_id)).stem
        except Exception:
            file_prefix = str(entity_id)

        # Save the raw transcription result
        raw_json_path = output_dir / f"{file_prefix}_raw_transcription.json"
        with open(raw_json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_json, f, ensure_ascii=False, indent=2)
            
        # Process the transcription into required formats
        dialog_path = output_dir / f"{file_prefix}_dialog.txt"
        
        # Process words into structured format
        transcript_by_speaker = {}
        for word in transcription_json.get('words', []):
            # Handle potential missing speaker_id gracefully
            speaker_id = word.get('speaker_id') 
            if speaker_id is None:
                speaker_id = 'unknown' # Assign a default if missing
            
            if speaker_id not in transcript_by_speaker:
                transcript_by_speaker[speaker_id] = []
            transcript_by_speaker[speaker_id].append({
                'text': word['text'],
                'start': word['start'],
                'end': word['end']
            })
        
        # Create dialog format transcript
        with open(dialog_path, 'w', encoding='utf-8') as f:
            previous_speaker = None
            utterance = []
            utterance_start_time = None
            utterance_end_time = None
            
            # Sort words by start time to ensure correct order
            all_words = []
            for speaker, words in transcript_by_speaker.items():
                for word in words:
                    all_words.append((word['start'], speaker, word))
            
            all_words.sort(key=lambda x: x[0])
            
            for start, speaker, word in all_words:
                # Use speaker ID directly if channel_names is None or ID not found
                display_name = speaker 
                if channel_names:
                    display_name = channel_names.get(str(speaker).replace('speaker_', ''), speaker) 
                
                if speaker != previous_speaker and utterance:
                    # End current utterance and write it
                    if previous_speaker is not None: # Check if previous_speaker is set
                        prev_display_name = previous_speaker
                        if channel_names:
                             prev_display_name = channel_names.get(str(previous_speaker).replace('speaker_', ''), previous_speaker)
                        # Format timestamp
                        timestamp_str = f"[{format_timestamp(utterance_start_time)} --> {format_timestamp(utterance_end_time)}]"
                        f.write(f"{timestamp_str} {prev_display_name}: {''.join(utterance)}\n")
                    # Reset for new utterance
                    utterance = []
                    utterance_start_time = word['start'] # Start time of the new utterance
                    
                # Append word and update end time
                utterance.append(word['text'])
                utterance_end_time = word['end'] # Keep track of the latest end time
                if utterance_start_time is None: # Set start time for the very first utterance
                    utterance_start_time = word['start']
                previous_speaker = speaker
                
            # Write the last utterance
            if utterance and previous_speaker is not None:
                final_display_name = previous_speaker
                if channel_names:
                    final_display_name = channel_names.get(str(previous_speaker).replace('speaker_', ''), previous_speaker)
                # Format timestamp for the last utterance
                timestamp_str = f"[{format_timestamp(utterance_start_time)} --> {format_timestamp(utterance_end_time)}]"
                f.write(f"{timestamp_str} {final_display_name}: {''.join(utterance)}\n")
        
        results = TranscriptionResults(
            dialog_file=str(dialog_path)
        )
        
        print(f"Processed transcription saved to {output_dir}")
        return results

    except Exception as e:
        print(f"Exception during transcription processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Command-line interface for ElevenLabs transcription."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file using ElevenLabs API.")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe.")
    parser.add_argument("--entity-id", default="cli_transcription", help="Entity ID for organizing output files (default: cli_transcription).")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="ElevenLabs API key (defaults to ELEVENLABS_API_KEY environment variable).")
    parser.add_argument("--lang", default="ru", help="Language code for transcription (default: ru).")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers to detect (default: 2).")
    parser.add_argument("--no-diarize", action="store_false", dest="diarize", help="Disable speaker diarization.")
    parser.add_argument("--prompt", default="", help="Transcription prompt.")
    parser.add_argument("--timestamps-granularity", choices=["none", "word", "character"], default="word",
                        help="Granularity of timestamps in the transcription (default: word).")
    parser.add_argument("--output-folder", default="transcriptions", help="Folder to save output files (default: transcriptions).")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key not provided. Set ELEVENLABS_API_KEY environment variable or use --api-key.")
        return

    print(f"Starting transcription for {args.audio_file}...")
    
    # Call the API
    transcription_json, stats = transcribe_elevenlabs(
        entity_id=args.entity_id,
        raw_audio_file=args.audio_file,
        api_key=args.api_key,
        language_code=args.lang,
        num_speakers=args.speakers,
        diarize=args.diarize,
        prompt=args.prompt,
        timestamps_granularity=args.timestamps_granularity,
        output_folder=args.output_folder  # Pass to function
        # channel_names and metadata are not needed for the API call itself
    )

    if transcription_json and stats:
        print("API call successful. Processing results...")
        
        # Example channel names and metadata (can be customized or passed as arguments if needed)
        # Generate channel names based on the actual speakers found OR the requested number
        # For simplicity, using requested number first. A more robust approach might inspect transcription_json.
        channel_names = {str(i): f"Speaker {i+1}" for i in range(args.speakers)} 
        # If diarization is off, or only 1 speaker requested, adjust default names
        if not args.diarize or args.speakers <= 1:
             channel_names = {"0": "Speaker"} # Or derive from transcription if speaker_id exists

        metadata = {str(i): {"role": f"Speaker {i+1}"} for i in range(args.speakers)} # Metadata remains optional for processing

        # Process the results
        results = process_elevenlabs_transcription(
            transcription_json=transcription_json,
            entity_id=args.entity_id,
            channel_names=channel_names,
            metadata=metadata,
            output_folder=args.output_folder  # Pass to function
        )

        if results:
            print("\nTranscription processing successful!")
            # print(f"  Merged transcript: {results.merged_file}") # Removed
            print(f"  Dialog transcript: {results.dialog_file}")
            # for channel, file_path in results.channel_files.items(): # Removed
            #     print(f"  Channel {channel} transcript: {file_path}") # Removed
            print(f"\nStats:")
            print(f"  Audio Duration: {stats.audio_duration:.2f} seconds" if stats.audio_duration is not None else "  Audio Duration: Unknown") # Print audio duration
            print(f"  API Call Duration: {stats.api_duration:.2f} seconds") # Updated label
            print(f"  File Size: {stats.file_size} bytes")
            print(f"  Model: {stats.model}")
            print(f"  Estimated Cost: ${stats.estimated_cost:.4f}" if stats.estimated_cost is not None else "  Estimated Cost: Unknown") # Print cost
        else:
            print("\nTranscription processing failed.")
            
    else:
        print("\nTranscription API call failed.")

if __name__ == "__main__":
    main()
