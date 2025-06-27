import os
import tempfile
import logging
from datetime import datetime
import whisper
from pydub import AudioSegment
import glob
import json
import hashlib
import time
from dotenv import load_dotenv
from subjective_abstract_data_source_package import SubjectiveDataSource

# Load environment variables
load_dotenv()

# ---------------------------- Configuration ---------------------------- #

# Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'
WHISPER_MODEL_SIZE = 'base'  # Adjust based on your system's capabilities


class SubjectiveTranscribeLocalVideoDataSource(SubjectiveDataSource):
    """
    Data source for transcribing local video files using Whisper AI.
    Converts video files to audio and generates transcriptions stored as JSON context files.
    """
    
    def __init__(self, name=None, session=None, dependency_data_sources=None, subscribers=None, params=None):
        super().__init__(
            name=name,
            session=session,
            dependency_data_sources=dependency_data_sources or [],
            subscribers=subscribers,
            params=params
        )
        
        # Initialize configuration from params or environment variables
        self.whisper_model_size = self.params.get('whisper_model_size', WHISPER_MODEL_SIZE)
        self.videos_dir = self.params.get('videos_dir', os.getenv("VIDEOS_DIR", "videos"))
        self.context_dir = self.params.get('context_dir', os.getenv("CONTEXT_DIR", "context"))
        self.specific_video_path = self.params.get('specific_video_path', None)
        
        # Initialize Whisper model (will be loaded when needed)
        self.whisper_model = None
        
        # Track processing state
        self.processed_count = 0
        self.skipped_count = 0

    def fetch(self):
        """
        Main method to fetch and process video files for transcription.
        """
        try:
            self._update_status("Starting video transcription process")
            
            # Get list of video files to process
            video_files = self._get_video_files()
            if not video_files:
                self._update_status("No video files found to process")
                return
            
            # Set up progress tracking
            self.set_total_items(len(video_files))
            self.set_processed_items(0)
            
            # Create context directory if it doesn't exist
            os.makedirs(self.context_dir, exist_ok=True)
            
            # Load Whisper model once for all transcriptions
            self._update_status(f"Loading Whisper model ({self.whisper_model_size})")
            self._load_whisper_model()
            
            start_time = time.time()
            
            # Process each video file
            for i, video_file in enumerate(video_files):
                video_path = video_file if os.path.isabs(video_file) else os.path.join(self.videos_dir, video_file)
                
                self._update_status(f"Processing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
                
                # Check if context file already exists
                if self._context_file_exists(video_path):
                    logging.info(f"Context file already exists for {os.path.basename(video_path)}, skipping")
                    self.skipped_count += 1
                    self.increment_processed_items()
                    self._update_progress()
                    continue
                
                # Process the video file
                success = self._process_video_file(video_path)
                if success:
                    self.processed_count += 1
                
                self.increment_processed_items()
                
                # Update processing time and progress
                elapsed_time = time.time() - start_time
                self.set_total_processing_time(elapsed_time)
                self._update_progress()
            
            # Final status update
            self._update_status(f"Transcription complete. Processed: {self.processed_count}, Skipped: {self.skipped_count}")
            self.set_fetch_completed(True)
            
            # Notify subscribers with summary data
            summary_data = {
                "type": "transcription_summary",
                "processed_count": self.processed_count,
                "skipped_count": self.skipped_count,
                "total_files": len(video_files),
                "context_dir": self.context_dir
            }
            self.update(summary_data)
            
        except Exception as e:
            error_msg = f"Error during video transcription: {str(e)}"
            logging.error(error_msg)
            self._update_status(error_msg)
            raise

    def get_icon(self):
        """
        Return SVG icon for the video transcription data source.
        """
        return """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="4" width="18" height="12" rx="2" stroke="currentColor" stroke-width="2" fill="none"/>
            <path d="M10 8L15 11L10 14V8Z" fill="currentColor"/>
            <path d="M3 20H21" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <circle cx="5" cy="20" r="1" fill="currentColor"/>
            <circle cx="12" cy="20" r="1" fill="currentColor"/>
            <circle cx="19" cy="20" r="1" fill="currentColor"/>
        </svg>
        """

    def get_connection_data(self):
        """
        Return connection configuration for the video transcription data source.
        """
        return {
            "connection_type": "LOCAL_VIDEO_TRANSCRIPTION",
            "fields": [
                "videos_dir",
                "context_dir", 
                "whisper_model_size",
                "specific_video_path"
            ]
        }

    def _get_video_files(self):
        """
        Get list of video files to process based on configuration.
        """
        if self.specific_video_path:
            # Process specific video file
            if not os.path.exists(self.specific_video_path):
                raise FileNotFoundError(f"Video file '{self.specific_video_path}' not found")
            
            if not self.specific_video_path.endswith(('.mp4', '.mkv')):
                raise ValueError(f"File '{self.specific_video_path}' is not a supported video format")
            
            return [self.specific_video_path]
        else:
            # Process all videos in directory
            if not os.path.exists(self.videos_dir):
                raise FileNotFoundError(f"Videos directory '{self.videos_dir}' not found")
            
            video_files = [f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4', '.mkv'))]
            if not video_files:
                return []
            
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.videos_dir, x)), reverse=True)
            return video_files

    def _load_whisper_model(self):
        """
        Load the Whisper model for transcription.
        """
        if not self.whisper_model:
            self.whisper_model = whisper.load_model(self.whisper_model_size)
            logging.info(f"Loaded Whisper model '{self.whisper_model_size}'")

    def _process_video_file(self, video_path):
        """
        Process a single video file for transcription.
        """
        try:
            logging.info(f"Processing video: {video_path}")
            
            # Create temporary directory for audio processing
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Extract audio from video
                wav_path = os.path.join(tmpdirname, "audio.wav")
                audio_file = self._extract_audio_from_video(video_path, wav_path)
                
                if audio_file:
                    # Transcribe audio
                    transcript = self._transcribe_audio(audio_file)
                    
                    if transcript:
                        # Save transcript as JSON
                        output_path = self._generate_output_path(video_path)
                        self._save_transcript_as_json(transcript, video_path, output_path)
                        
                        # Notify subscribers with transcription data
                        transcription_data = {
                            "type": "video_transcription",
                            "video_path": video_path,
                            "video_filename": os.path.basename(video_path),
                            "transcript": transcript,
                            "output_path": output_path,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.update(transcription_data)
                        
                        logging.info(f"Transcript saved to {output_path}")
                        return True
                    else:
                        logging.warning(f"No transcript was generated for {os.path.basename(video_path)}")
                else:
                    logging.error(f"Failed to extract audio from {os.path.basename(video_path)}")
                    
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")
            
        return False

    def _extract_audio_from_video(self, video_path, output_path):
        """
        Extract audio from video file and save as WAV.
        """
        try:
            audio = AudioSegment.from_file(video_path)
            # Convert to mono for better transcription
            audio = audio.set_channels(1)
            # Export as WAV
            audio.export(output_path, format="wav")
            logging.info(f"Extracted audio from {video_path} to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error extracting audio from {video_path}: {e}")
            return None

    def _transcribe_audio(self, audio_path):
        """
        Transcribe audio to text using Whisper.
        """
        try:
            result = self.whisper_model.transcribe(audio_path)
            logging.info(f"Transcribed audio file {audio_path}")
            return result['text']
        except Exception as e:
            logging.error(f"Error transcribing {audio_path}: {e}")
            return ""

    def _generate_output_path(self, video_path):
        """
        Generate output filename using video recording time.
        """
        video_mtime = os.path.getmtime(video_path)
        video_recording_datetime = datetime.fromtimestamp(video_mtime)
        timestamp = video_recording_datetime.strftime("%Y%m%d%H%M%S")
        output_filename = f"context-{timestamp}.json"
        return os.path.join(self.context_dir, output_filename)

    def _save_transcript_as_json(self, transcript, video_path, output_path):
        """
        Save transcript as JSON with all metadata and transcription field.
        """
        video_hash = self._get_video_hash(video_path)
        video_size = os.path.getsize(video_path)
        video_mtime = os.path.getmtime(video_path)
        
        # Convert timestamp to readable format
        video_recording_time = datetime.fromtimestamp(video_mtime).isoformat()
        
        data = {
            'video_path': video_path,
            'video_filename': os.path.basename(video_path),
            'video_hash': video_hash,
            'video_size': video_size,
            'video_mtime': video_mtime,
            'video_recording_time': video_recording_time,
            'transcription_time': datetime.now().isoformat(),
            'whisper_model': self.whisper_model_size,
            'transcription': transcript
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_video_hash(self, video_path):
        """
        Generate a hash of the video file for unique identification.
        """
        try:
            with open(video_path, 'rb') as f:
                # Read first 1MB and last 1MB for faster hashing
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                
                if file_size <= 2 * 1024 * 1024:  # If file is 2MB or smaller, hash entire file
                    f.seek(0)
                    data = f.read()
                else:
                    f.seek(0)
                    data = f.read(1024 * 1024)  # First 1MB
                    f.seek(-1024 * 1024, 2)  # Last 1MB
                    data += f.read()
                
                return hashlib.md5(data).hexdigest()
        except Exception as e:
            logging.error(f"Error generating hash for {video_path}: {e}")
            return None

    def _context_file_exists(self, video_path):
        """
        Check if a context file already exists for this video using multiple methods.
        """
        video_filename = os.path.basename(video_path)
        video_hash = self._get_video_hash(video_path)
        
        # Get all context files and check each one
        context_files = glob.glob(os.path.join(self.context_dir, "*.json"))
        
        for context_file in context_files:
            if self._check_context_metadata(context_file, video_path, video_hash, video_filename):
                return True
        
        return False

    def _check_context_metadata(self, context_file, video_path, video_hash, video_filename):
        """
        Check if a context file contains metadata matching the video.
        """
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check multiple ways to match the video
            # Method 1: Exact video path match
            if data.get('video_path') == video_path:
                return True
                
            # Method 2: Video filename match
            if data.get('video_filename') == video_filename:
                return True
                
            # Method 3: Hash match (if available)
            if video_hash and data.get('video_hash') == video_hash:
                return True
        
        except Exception as e:
            logging.error(f"Error checking metadata in {context_file}: {e}")
        
        return False

    def _update_status(self, status):
        """
        Update status via callback if available.
        """
        if self.status_callback:
            self.status_callback(self.get_name(), status)
        logging.info(f"[{self.get_name()}] {status}")

    def _update_progress(self):
        """
        Update progress via callback if available.
        """
        if self.progress_callback:
            self.progress_callback(
                self.get_name(),
                self.get_total_to_process(),
                self.get_total_processed(),
                self.estimated_remaining_time()
            ) 