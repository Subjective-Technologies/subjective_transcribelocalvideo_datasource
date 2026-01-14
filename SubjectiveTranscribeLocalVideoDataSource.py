import os
import tempfile
import logging
from datetime import datetime
import whisper
from pydub import AudioSegment
from pydub.utils import which
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
        self.whisper_model_size = self.params.get('whisper_model_size') or os.getenv("WHISPER_MODEL_SIZE") or WHISPER_MODEL_SIZE
        self.videos_dir = self.params.get('videos_dir') or os.getenv("VIDEOS_DIR")
        context_dir = self.params.get('context_dir') or self.params.get("TARGET_DIRECTORY") or os.getenv("CONTEXT_DIR")
        self.context_dir = context_dir if context_dir else "context"
        self.specific_video_path = self.params.get('specific_video_path', None)

        self._configure_ffmpeg()
        
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

            if not self.specific_video_path and not self.videos_dir:
                self._update_status("No videos_dir configured; waiting for pipeline input")
                return
            
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
        import os
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.svg')
        try:
            if os.path.exists(icon_path):
                with open(icon_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception:
            pass
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="2" fill="#111827"/><path d="M10 8L15 11L10 14V8Z" fill="#fff"/></svg>'

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

    def process_input(self, data):
        """
        Process a single video file triggered by pipeline input.
        """
        if not isinstance(data, dict):
            return

        video_path = data.get("path") or data.get("dest_path")
        if not video_path:
            return

        if not video_path.endswith(('.mp4', '.mkv')):
            return

        if not self.context_dir:
            self.context_dir = self._resolve_context_path()

        os.makedirs(self.context_dir, exist_ok=True)

        if not self.whisper_model_size:
            self.whisper_model_size = WHISPER_MODEL_SIZE

        self._load_whisper_model()
        self._process_video_file(video_path)

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
            if not self.videos_dir:
                return []
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
                        output_payload = self._build_transcript_payload(transcript, video_path)
                        output_path = self._write_context_output(output_payload)
                        
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

    def _configure_ffmpeg(self):
        """
        Ensure ffmpeg is available for pydub audio extraction.
        """
        if which("ffmpeg"):
            return

        try:
            import imageio_ffmpeg
        except Exception:
            logging.warning("ffmpeg not found and imageio-ffmpeg not available")
            return

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            AudioSegment.converter = ffmpeg_path

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

    def _build_transcript_payload(self, transcript, video_path):
        """
        Build transcript payload for parent context writer.
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
        return data

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

    def process_input(self, data):
        """
        Process input data from a pipeline dependency (e.g., file change notification).
        This method is called when this data source is part of a pipeline and receives
        data from a dependency data source.

        Args:
            data: Input data from the dependency (typically a file change notification)
        """
        try:
            logging.info(f"Received pipeline input: {data}")

            # Extract the file path from the notification data
            file_path = None
            if isinstance(data, dict):
                # Try different common keys for file path
                file_path = data.get('path') or data.get('dest_path') or data.get('file_path')
            elif isinstance(data, str):
                file_path = data

            if not file_path:
                logging.warning(f"No file path found in pipeline input: {data}")
                return

            # Check if it's a video file
            if not file_path.endswith(('.mp4', '.mkv')):
                logging.info(f"Ignoring non-video file: {file_path}")
                return

            # Check if the file exists
            if not os.path.exists(file_path):
                logging.warning(f"File does not exist: {file_path}")
                return

            # Check if context file already exists
            if self._context_file_exists(file_path):
                logging.info(f"Context file already exists for {os.path.basename(file_path)}, skipping")
                return

            # Load Whisper model if not already loaded
            if not self.whisper_model:
                self._update_status(f"Loading Whisper model ({self.whisper_model_size})")
                self._load_whisper_model()

            # Process the video file
            self._update_status(f"Processing video from pipeline: {os.path.basename(file_path)}")
            success = self._process_video_file(file_path)

            if success:
                self._update_status(f"Successfully transcribed: {os.path.basename(file_path)}")
                self.processed_count += 1
            else:
                self._update_status(f"Failed to transcribe: {os.path.basename(file_path)}")

        except Exception as e:
            error_msg = f"Error processing pipeline input: {str(e)}"
            logging.error(error_msg)
            self._update_status(error_msg) 
