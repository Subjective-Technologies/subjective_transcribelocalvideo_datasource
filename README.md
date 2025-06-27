# Transcribe Local Video Data Source

A BrainBoost data source for transcribing local video files using OpenAI's Whisper AI. This project converts video files to audio and generates accurate transcriptions stored as JSON context files.

## Features

- 🎥 **Video Processing**: Supports MP4 and MKV video formats
- 🔊 **Audio Extraction**: Automatically extracts and converts audio from video files
- 🤖 **AI Transcription**: Uses OpenAI Whisper for high-quality speech-to-text conversion
- 📊 **Progress Tracking**: Real-time progress updates and status callbacks
- 🔄 **Subscriber Pattern**: Notifies subscribers with transcription data
- 📁 **Smart Deduplication**: Avoids re-processing already transcribed videos
- ⚙️ **Configurable**: Flexible configuration through parameters or environment variables
- 🔗 **BrainBoost Integration**: Fully compatible with BrainBoost data pipeline system

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video/audio processing)

#### Install FFmpeg:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use chocolatey:
```bash
choco install ffmpeg
```

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Subjective-Technologies/transcribelocalvideo_datasource.git
cd transcribelocalvideo_datasource
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### As a BrainBoost Data Source

```python
from transcribe_local_video import SubjectiveTranscribeLocalVideoDataSource

# Create the data source
data_source = SubjectiveTranscribeLocalVideoDataSource(
    name="VideoTranscription",
    params={
        'videos_dir': '/path/to/videos',
        'context_dir': '/path/to/context',
        'whisper_model_size': 'base'
    }
)

# Set up callbacks (optional)
def progress_callback(name, total, processed, estimated_time):
    print(f"{name}: {processed}/{total} files processed, {estimated_time:.1f}s remaining")

def status_callback(name, status):
    print(f"{name}: {status}")

data_source.set_progress_callback(progress_callback)
data_source.set_status_callback(status_callback)

# Subscribe to receive transcription data
class MySubscriber:
    def notify(self, data):
        if data['type'] == 'video_transcription':
            print(f"Transcribed: {data['video_filename']}")
            print(f"Transcript: {data['transcript'][:100]}...")

data_source.subscribe(MySubscriber())

# Start processing
data_source.fetch()
```

### Command Line Usage (Legacy)

```bash
# Process all videos in default directory
python transcribe_local_video.py

# Process specific video file
python transcribe_local_video.py /path/to/video.mp4
```

## Configuration

### Parameters

- `videos_dir`: Directory containing video files (default: "videos")
- `context_dir`: Directory for output JSON files (default: "context")
- `whisper_model_size`: Whisper model size - 'tiny', 'base', 'small', 'medium', 'large' (default: 'base')
- `specific_video_path`: Process a single video file instead of directory

### Environment Variables

You can also configure using environment variables:

```bash
export VIDEOS_DIR="/path/to/videos"
export CONTEXT_DIR="/path/to/context"
```

Or create a `.env` file:
```
VIDEOS_DIR=/path/to/videos
CONTEXT_DIR=/path/to/context
```

## Output Format

The transcription data is saved as JSON files with comprehensive metadata:

```json
{
  "video_path": "/path/to/video.mp4",
  "video_filename": "video.mp4",
  "video_hash": "abc123...",
  "video_size": 1048576,
  "video_mtime": 1704067200.0,
  "video_recording_time": "2024-01-01T12:00:00",
  "transcription_time": "2024-01-08T15:30:00.123456",
  "whisper_model": "base",
  "transcription": "Hello, this is the transcribed text from the video..."
}
```

## Data Notifications

The data source sends structured notifications to subscribers:

### Individual Video Transcription
```json
{
  "type": "video_transcription",
  "video_path": "/path/to/video.mp4",
  "video_filename": "video.mp4",
  "transcript": "Transcribed text content...",
  "output_path": "/path/to/context/context-20240108120000.json",
  "timestamp": "2024-01-08T12:00:00.000000"
}
```

### Processing Summary
```json
{
  "type": "transcription_summary",
  "processed_count": 5,
  "skipped_count": 2,
  "total_files": 7,
  "context_dir": "/path/to/context"
}
```

## Whisper Models

Choose the appropriate model size based on your needs:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny  | 39 MB | Fastest | Good |
| base  | 74 MB | Fast | Better |
| small | 244 MB | Medium | Good |
| medium| 769 MB | Slow | Very Good |
| large | 1550 MB | Slowest | Best |

## Performance

- **GPU Acceleration**: Automatically uses CUDA if available for faster processing
- **Smart Deduplication**: Skips already processed videos using multiple identification methods
- **Memory Efficient**: Uses temporary files for audio processing
- **Progress Tracking**: Real-time progress updates with time estimation

## Dependencies

- `subjective-abstract-data-source-package` - BrainBoost data source framework
- `openai-whisper` - AI transcription engine
- `pydub` - Audio processing
- `ffmpeg-python` - Video/audio conversion
- `python-dotenv` - Environment configuration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Issues: [GitHub Issues](https://github.com/Subjective-Technologies/transcribelocalvideo_datasource/issues)
- Email: support@subjectivetechnologies.com

## Changelog

### 1.0.0
- Initial release
- SubjectiveTranscribeLocalVideoDataSource implementation
- OpenAI Whisper integration
- Progress tracking and status callbacks
- BrainBoost data pipeline compatibility 