import os
import sys
import logging
from dotenv import load_dotenv
from SubjectiveTranscribeLocalVideoDataSource import SubjectiveTranscribeLocalVideoDataSource

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    filename='local_video_transcription.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Legacy main function for backward compatibility
def main():
    """
    Legacy main function that creates and runs the data source.
    Maintained for backward compatibility with the original script.
    """
    # Determine parameters based on command line arguments
    params = {}
    
    if len(sys.argv) > 1:
        # Process specific video file
        video_path = sys.argv[1]
        params['specific_video_path'] = video_path
        print(f"Processing specific video file: {video_path}")
    else:
        # Use default directories
        videos_dir = os.getenv("VIDEOS_DIR", "videos")
        context_dir = os.getenv("CONTEXT_DIR", "context")
        params['videos_dir'] = videos_dir
        params['context_dir'] = context_dir
        print(f"Processing videos from directory: {videos_dir}")
    
    # Create and run the data source
    data_source = SubjectiveTranscribeLocalVideoDataSource(
        name="LocalVideoTranscription",
        params=params
    )
    
    try:
        data_source.fetch()
        print(f"\nBatch processing complete!")
        print(f"Processed: {data_source.processed_count} files")
        print(f"Skipped (already exists): {data_source.skipped_count} files")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 