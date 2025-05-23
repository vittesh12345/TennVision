# TennVision - Tennis Stroke Analysis System

TennVision is an advanced tennis stroke analysis system that uses computer vision and machine learning to analyze tennis strokes in real-time. The system can detect and classify different types of tennis strokes, track player movement, and provide detailed feedback on technique.

## Features

- Real-time tennis stroke detection and classification
- Pose estimation for detailed movement analysis
- Specialized tennis ball tracking
- Detailed stroke analysis and feedback
- User-friendly GUI interface
- Support for video upload and playback
- Professional coaching tips and recommendations

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow
- YOLOv8
- Tkinter
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vittesh12345/TennVision.git
cd TennVision
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root directory
   - Add your DeepSeek API key:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```
   - Note: Keep your `.env` file secure and never commit it to version control

5. Download required model files:
   - Run the model download script:
   ```bash
   python download_models.py
   ```
   - This will download:
     - YOLOv5 tennis ball detection model
     - YOLOv8 general model (as fallback)
   - Download the tennis stroke classification model:
     - Option 1: Train your own model using `train_model.py`
     - Option 2: Download the pre-trained model from [Releases](https://github.com/vittesh12345/TennVision/releases)
     - Place the `tennis_stroke_model.h5` file in the project root directory

## Usage

1. Run the main application:
```bash
python tennis_analyzer.py
```

2. Use the GUI to:
   - Upload a tennis video
   - Play/pause the video
   - Analyze strokes
   - View detailed feedback and tips

## Project Structure

- `tennis_analyzer.py`: Main application file
- `train_model.py`: Script for training the stroke classification model
- `organize_clips.py`: Utility for organizing training data
- `download_models.py`: Script to download required models
- `requirements.txt`: Project dependencies
- `tennis_stroke_model.h5`: Pre-trained stroke classification model (not included in git)
- `.env`: Environment variables file (not tracked in git)
- `models/`: Directory containing downloaded models (not tracked in git)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose estimation
- YOLOv8 for object detection
- YOLOv5 for tennis ball detection
- TensorFlow for machine learning capabilities 