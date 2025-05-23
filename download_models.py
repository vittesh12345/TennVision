import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from a URL with a progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Download tennis ball detection model
    model_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    model_path = os.path.join('models', 'tennis_ball_model.pt')
    
    if not os.path.exists(model_path):
        print("Downloading tennis ball detection model...")
        download_file(model_url, model_path)
        print("Model downloaded successfully!")
    else:
        print("Tennis ball detection model already exists.")

if __name__ == "__main__":
    main() 