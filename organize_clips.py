import os
import shutil

def organize_clips():
    # Create training directories if they don't exist
    os.makedirs('tennis_clips/forehand', exist_ok=True)
    os.makedirs('tennis_clips/backhand', exist_ok=True)
    os.makedirs('tennis_clips/serve', exist_ok=True)
    
    # Copy forehand videos from the provided path
    forehand_dir = '/Users/vitteshmaganti/my_python_scripts/tennisClass/forehand'
    if os.path.exists(forehand_dir):
        for video in os.listdir(forehand_dir):
            if video.endswith(('.mp4', '.avi', '.mov')):
                src = os.path.join(forehand_dir, video)
                dst = os.path.join('tennis_clips/forehand', video)
                shutil.copy2(src, dst)
                print(f"Copied {video} to forehand directory")
    
    # Copy backhand videos from the provided path
    backhand_dir = '/Users/vitteshmaganti/Desktop/Tennis1/backhand'
    if os.path.exists(backhand_dir):
        for video in os.listdir(backhand_dir):
            if video.endswith(('.mp4', '.avi', '.mov')):
                src = os.path.join(backhand_dir, video)
                dst = os.path.join('tennis_clips/backhand', video)
                shutil.copy2(src, dst)
                print(f"Copied {video} to backhand directory")

if __name__ == "__main__":
    organize_clips() 