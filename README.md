#  Gesture Snake Game
This project is a real-time interactive Snake game system based on hand gesture recognition. 
This project combines computer vision techniques with classic gaming to create an innovative, contactless control experience. 
Using a standard webcam, the system captures hand gestures and translates them into game control commands, allowing players to control the snake's movement through intuitive hand motions.  


This project is created by two students from Macau University of Science and Technology.

##  Software Requirements  
The following Python packages are required to run the project:  
torch==2.5.2  
torchvision==0.20.2  
opencv-python==4.10.0.84  
mediapipe==0.10.18  
pygame==2.6.1  
numpy==1.26.4  
pandas==2.2.3  
matplotlib==3.8.0  
scikit-learn==1.5.2  
tqdm==4.66.5  
pillow==10.3.0  

##  Testing Commands  
Test model performance  
```python
python test.py
```
Run Real-time Gesture Recognition
```python
python predict.py
```

Run Complete Game System
```python
python gui_camera.py
```

##  Game Controls
The game control gestures are as follows:  
Thumbs up (thumb extended, other four fingers closed, thumb pointing upward): Move up  
Thumbs down: Move down  
Thumbs left: Move left  
Thumbs right: Move downr right  
Fist (all fingers closed into a fist): Pause game  
Open hand (all five fingers extended and spread): Continue game  

##  Known Issues
Left-Right Direction Inversion: Due to dataset characteristics, when using the left hand for recognition, left and right directions may be inverted. This issue does not occur with the right hand. For optimal performance, use your right hand for gesture control.  

Camera Initialization Delay: There is a brief delay (typically 2-3 seconds) when starting the game as the camera initializes and begins capturing. This is normal behavior and ensures stable video feed acquisition.
