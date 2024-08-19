# Tello-Obstacle-Detection-using-Depth-Anything
This code is the basis for creating an obstacle detection system for autonomous navigation of a DJI Tello using Depth Anything

# Requirements/Setup  
### TelloPy:
Allows you to easily send commands and receive video from the drone. Can be easily installed running:
```bash
pip install tellopy
```
### Depth Anything
This is the heart of the code, capable of creating a depth estimate through images or videos.
In this code I still use version 1. To install, access the developer's Git page [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything)

### Usage
This code controls Tello manually, using the following controls:

- **T** - takeoff
- **L** - land
- **Q** - rotate counter clockwise
- **E** - rotate clockwise
- **D** - move right
- **A** - move left
- **W** - move forward
- **S** - move backward
- **U** - move up
- **J** - move down

### How it works
This code is a type of "onboard control", and the drone is controlled manually, but the Depth Anythig algorithm detects whether the area within the rectangle drawn in the image is free or closed. If it is closed, it divides the rectangle into four parts and detects which one is freest from obstacles. When performing this loop, a message like "free path to the left..." or "free path to the top" appears in the prompt.

### Future work
Implement a pre-defined route system so that the drone can avoid the obstacle and return to its original path. It is also worth studying reinforcement learning and working with simulations.
