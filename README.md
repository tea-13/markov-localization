# Markov Localization

![alt text](Assets/image.png)

## 🔹 **General description**

The application is designed to simulate the movement of a robot in a two-dimensional space taking into account **Markov localization (ML)**. It models the process of robot navigation in an environment with an unknown starting position and orientation, using information about movements and detected marks on the map.

## 🔹 **Main functions**

✅ **Map definition** - the user defines a discrete space (a grid saved as an image `map.png`), where each cell can contain colored markers for orientation.  
✅ **Movement definition** - the robot executes commands: **"F" (forward), "B" (backward), "CW" (turn clockwise), "CCW" (turn counterclockwise)**.  
✅ **Cyclic map** - the robot can move beyond the map boundaries, appearing on the opposite side.  
✅ **Trajectory recording** — after finding a marker, the robot saves the entire path to the next checkpoint.  
✅ **Start position determination** — based on the trajectory and detected markers, the algorithm restores the initial position, orientation, and final state.  
✅ **Visualization** — displaying the robot's movement on the map in real time with animation of its turns and movements.  


## 🔹 **Key features**

🔹 **Reverse path analysis** — the algorithm compares the recorded route with the map to determine the starting position.  
🔹 **Flexibility** — the user can set up custom maps with any number of color marks.  
🔹 **Robot settings** — changing the movement model, probabilistic errors, and localization methods.  


## 🔹 **Application**

📍 **Robotics Research** — testing localization and navigation algorithms.

📍 **Training** — learning the principles of Markov localization visually.

📍 **Autonomous Systems Development** — testing navigation in closed environments.

## Original paper
[paper](https://arxiv.org/abs/1106.0222)
