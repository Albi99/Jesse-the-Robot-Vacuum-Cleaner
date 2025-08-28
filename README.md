# Jesse the (ro)bot 🤠🤖🧹
*A robotic vacuum cleaner simulation powered by Reinforcement Learning.*

![cover](images/cover.png)

## 🚀 About
This repository contains the implementation of **Jesse the (ro)bot**, a simulated robotic vacuum cleaner trained with Reinforcement Learning as part of the course *Complex Systems: Models and Simulation* at Università degli studi di Milano-Bicocca (Italy).  

The goal was not to deliver a finished industrial product, but to explore RL in practice, design a realistic environment, and see how an agent can learn to clean, avoid collisions, and return to its base — starting from zero knowledge.

## 🧠 Main Features
- Continuous 2D environment + discrete occupancy grid abstraction.
- Robot equipped with LiDAR and collision sensors.
- **Two RL agents implemented**:  
  - Q-learning (baseline)  
  - PPO (single-MLP and multi-encoder)  
- Curriculum learning with maps of increasing difficulty.  
- Reward shaping for exploration, cleaning, and safe return.  
- Modular PPO multi-encoder architecture:  
  - 2D CNN for local occupancy patch  
  - 1D CNN for LiDAR encoder  
  - Side-views encoder  
  - Scalars encoder  

## 📊 Results
- The PPO multi-encoder agent developed systematic exploration and cleaning behaviors.  
- Training on CPU-only hardware was extremely slow: each meaningful modification required some times of training just to observe its effect.  
- Despite this, the project demonstrated how RL, careful reward shaping, and structured state encoding can produce meaningful autonomous behavior.

## 🔮 Future Work
- Improve reward shaping (the most difficult challenge so far).  
- Continue training on harder maps (requires GPU/HPC resources).  
- Experiment with network architectures (deeper CNNs, attention).  
- Add noise to motion dynamics and sensors for realism.  
- Introduce static and dynamic obstacles (e.g., furniture, pets 🐱).  
- Extend the battery model with orientation/acceleration costs.  

## 📂 Repository Structure
```
Jesse-the-Robot-Vacuum-Cleaner
│   README.md
│   report.pdf
│   cover.png
│
└───code
    │   play.py                # entry point to run the simulation
    │
    └───rlrc
        │   joystick.py        # optional joystick controller
        │   test.py            # evaluation script
        │   train.py           # training script (PPO with curriculum learning)
        │   utils.py           # helper functions (plotting, metrics, etc.)
        │   __init__.py
        │
        ├───classes
        │   │   agent.py       # Q-learning and PPO agents (you have to switch between branches)
        │   │   encoders.py    # CNN/MLP encoders for multi-modal inputs
        │   │   environment.py # continuous environment simulation
        │   │   graphics.py    # rendering and visualization
        │   │   robot.py       # robot dynamics, sensors, reward shaping
        │
        └───constants
            │   colors.py
            │   configuration.py
            │   maps.py        # predefined training/test maps
```

## 📖 Documentation
For a complete description of the project, see the full report:  
👉 [Report PDF](./report.pdf)

---

*Created by Alberto Sormani (2025) as part of one of the courses of the Master's Degree in Computer Science at the University of Milan-Bicocca.*
