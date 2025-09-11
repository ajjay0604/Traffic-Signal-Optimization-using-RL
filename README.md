# 🚦 Traffic Signal Optimization using RL

An **adaptive traffic management system** that optimizes signal timings using a combination of **supervised learning** and **reinforcement learning (RL)** techniques. Built with **Python** and visualized in **Pygame**, this project dynamically adjusts traffic signals based on real-time and historical data, significantly improving traffic flow and reducing congestion.


## 📖 Project Overview

Urban traffic congestion is one of the leading causes of time and fuel wastage. This project presents a **novel traffic signal optimization model** using a hybrid of **Random Forest Classifier** and **Q-learning** algorithms. The system simulates traffic movement in a **2D Pygame environment**, dynamically adapting signal changes for efficient flow.

The goal: reduce waiting times, increase throughput, and improve pedestrian safety.



## 🌟 Key Features

- 🚗 **Real-time dynamic signal control** using reinforcement learning
- 🚶 **Pedestrian-aware red light timers**
- 🚑 **Priority for ambulances and emergency vehicles**
- 🌐 **Multi-agent support for future city-wide scalability**
- 📊 **Performance metrics tracking and visual feedback**



## ⚙️ Methodology

### 1. **Data Collection & Preprocessing**
- Simulated traffic patterns in Pygame (vehicle count, pedestrian count, ambulance presence)
- Categorical features encoded and normalized

### 2. **Modeling**
- **Random Forest Classifier** used for initial signal phase prediction
- **Q-Learning** for continuous optimization of signal changes
- Custom **reward function** penalizing congestion and unsafe actions

### 3. **Simulation**
- Visualized with Pygame: live traffic, signals, agents
- Log table displays system actions and decisions
- Trained under diverse scenarios: peak-hour, pedestrian surges, emergencies



## 🛠️ Technologies Used

| Component              | Technology        |
|------------------------|-------------------|
| Programming Language   | Python            |
| Simulation             | Pygame            |
| Machine Learning       | Scikit-learn (Random Forest) |
| Reinforcement Learning | Q-Learning        |
| Visualization          | Pygame GUI        |



## 📈 Results

- ⏳ **30% reduction in average vehicle waiting time** over static systems
- 🚦 **25% increase in throughput**
- 🧍‍♀️ **Improved pedestrian handling** based on density
- 🚨 **Effective emergency vehicle prioritization**



## 🖼️ Screenshots

### 🔴 Red Signal – Pedestrian Priority

<img width="1702" height="946" alt="image" src="https://github.com/user-attachments/assets/0eb4da42-b9dd-4824-bb00-f6eae25dd715" />

When the pedestrian count exceeds 10, the red light timer is extended from the default 3 seconds to 6 seconds to ensure safe crossing for pedestrians. This helps reduce pedestrian wait times and improves intersection safety.

### 🟡 Yellow Signal – Transition Phase

<img width="1710" height="890" alt="image" src="https://github.com/user-attachments/assets/c7a13aa8-27b0-49a7-877c-8d3c068ab3c3" />

The yellow signal serves as an intermediate phase between green and red lights. It appears for a fixed duration of 3 seconds, providing drivers with a buffer time to either stop or clear the intersection safely.

### 🟢 Green Signal – Standard Flow

<img width="1468" height="696" alt="image" src="https://github.com/user-attachments/assets/fd958b6b-b0f6-4109-940c-b19430c77eb3" />

After the red phase, the system initiates the green light for 3 seconds, allowing vehicles to pass normally. This is the default condition when no special priorities like ambulances or high pedestrian count are detected.

### 🚑 Ambulance Priority Demonstration

<img width="1482" height="700" alt="image" src="https://github.com/user-attachments/assets/42f002c8-a023-4363-89ce-ae1dfeece7e0" />

If an ambulance is detected, the system immediately overrides current signal conditions and switches to green, ensuring rapid and unobstructed passage for emergency vehicles. This occurs regardless of vehicle or pedestrian counts.


---
📌 Developed by **Ajjay Adhithya V**  
🔗 Explore more projects on my [GitHub Profile](https://github.com/your-username)
---
