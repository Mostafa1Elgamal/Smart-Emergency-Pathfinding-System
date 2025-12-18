# 🚑 S.E.R.S - Smart Emergency Response System

**S.E.R.S** is an interactive AI simulation designed to optimize ambulance routing in a dynamic city grid. The project demonstrates various pathfinding algorithms to navigate through traffic, highways, and obstacles to reach patients before their health runs out.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Pygame](https://img.shields.io/badge/Library-Pygame-green?style=flat&logo=pygame)
![Status](https://img.shields.io/badge/Status-Educational-orange)

---


## 🚀 Features

- **Dynamic City Generation:** Randomly generates city layouts with buildings, highways (low cost), streets (medium cost), and traffic jams (high cost).
- **Patient Health Decay:** Simulation of real-time urgency; patients lose health over time, requiring the most efficient path to save them.
- **Visual vs. Instant Mode:** Watch the algorithms explore the map step-by-step (Visual) or calculate the result instantly (Real-time).
- **Multiple Algorithms:** Implementation of both Uninformed and Informed search strategies.
- **TSP Solver:** Uses a Genetic Algorithm to solve the Traveling Salesman Problem when multiple patients are present.

---

## 🤖 Implemented Algorithms

The system allows you to switch between algorithms on the fly to compare their performance:

### Uninformed Search
1. **BFS (Breadth-First Search):** Guarantees the shortest path in an unweighted grid.
2. **DFS (Depth-First Search):** Explores as far as possible along each branch before backtracking.
3. **IDS (Iterative Deepening Search):** A combination of DFS's space efficiency and BFS's completeness.

### Informed Search
4. **UCS (Uniform Cost Search):** Explores paths based on path cost (accounts for traffic vs. highways).
5. **A* (A-Star):** Uses a heuristic function (Manhattan distance) + cost to find the optimal path efficiently.
6. **Hill Climbing:** A local search algorithm that moves towards the goal based on heuristic value (greedy approach).

### Optimization
7. **Genetic Algorithm:** Used for the **Multiple Patients** scenario. It treats the route as a TSP (Traveling Salesman Problem) to find the best order to visit all patients.

---

## 🎮 Controls

| Key | Function |
| :--- | :--- |
| **B** | Run **BFS** |
| **D** | Run **DFS** |
| **U** | Run **UCS** (Uniform Cost Search) |
| **I** | Run **IDS** (Iterative Deepening) |
| **A** | Run **A*** Search |
| **H** | Run **Hill Climbing** |
| **G** | Run **Genetic Algorithm** (for multiple patients) |
| **Shift + Key** | Run the selected algorithm in **Instant Mode** (no visualization) |
| **M** | Spawn **5 Random Patients** (Test for Genetic Algo) |
| **R** | **Reset/Reboot** the Map (New City Layout) |
| **P** | Generate **New Start/Goal** points |
| **C** | **Clear** current path |

---

## 🗺️ Map Legend

- **🟩 Green Block:** Ambulance (Start)
- **🔵 Blue Circle:** Patient
- **⬜ White/Grey:** Visited/Path Nodes
- **🟦 Dark Blue Road:** Highway (Cost: 1)
- **🟫 Grey Road:** Normal Street (Cost: 3)
- **🟥 Red Road:** Heavy Traffic (Cost: 15)
- **⬛ Black:** Building (Obstacle)

---

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mostafa1Elgamal/Smart-Emergency-Pathfinding-System.git](https://github.com/Mostafa1Elgamal/Smart-Emergency-Pathfinding-System.git)
   cd Smart-Emergency-Pathfinding-System
   ```

2. **Install dependencies:**
   You need Python installed. Then install `pygame`:
   ```bash
   pip install pygame
   ```

3. **Run the simulation:**
   ```bash
   python AiProject.py
   ```

---
      
