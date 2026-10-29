# Tutorial-CE-# 🧬 Evolutionary Computing – Genetic Algorithm Assignment (JIE42903)

This repository contains the source code and implementation for the **JIE42903 Evolutionary Computing** course assignment at **Universiti Malaysia Kelantan (UMK)**.

---

## 📌 Overview

This project explores two main applications of **Genetic Algorithms (GA)** built using **Python** and deployed as an interactive **Streamlit** multi-page web application:

1. **TV Program Scheduling Optimization** (Course Assignment Core Task)
2. **Traveling Salesperson Problem (TSP)** (Interactive Route Optimization Page)

---

## 🚀 Features & Modules

### 📅 Module 1: TV Program Scheduling GA
Optimizes television program placement across 18 hourly time slots (06:00 to 24:00) to maximize total audience reach based on historical viewer ratings.
* **Custom Dataset:** Reads program rating scores directly from a modified `program_ratings.csv`.
* **Interactive Parameters:** Allows custom tuning of:
  * **Crossover Rate ($CO\_R$):** Configurable range ($0.00$ – $0.95$, default $0.80$).
  * **Mutation Rate ($MUT\_R$):** Configurable range ($0.01$ – $0.05$).
* **Experimentation Support:** Facilitates running multiple trials (Trial 1, Trial 2, Trial 3) for comparative analysis in the assignment report.

### 🗺️ Module 2: Traveling Salesperson Problem (TSP) GA
Solves the classical NP-hard Traveling Salesperson Problem across cities using evolutionary operations.
* **Custom Coordinates:** Interactive Streamlit form allowing users to input custom city names and 2D coordinates $(x, y)$.
* **Visual Route Plotting:** Generates matplotlib route maps highlighting city coordinates, connections, and the optimal path sequence.
* **Selection & Variation:** Implements Roulette Wheel Selection, Order/Cut Crossover, and Swap Mutation.

---
## Author

* Student Name: Amir Ruzaimi Bin Kamarulzaman
* Course: JIE42903 – Evolutionary Computing
* Institution: Universiti Malaysia Kelantan (UMK)

