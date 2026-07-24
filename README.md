# 🌊 AquaMind

## AI-Powered Marine Ecosystem Intelligence Platform for Real-Time Plankton Monitoring and Early Harmful Algal Bloom (HAB) Prediction


## 📖 About the Project

AquaMind is an AI-powered marine ecosystem intelligence platform that combines computer vision, IoT sensors, embedded systems, and machine learning to monitor aquatic ecosystems and predict Harmful Algal Blooms (HABs).

## 👥 Team Details

| Team Member | Role | Responsibilities |
|------------|------|------------------|
| Haripriya | AI/ML Research & Data Acquisition Contributor | Dataset research, acquisition, organization, documentation. |
| Aishwarya | Hardware Development Contributor | Hardware prototype, sensor integration, testing. |
| Krithik Nithin | Embedded Systems & PCB Design Contributor | Circuit schematics, PCB design, hardware integration. |
| Shashwat Raam | Frontend & Backend Development Contributor | Frontend research, UI development, backend support. |
| Sesu Roshan | Budget Analysis & Market Research Contributor | Budget sheet, market research, price comparison, pie chart analysis. |
| Sharika | AI/ML & System Integration Contributor | HAB Prediction Model (Model 2), dataset acquisition, preprocessing, model training & evaluation, ESP32 programming, frontend support, system integration. |

## 🎯 Problem Statement

Marine ecosystems are increasingly threatened by pollution, climate change, and Harmful Algal Blooms (HABs). Traditional monitoring methods rely on manual sampling and laboratory analysis, making continuous monitoring expensive and difficult.

## 💡 Our Solution

AquaMind integrates AI, IoT, environmental sensing, and embedded systems to automate plankton monitoring, assess water quality, and predict HAB events through an intelligent dashboard.

## ✨ Key Features

- Automated plankton analysis
- HAB prediction
- Real-time water quality monitoring
- ESP32-based sensing
- Interactive dashboard
- Historical analytics
- Environmental alerts

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| AI/ML | Python, PyTorch, OpenCV, Scikit-learn |
| Backend | FastAPI |
| Frontend | React, Tailwind CSS |
| Database | MongoDB |
| Hardware | ESP32 |
| Sensors | pH, Temperature, Turbidity, DO, TDS |

---
## System Architecture

![System Architecture](Downloads/arch.png)

### Description
The AquaMind platform integrates environmental sensors, an ESP32 microcontroller, backend services, AI/ML models, a database, and a web dashboard into a single monitoring pipeline.

**Workflow**

1. Water quality sensors collect environmental parameters.
2. ESP32 reads and transmits sensor data.
3. Backend API validates and stores incoming data.
4. AI models process plankton images and environmental parameters.
5. Prediction results are stored in the database.
6. Dashboard visualizes real-time and historical information.

---

## Project Workflow

![Project Workflow](assets/images/project_workflow.png)

```text
Water Sample
      ↓
Sensor Data Collection
      ↓
ESP32
      ↓
Backend API
      ↓
Database
      ↓
AI Models
      ↓
Prediction
      ↓
Dashboard
      ↓
Alerts
```

---

# 🤖 AI / ML Workflow

![AI Workflow](assets/images/ml_workflow.png)

### Model 1 – Plankton Classification

Dataset → Image Preprocessing → CNN Model → Species Classification → Dashboard

### Model 2 – HAB Prediction

Dataset Acquisition → Data Cleaning → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → HAB Risk Prediction

---

# 🧠 AI Model Overview

## Model 1 – Plankton Classification

- Input: Plankton Images
- Processing: Image preprocessing & deep learning
- Output: Species classification
its not over yet , but the dataset is taken from kaggle :
https://www.kaggle.com/datasets/muhammadsyaugishahab/whoi-plankton

## Model 2 – HAB Prediction

- Input: Water quality parameters
- Processing: Data preprocessing, feature engineering, model training
- Output: HAB risk prediction

we are still researching about the dataset to use for HAB prediction model , the links used are :
1.https://www.epa.gov/water-research/cyanobacteria-assessment-network
2.https://www.usgs.gov/data/harmonized-continuous-water-quality-data-support-modeling-harmful-algal-blooms-united-states
3.https://www.ncei.noaa.gov/products/harmful-algal-bloom-archived-data


---

# 🔌 Hardware Architecture


Components:

- ESP32
- Temperature Sensor
- pH Sensor
- Turbidity Sensor
- Dissolved Oxygen Sensor
- TDS Sensor
- Power Supply
- Wi-Fi Communication

---

# 🔧 Wiring Diagram

![Wiring Diagram](assets/images/wiring_diagram.png)

Include the complete ESP32 wiring showing all sensor connections.

---

# 📸 Hardware Prototype

![Hardware Prototype](assets/images/hardware_setup.jpg)

Will Add photographs of:
- Complete prototype
- Top view
- Side view
- Sensor placement

---

# 💻 Dashboard Preview
will display the dashboard done with synthetic data :
- Home Dashboard
- secure sign in and sign up 
- Live Sensor Values
- Graphs



# 🗄️ Database Design

- sensor_data
- plankton_predictions
- hab_predictions
- users
- alerts

---

# 📁 Project Folder Structure

```text
AquaMind/
│
├── backend/
├── frontend/
├── hardware/
├── ai_models/
│   ├── plankton_classification/
│   └── hab_prediction/
├── datasets/
├── docs/
├── assets/
│   └── images/
├── README.md
└── requirements.txt
```

---

# 🔗 API Workflow

```text
ESP32
   │
POST /sensor-data
   │
Backend
   │
MongoDB
   │
POST /predict
   │
AI Models
   │
Prediction Response
   │
Dashboard
```
