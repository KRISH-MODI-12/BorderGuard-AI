FOR SETUP- CHECK
SETUP.md file
# BorderGuard-AI
AI-powered CCTV and border surveillance demo using computer vision for person and vehicle detection, tracking, face and number-plate verification, unattended-object detection, and security alerts. Built with Python, YOLO, OpenCV, FastAPI, and AI-based video analytics.
# 🛡️ BorderGuard AI

### AI-Powered Intelligent CCTV & Border Surveillance Video Analytics

BorderGuard AI is an AI-based video analytics demonstration designed to show how existing CCTV infrastructure can be enhanced with intelligent computer-vision capabilities.

The system analyzes CCTV/video footage and assists operators by detecting people and vehicles, tracking individuals, verifying authorized and unauthorized subjects, detecting unattended objects, and generating security alerts.

> ⚠️ **Project Status:** Academic / Hackathon Prototype  
> This project is intended for educational, research, and demonstration purposes.

---

## 🚀 Overview

Traditional CCTV systems primarily provide video footage that requires security personnel to continuously monitor multiple camera feeds.

BorderGuard AI demonstrates an additional AI analytics layer that can process video and highlight events that may require operator attention.

### Basic Workflow

```text
CCTV Camera / Uploaded Video
            ↓
     Video Processing
            ↓
       AI Detection
            ↓
    Object Tracking
            ↓
 ┌──────────┼──────────┐
 ↓          ↓          ↓
Person    Vehicle    Objects
 ↓          ↓          ↓
Face      Number     Object
Verify    Plate      Monitoring
 ↓          ↓          ↓
 └──────────┼──────────┘
            ↓
     Security Rules
            ↓
     Alerts & Results
            ↓
     Command Dashboard
