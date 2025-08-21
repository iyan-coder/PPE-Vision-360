# SafeSight AI Assistant — README

**Project tagline:** *From boots to helmets — vision-powered PPE compliance for safer worksites.*

---

### ### Overview (Problem → MVP solution)

**Problem:**  
In many industrial and construction environments people forget, misuse, or lack awareness about required personal protective equipment (PPE). Existing inspection workflows are slow, manual, and error-prone. As an engineer familiar with AI, I wanted to create a lightweight, intelligent tool that helps supervisors and workers check PPE compliance in real-time.

**Our approach:**  
SafeSight AI Assistant is an MVP that automatically inspects **images only** (no live camera support yet) to detect core PPE items (helmet, boots, gloves, vest) and provides human-readable compliance reasoning. It uses a small dataset and runs on modest hardware, making it a practical starting point for real-world testing.

**Why this matters:**  
Even a simple assistant can reduce workplace injuries and help sites stay audit-ready. Over time, SafeSight AI can evolve into a more comprehensive AI Safety System for recommendations, auditing, and site-wide compliance.

---

### ### Single-sentence MVP description

An **image-based PPE compliance assistant** that detects helmets, boots, gloves, and vests, classifies their presence, and generates explainable compliance decisions and recommendations.

---

### ### Features

* **YOLOv8-based object detection** for PPE items.
* **Segmentation & classification** to handle partial occlusions and worn vs. not-worn items.
* **BERT classifier** for OSHA-style question mapping (from CSV Q&A).
* **NER & compliance reasoning** modules for human-readable explanations.
* **LLM-assisted chat & RAG** for generating natural language reasoning and suggestions.
* **Streamlit dashboard** for image viewing, model outputs, and report generation.

> Note: MVP currently handles **images only**. Future iterations may integrate embedded systems for live monitoring.

---

### ### Architecture (Mermaid diagrams)

#### System Overview

```mermaid
flowchart LR
    A[Input: Image] --> B[YOLOv8 Detection]
    B --> C{Detected Items}
    C -->|Helmet| D[Helmet segmentation & worn verification]
    C -->|Boots| E[Boots classifier]
    C -->|Gloves| F[Glove classifier]
    C -->|Vest| G[Vest classifier]
    D --> H[Compliance rules & thresholds]
    E --> H
    F --> H
    G --> H
    H --> I[Compliance reasoning (NER + BERT)]
    I --> J[LLM + RAG Assistant]
    J --> K[Streamlit Dashboard & Report Export]
