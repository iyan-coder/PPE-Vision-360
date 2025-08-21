# SafeSight AI Assistant — README

**Project tagline:** *From boots to helmets — AI-powered PPE compliance for safer construction worksites.*

---

### Overview (Problem → MVP solution)

**Problem:**  
In construction and industrial environments, workers often forget or misuse PPE (helmets, boots, gloves, vests). Current inspection workflows are **manual, slow, and error-prone**, increasing risk of accidents and compliance issues.

**Solution / MVP:**  
SafeSight AI Assistant is an **image-based PPE compliance system**. Users upload images of workers or site conditions, and the system:  
- Detects PPE items using **YOLOv8**.  
- Classifies worn vs. not-worn PPE.  
- Generates human-readable compliance reasoning using **BERT + NER**.  
- Provides a **retrieval-augmented chatbot (FAISS + LLM)** for PPE-specific questions.

> Note: This MVP currently **only works with images**, not live camera feeds. Accuracy is limited due to **small dataset and GPU constraints**. Adding more labeled images will make the system **more robust and reliable**. Future deployment to embedded systems (Raspberry Pi / edge devices) is planned.

---

### Single-sentence MVP description

An image-based PPE compliance assistant that detects PPE items in photos, classifies presence/absence, and provides explainable compliance reasoning and corrective recommendations.

---

### What we built (current features)

- **YOLOv8 object detection** for helmets, boots, gloves, and vests.  
- **Segmentation & classification** for partial or incorrectly worn PPE.  
- **BERT-based OSHA-style question classifier** for compliance reasoning.  
- **NER module** to extract PPE-related entities from reports.  
- **LLM-powered chatbot (RAG)** that answers only PPE compliance questions using FAISS retrieval.  
- **Streamlit dashboard** to visualize images, detections, explanations, and compliance reports.  

> Note: Future improvements include adding more PPE classes, expanding chatbot knowledge, and integrating real-time video monitoring.

---

### Architecture (Mermaid diagrams)

#### System overview (flowchart)


```mermaid
flowchart LR
    A[User uploads Image] --> B[Preprocessing - resize & augment]
    B --> C[YOLOv8 Detection]
    C --> D{Detected PPE Items}
    D -->|Helmet| E[Helmet classifier & segmentation]
    D -->|Boots| F[Boots classifier]
    D -->|Gloves| G[Gloves classifier]
    D -->|Vest| H[Vest classifier]
    E --> I[Compliance rules & thresholds]
    F --> I
    G --> I
    H --> I
    I --> J[Compliance reasoning - NER & BERT]
    J --> K[LLM Assistant / Chatbot]
    K --> L[Dashboard / Exportable Report - CSV/PDF]

```
---

### Sequence: Image → Decision → Report

```mermaid
sequenceDiagram
    participant U as User
    participant I as Image
    participant M as Model (YOLO + Classifiers)
    participant R as Reasoner (BERT + NER)
    participant L as LLM
    participant D as Dashboard

    U->>I: Upload image
    I->>M: Detect PPE items
    M->>R: Classified & verified PPE
    R->>L: Generate compliance reasoning
    L->>D: Return explanation & chatbot response
    D->>U: Display results + download options
 ```   
 ---

### Chatbot RAG Explanation (Hybrid LLM + FAISS)
```mermaid
flowchart LR
    A[User Query / PPE Question] --> B[FAISS Vector Search]
    B --> C{Relevant PPE Context Found?}
    C -->|Yes| D[Retrieve Context & Metadata]
    C -->|No| E[Return Cannot Answer Outside PPE Scope]
    D --> F[LLM Generates Response Using Retrieved Context]
    F --> G[Provide Human-Friendly Answer / Recommendation]
    G --> H[Dashboard / Chat Interface]
    H --> A[User Sees Answer]

---
### Quickstart — How to Run Locally

#### Prerequisites
- Python 3.9+
- pip
- Virtual environment (venv / conda)
- (Optional) GPU + CUDA for faster YOLO training

----
#### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---
#### Prepare Data

- Place labeled images in `data/` (YOLO format).  
- Include `osha_qna.csv` for BERT + FAISS compliance queries.
---
#### Run Inference / Dashboard

```bash
# Detect PPE in images
python run_inference.py --weights runs/best.pt --source demo_images/

# Run Streamlit dashboard
streamlit run app/streamlit_app.py
```

#### Streamlit Cloud / Demo Link Placeholder

[![Launch SafeSight AI Assistant](https://img.shields.io/badge/Launch-Streamlit-blue)]([[insert your Streamlit link here]])

### Limitations

- **Dataset size:** Current dataset is small → limited model accuracy.  
- **Only images supported:** No live camera feed yet.  
- **FAISS + LLM chatbot:** Can only answer PPE-related questions.  
- **GPU requirements:** Training larger models may fail without GPU.  

> These limitations are expected for an MVP; scaling the dataset and improving compute resources will increase robustness.

---

### Roadmap & Future Improvements

- Expand dataset with more images from diverse construction sites.  
- Add more PPE classes and edge cases for YOLO.  
- Improve chatbot coverage with larger FAISS index and dynamic RAG.  
- Deploy on embedded devices for real-time monitoring.  
- Add role-based dashboard with audit logs and report exports.

---

### Screenshots / UI Preview

You can add images here to show your dashboard:

![Dashboard preview](docs/screenshots/PPE_Vision_360_1.png)
![Dashboard preview](docs/screenshots/PPE_Image_Compliance_1.png)
![Dashboard preview](docs/screenshots/PPE_OSHA_QA_Chatbot_1.png)
![Dashboard preview](docs/screenshots/PPE_BERT_Classifier_1.png)
![Dashboard preview](docs/screenshots/PPE_NER_Tagger_1.png)
![Dashboard preview](docs/screenshots/PPE_Compliance_Report_Generator_1.png)
![Dashboard preview](docs/screenshots/PPE_Chat_Assistant_1.png)
![Dashboard preview](docs/screenshots/PPE_Recommendation_System_1.png)

---

### Example Compliance Output
```json
{
  "image": "site_001.jpg",
  "detections": [{"class":"helmet","conf":0.92,"bbox":[x,y,w,h]}],
  "compliant": false,
  "reasons": ["No safety vest detected", "Helmet strap not fastened"],
  "recommendation": "Provide reflective vest and ensure helmet strap is secured."
}
```
### How to Contribute
- Fork the repository.
- Add or improve dataset, annotations, or model code.
- Submit a Pull Request (PR) including tests and an example image demonstrating your changes.

### Credits & Contact
- Project lead: You (SafeSight AI Assistant)
- Tools & models used: YOLOv8, BERT, FAISS, Streamlit, Python


