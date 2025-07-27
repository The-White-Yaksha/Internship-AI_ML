# 🚀 Developers Hub Internship Portfolio

Welcome to my **Machine Learning & NLP Internship Project Repository** from Developers Hub Corporation!  
This collection of projects reflects my transition from classical data analysis and ML to cutting-edge **transformer-based NLP**, **LangGraph orchestration**, and **prompt engineering**.

> 🎓 Status: Software Engineering Undergrad (5th Semester)  
> 📅 Internship Duration: Summer 2025  
> 📌 Focus Areas: EDA, Predictive Modeling, Transformers, Prompt Engineering, LangGraph, Pinecone, RAG

---

## 📚 Table of Contents

- [🧠 Overview](#-overview)
- [📁 Project Structure](#-project-structure)
- [✅ Core Internship Tasks](#-core-internship-tasks)
- [🚀 Advanced NLP & LangGraph Tasks](#-advanced-nlp--langgraph-tasks)
- [📊 Skills & Tools Used](#-skills--tools-used)
- [📌 Learnings & Highlights](#-learnings--highlights)
- [📥 Installation](#-installation)
- [🙋 About Me](#-about-me)

---

## 🧠 Overview

This internship allowed me to explore real-world business datasets, build predictive models, fine-tune transformer architectures, and orchestrate prompt refinement pipelines using **LangGraph** and **RAG**.

I worked on:
- Exploratory Data Analysis
- Logistic & Linear Regression
- BERT Fine-Tuning
- Zero-shot vs Few-shot Prompting
- Memory-integrated Prompt Systems (LangGraph + Pinecone)
- Evaluation Metrics & Human-in-the-Loop logic

---

## 📁 Project Structure

```

.
├── Internship Task 1.ipynb             # EDA on Penguins
├── Internship Task 3.ipynb             # Heart Disease Prediction
├── Internship Task 6.ipynb             # House Price Regression
├── Advance Internship Task 1/          # BERT Headline Classifier
├── Advance Internship Task 4/          # LangGraph Prompt Refiner
├── Advanced Internship Task 5/         # Ticket Intent Tagging
└── README.md                           # This file

```

---

## ✅ Core Internship Tasks

### 🐧 Internship Task 1 – Penguin Dataset EDA
> Dataset: Palmer Penguins (`sns.load_dataset('penguins')`)

🔍 **Goal:** Explore species-based visual trends using Seaborn  
📊 **Visuals Used:** Pairplot, Violin, Scatter, Count  
🧼 **Cleaning:** Dropped nulls, verified feature types  
🎯 **Insight:** Species and islands influence physical traits  
📂 **Libraries:** `pandas`, `matplotlib`, `seaborn`

---

### ❤️ Internship Task 3 – Heart Disease Classifier
> Dataset: `heart-disease.csv` (Binary classification)

🔍 **Goal:** Predict heart disease presence (Yes/No)  
⚙️ **Model:** Logistic Regression  
🧼 **Cleaning:** Removed `sex` column to avoid bias  
📈 **Accuracy:** ~90%  
📂 **Libraries:** `scikit-learn`, `matplotlib`, `pandas`

---

### 🏠 Internship Task 6 – Seattle House Price Regression
> Dataset: `train.csv`, `test.csv` from Kaggle

🔍 **Goal:** Predict house prices  
🧼 **Preprocessing:** Removed nulls, outliers, irrelevant features  
⚙️ **Models:**  
- Linear Regression (underfit)  
- Polynomial Regression (degree=2)  
- Ridge & Lasso (regularized)  

📊 **Evaluation:** R², MAE, RMSE  
📂 **Libraries:** `sklearn`, `numpy`, `pandas`

---

## 🚀 Advanced NLP & LangGraph Tasks

### 📰 Advanced Task 01 – BERT News Headline Classifier
> Dataset: AG News | Model: `bert-base-uncased`

🧠 **Goal:** Classify news headlines by category  
🧪 **Fine-Tuning:** Hugging Face `Trainer`, 3 epochs  
📈 **Metrics:**  
- Accuracy: **97.54%**  
- Eval Loss: 0.1323  
- Runtime: 88.9 sec  
📦 **Features:** CLI inference script (`predict.py`)  
📂 **Libraries:** `transformers`, `datasets`, `sklearn`

---

### 🧠 Advanced Task 04 – LangGraph Prompt Refiner (RAG + Pinecone)
> Uses: LangGraph, OpenRouter API, Pinecone, Local Text Knowledge

📌 **Goal:** Build a memory-integrated, multi-phase LLM prompt refiner  
🧱 **Architecture:**  
```

User Input → Preprocess → Embed → Retrieve → Refine Prompt →
Human Editor → Evaluate → Store in Memory (Pinecone)

````

💡 **Highlights:**
- Semantic memory with Pinecone embeddings  
- Human-in-the-loop checkpoint  
- Auto-evaluation (pass/fail routing)  
- Custom RAG injection via `hassnain_info.txt`

📂 **Libraries:** `langgraph`, `openrouter`, `pinecone`, `sentence-transformers`

---

### 🏷️ Advanced Task 05 – Support Ticket Tagging Pipeline

📌 **Goal:** Compare three approaches to intent tagging for support tickets  
📋 **Dataset:** 25 real-world ticket samples  
🧪 **Methods Compared:**
1. Zero-Shot Prompting (GPT-4o)
2. Fine-Tuned `distilbert-base-uncased`
3. Few-Shot Prompt Engineering

📊 **Results:**

| Method      | Accuracy |
|-------------|----------|
| Zero-Shot   | 4%       |
| Fine-Tuned  | 0% ❌     |
| Few-Shot    | **28% ✅** |

💡 Few-shot prompting outperformed others significantly.

📂 **Columns:** `instruction`, `intent`, `few_shot_tags`, `*_match`  
📂 **Libraries:** `transformers`, `sklearn`, `pandas`

---

## 📊 Skills & Tools Used

| Category            | Tools & Frameworks |
|---------------------|--------------------|
| **Languages**       | Python             |
| **ML Models**       | Logistic, Linear, Polynomial, Ridge, Lasso |
| **NLP**             | BERT, DistilBERT, GPT-4o |
| **Frameworks**      | scikit-learn, Transformers, LangGraph |
| **Orchestration**   | LangGraph, Pinecone, OpenRouter |
| **Prompting**       | Zero-shot, Few-shot, Human-in-the-loop |
| **Visualization**   | Seaborn, Matplotlib |
| **Data Handling**   | Pandas, NumPy |

---

## 📌 Learnings & Highlights

- ✅ Transitioned from traditional ML to **modern NLP pipelines**
- 🚀 Understood when to **fine-tune models vs use prompts**
- 🧠 Applied **LangGraph** to build modular LLM workflows
- 🧩 Built real-world pipelines with **evaluation + memory sync**
- 📈 Reinforced strong command on data cleaning, transformation & metrics

---

## 📥 Installation

### 🔹 Clone the Repo
```bash
git clone https://github.com/yourusername/devhub-internship.git
cd devhub-internship
````

### 🔹 For Advanced Tasks

```bash
cd Advance Internship Task 1
pip install -r requirements.txt
```

---

## 🙋 About Me

I'm a passionate software engineering student with a deep love for **data, design, and systems**.
Through this internship, I explored how raw data can be shaped into meaningful insights and intelligent systems.


---

### ⭐️ If you liked this project, don't forget to leave a star on GitHub!

```


