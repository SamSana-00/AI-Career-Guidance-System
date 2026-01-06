# AI-Powered Career Guidance System

A real-time AI-powered web application that provides personalized career guidance by analyzing psychometric traits, resumes, and skill gaps.

---

## 🚀 Overview

Choosing the right career path is often confusing due to lack of personalized guidance.  
This project aims to solve that problem by using AI-driven insights to help users understand their strengths, interests, and skill gaps, and receive actionable career recommendations.

The system analyzes:
- Psychometric responses
- Resume data (PDF/DOCX)
- User-selected interests and skills

Based on this, it generates:
- Suitable career paths
- Skill-gap analysis
- Interview preparation guidance

---

## 🧠 Key Features

- Resume parsing (PDF & DOCX formats)
- Psychometric evaluation
- Skill-gap analysis
- AI-generated career recommendations
- Interview preparation insights
- Secure environment variable handling
- Modular Flask-based architecture

---

## 🛠️ Tech Stack

**Backend**
- Python
- Flask

**AI & Data Processing**
- Pandas
- NumPy
- PyPDF2
- python-docx
- LLM integration via Groq API

**Frontend**
- HTML
- CSS

---

## 📁 Project Structure

├── static/
│ └── css/
│ └── style.css
├── templates/
│ ├── index.html
│ ├── login.html
│ ├── signup.html
│ ├── dashboard.html
│ ├── career_guidance.html
│ ├── career_comparison.html
│ ├── skill_gap.html
│ └── interview_prep.html
├── personal_career.py
├── requirements.txt
├── users.json
├── README.md
└── .gitignore

