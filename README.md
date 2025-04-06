# 🩺 Smart Medical Cart

A graduation project that provides an AI-powered smart medical cart to assist doctors and nurses in hospitals. The system includes patient monitoring, AI diagnostics, real-time alerts, and secure medical notes management.

## 🚀 Features

- Real-time patient vitals monitoring
- AI diagnosis assistant
- Medical notes (CRUD system)
- User authentication & role-based access
- Admin dashboard
- MySQL database integration
- Built with Angular + Flask + MySQL


## 📁 Project Structur

smart-medical-cart/
│
├── backend/              # Python Flask API + AI Models
│   ├── app.py
│   └── model.pkl
│
├── frontend/             # Angular Frontend
│   ├── src/
│   └── ...
│
├── database/             # MySQL Scripts
│   └── schema.sql
│
├── docs/                 # Documentation, Diagrams, Presentations
│
├── README.md
└── .gitignore
  
## ⚙️ Tech Stack

- Frontend: Angular
- Backend: Python + Flask
- Database: MySQL
- AI: Scikit-learn
- Hosting: AWS IoT / Firebase (TBD)

## 📌 Usage

```bash
# Run backend
cd backend
python app.py

# Run frontend
cd frontend
ng serve
