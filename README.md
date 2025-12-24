# 💓 Fitbit HRV Analyzer

A professional-grade web application that connects to the Fitbit API to fetch second-by-second heart rate data, calculates Heart Rate Variability (HRV) metrics, and provides AI-powered health feedback using Google Gemini.

---

## 🚀 Features
- **OAuth2 Integration**: Securely connects to Fitbit accounts.  
- **Persistent Sessions**: Uses SQLite to store refresh tokens for months of access.  
- **HRV Analytics**: Calculates RMSSD, SDNN, and LF/HF ratios using SciPy.  
- **AI Feedback**: Integrated with Gemini 2.0 Flash for personalized recovery insights.  
- **Secure**: Uses `.env` files to protect API credentials.  

---

## 🛠️ Setup Instructions

**Clone the repository and install dependencies:**

```bash
pip install -r requirements.txt
