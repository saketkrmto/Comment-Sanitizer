# 🛡️ Comment Sanitizer

This application allows you to sanitize comments and text files by automatically detecting and filtering out bad words using a Machine Learning model.

It consists of two parts:
1. **A Python Backend** that runs the machine learning model.
2. **An HTML Frontend** that provides the user interface.

---

## 🚀 Step-by-Step Guide to Run the App

### Step 1: Start the Backend Server
The backend is a Python application that needs to be running in the background to process the text.

1. Open your **Command Prompt** (or PowerShell or Terminal).
2. Navigate to the `backend` folder inside this project by typing:
   ```bash
   cd backend
   ```
3. Run the following command to start the server:
   ```bash
   python -m uvicorn main:app --reload
   ```
4. **Wait a few seconds**. The terminal will say `Training ML model on a large dataset of bad words...`. Once you see `Application startup complete.`, the server is running successfully! 
5. *(Keep this terminal window open! If you close it, the backend will stop).*

### Step 2: Open the Website
1. Open your File Explorer.
2. Navigate to the project directory where you cloned or downloaded this repository.
3. Simply **double-click** the `index.html` file.
4. It will open in your default web browser (like Chrome or Edge).

### Step 3: Test the App!
1. Once the app opens in your browser, go to the **Home** tab.
2. Type a sentence in the box (try mixing normal words with some bad words).
3. Click the **Sanitize Comment** button.
4. The text will be sent to your Python server, sanitized by the ML model, and sent back to your screen with the bad words replaced by asterisks `***`!

---

## 🛠️ Requirements
The following Python packages must be installed for the backend to work:
- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pydantic`

*(Note: If you ever need to install them again, you can run `pip install -r backend/requirements.txt` from your terminal).*