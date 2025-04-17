@echo off
echo Starting Flask backend...
cd backend
call venv\Scripts\activate
start python app.py

echo Starting React frontend...
cd ..\frontend
start npm start

echo PDF Diff Tool is running!
echo Frontend: http://localhost:3000
echo Backend: http://localhost:5000