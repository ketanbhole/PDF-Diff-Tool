A powerful web application that allows users to compare two PDF documents using LLaMA 2-based LLM intelligence. It visually highlights differences with color-coded changes, supports large documents, and runs fully locally—no external API required.

✨ Features
📄 Upload two PDF files for intelligent comparison

🤖 Powered by LLaMA 2 7B Chat running locally via Hugging Face Transformers

🔍 Semantic diffing with color-coded highlights:

✅ Added content (green)

❌ Removed content (red)

✏️ Modified content (yellow)

📊 Summary of all detected changes

📱 Responsive interface for desktop and tablet

📤 Export the comparison results as a downloadable PDF

🧠 Supports large PDF documents using chunked processing

🚀 Live Demo
👉 Check out the live demo

Note: The demo may use fallback diff logic (difflib) due to limitations of running LLaMA 2 in-browser. Full LLM-powered comparison is available in the local version.

🛠 Tech Stack
Frontend
React.js

TailwindCSS

pdf.js (for rendering and extracting content)

Backend
Flask (Python)

Local LLaMA 2 7B Chat (Quantized, running via Hugging Face Transformers)

Custom logic for diffing, chunked PDF comparison

Deployment
Dockerized full-stack app:

Frontend: Vercel (for public demo)

Backend: Local via Docker or Railway (for remote hosting)

GPU acceleration for LLMs (if available)

🧑‍💻 Local Development
Prerequisites
Python 3.12

Node.js 18+

Docker (optional, for containerized setup)

A machine with sufficient memory and optionally a GPU

LLaMA 2 7B Chat model downloaded locally

Clone and Setup

git clone https://github.com/ketanbhole/PDF-Diff-Tool
.git
cd pdf-diff-tool
Backend Setup
Navigate to backend:


cd backend
(Optional) Create and activate a virtual environment:


python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:


pip install -r requirements.txt
Ensure LLaMA 2 model is downloaded and configured:

Place model files under: backend/models/llama-2-7b-chat

Use 4-bit quantization if needed for performance

Run the server:


python app.py
Frontend Setup
Navigate to frontend:


cd ../frontend
Install dependencies:


npm install
Start the frontend:


npm run dev
Docker Setup (Full Stack)
docker-compose up --build
Configure Docker to use your preferred drive (e.g., D:\pdf\docker) if needed.


🤝 Contributing
Pull requests are welcome! If you find a bug or want to suggest an enhancement, feel free to open an issue.

📄 License
This project is licensed under the MIT License.

