# Menu Passport Backend

A FastAPI-based backend service that translates restaurant menus with the option of using a traditional pipeline or 
using agentic AI. Upload a menu image, and get back translated dishes with descriptions, converted prices, and visual 
search results.


## Installation

### Clone the repository

```bash
git clone <your-repo-url>
cd menu-passport-backend
```

### Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Set up environment variables

```bash
# Copy example env file and edit with your own keys
cp .env.example .env
```

## Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit: http://localhost:8000/docs
