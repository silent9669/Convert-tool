# Deploy to Railway

## Quick Deploy

1. **Fork/Clone** this repository
2. **Sign up** at [Railway.app](https://railway.app)
3. **Connect** your GitHub repository
4. **Deploy** automatically

## Railway Setup

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Health Check**: `/health` endpoint

## Environment Variables

Railway will automatically set:
- `PORT`: Server port
- `RAILWAY_ENVIRONMENT`: Deployment environment

## Access Your App

After deployment, Railway provides a public URL.
Your app will be available at: `https://your-app-name.railway.app`
