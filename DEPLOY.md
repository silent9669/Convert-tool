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
- **Port**: Automatically set by Railway

## Environment Variables

Railway will automatically set:
- `PORT`: Server port (required)
- `RAILWAY_ENVIRONMENT`: Set to "production"

## Railway Configuration Files

- **`railway.json`**: Main configuration
- **`Procfile`**: Process definition
- **`runtime.txt`**: Python version
- **`.railwayignore`**: Exclude unnecessary files

## Access Your App

After deployment, Railway provides a public URL.
Your app will be available at: `https://your-app-name.railway.app`

## Troubleshooting

### Build Issues
- Check Python version compatibility (3.11.9+)
- Ensure all dependencies are in requirements.txt
- Verify railway.json configuration

### Runtime Issues
- Check logs in Railway dashboard
- Verify health check endpoint `/health`
- Ensure PORT environment variable is set

### File Processing Issues
- Check file size limits (100MB max)
- Verify PDF file format
- Check temporary directory permissions
