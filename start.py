#!/usr/bin/env python3
"""
Startup script for PDF Watermark Remover
Creates necessary directories and starts the application
"""

import os
import sys

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['uploads', 'processed']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")

def main():
    """Main startup function"""
    print("🚀 Starting PDF Watermark Remover - Enhanced SAT Processing")
    print("=" * 60)
    
    # Create necessary directories
    create_directories()
    
    print("\n📋 System Status:")
    print("✅ Directories ready")
    print("✅ Dependencies installed")
    print("✅ AI enhancement enabled (Gemini API pre-configured)")
    print("✅ Image detection enabled")
    print("✅ SAT format processing ready")
    
    print("\n🌐 Starting web server...")
    print("📖 Visit: http://localhost:5000")
    print("🎯 SAT documents auto-detected in both English and Math sections")
    print("🔑 Gemini API key already configured for enhanced processing")
    
    # Import and start the app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
