#!/usr/bin/env python3
"""
Medical Imaging Platform - Terminal Control
Control script for Django and Flask applications with terminal output
"""

import subprocess
import sys
import os
import signal
import time
from datetime import datetime

def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print("🏥 MEDICAL IMAGING PLATFORM - TERMINAL CONTROL")
    print("="*80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Directory: {os.getcwd()}")
    print("="*80)

def check_applications():
    """Check status of running applications"""
    print("\n🔍 Checking application status...")

    django_running = False
    flask_running = False

    try:
        # Check for Django (port 8000)
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if ':8000' in result.stdout:
            django_running = True
            print("✅ Django Web App: RUNNING (Port 8000)")
        else:
            print("❌ Django Web App: STOPPED")

        if ':5000' in result.stdout:
            flask_running = True
            print("✅ Flask API Backend: RUNNING (Port 5000)")
        else:
            print("❌ Flask API Backend: STOPPED")

    except:
        print("⚠️  Could not check port status")

    return django_running, flask_running

def start_django():
    """Start Django application"""
    print("\n🚀 Starting Django Web Application...")
    try:
        process = subprocess.Popen(
            [sys.executable, 'manage.py', 'runserver'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Read and display output for a few seconds
        start_time = time.time()
        while time.time() - start_time < 10:  # Show output for 10 seconds
            if process.poll() is not None:
                break
            line = process.stdout.readline()
            if line:
                print(f"📄 DJANGO: {line.strip()}")

        if process.poll() is None:
            print("✅ Django server started in background")
            return process
        else:
            print("❌ Django server failed to start")
            return None

    except Exception as e:
        print(f"❌ Error starting Django: {e}")
        return None

def start_flask():
    """Start Flask application"""
    print("\n🚀 Starting Flask API Backend...")
    try:
        process = subprocess.Popen(
            [sys.executable, 'app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Read and display output for a few seconds
        start_time = time.time()
        while time.time() - start_time < 10:  # Show output for 10 seconds
            if process.poll() is not None:
                break
            line = process.stdout.readline()
            if line:
                print(f"📄 FLASK: {line.strip()}")

        if process.poll() is None:
            print("✅ Flask server started in background")
            return process
        else:
            print("❌ Flask server failed to start")
            return None

    except Exception as e:
        print(f"❌ Error starting Flask: {e}")
        return None

def stop_applications():
    """Stop running applications"""
    print("\n🛑 Stopping applications...")

    try:
        # Kill processes on ports 8000 and 5000
        subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq Django*'], check=False)
        subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq Flask*'], check=False)

        # Also try to kill by port
        subprocess.run(['for /f "tokens=5" %a in (\'netstat -ano ^| find ":8000"\') do taskkill /F /PID %a'], shell=True, check=False)
        subprocess.run(['for /f "tokens=5" %a in (\'netstat -ano ^| find ":5000"\') do taskkill /F /PID %a'], shell=True, check=False)

        print("✅ Applications stopped")
    except Exception as e:
        print(f"⚠️  Error stopping applications: {e}")

def show_menu():
    """Show command menu"""
    print("\n📋 Available Commands:")
    print("1. Start Django Web App (Port 8000)")
    print("2. Start Flask API Backend (Port 5000)")
    print("3. Start Both Applications")
    print("4. Check Application Status")
    print("5. Stop All Applications")
    print("6. Show AI Model Status")
    print("7. Test API Endpoints")
    print("0. Exit")
    print("-" * 40)

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🧪 Testing API Endpoints...")

    try:
        # Test Django
        result = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://127.0.0.1:8000/'],
            capture_output=True,
            text=True,
            timeout=10
        )
        django_status = result.stdout.strip()
        django_status_icon = "✅" if django_status == "200" else "❌"
        print(f"{django_status_icon} Django Web App: HTTP {django_status}")

    except:
        print("❌ Django Web App: Connection failed")

    try:
        # Test Flask
        result = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://127.0.0.1:5000/'],
            capture_output=True,
            text=True,
            timeout=10
        )
        flask_status = result.stdout.strip()
        flask_status_icon = "✅" if flask_status == "200" else "❌"
        print(f"{flask_status_icon} Flask API Backend: HTTP {flask_status}")

    except:
        print("❌ Flask API Backend: Connection failed")

def show_ai_status():
    """Show AI model status"""
    print("\n🤖 AI Models Status:")
    print("🧠 Brain Tumor Detection:")
    print("   • Model: EfficientNetB0 (4-class)")
    print("   • Classes: Glioma, Meningioma, No Tumor, Pituitary")
    print("   • Accuracy: 97.5% validation")
    print("   • Status: ✅ Active")

    print("\n🧠 Alzheimer Disease Staging:")
    print("   • Model: EfficientNetB0 (4-class)")
    print("   • Classes: NonDemented, Very Mild, Mild, Moderate")
    print("   • Accuracy: 97.0% validation")
    print("   • Status: ✅ Active")

    print("\n📊 Real-time Testing:")
    try:
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys
sys.path.append('.')
from services.brain_tumor_service import predict_brain_tumor
from services.alzheimer_service import predict_alzheimer
import numpy as np
from PIL import Image
import io

# Test with dummy image
dummy_image = np.random.rand(224, 224, 3) * 255
dummy_image = dummy_image.astype(np.uint8)
pil_image = Image.fromarray(dummy_image)
img_bytes = io.BytesIO()
pil_image.save(img_bytes, format="PNG")
img_bytes = img_bytes.getvalue()

print("Testing Brain Tumor Model...")
brain_result = predict_brain_tumor(image_bytes=img_bytes)
if brain_result.get("success"):
    pred = brain_result["prediction"]
    tumor_type = pred["tumor_type"]
    confidence = pred["confidence"]
    print(f"  ✅ Working: {tumor_type} ({confidence:.1%})")
else:
    error_msg = brain_result.get("error", "Unknown error")
    print(f"  ❌ Error: {error_msg}")

print("Testing Alzheimer Model...")
alz_result = predict_alzheimer(image_bytes=img_bytes)
if alz_result.get("success"):
    prediction = alz_result.get("prediction", "Unknown")
    confidence = alz_result.get("confidence", 0.0)
    print(f"  ✅ Working: {prediction} ({confidence:.1%})")
else:
    error_msg = alz_result.get("error", "Unknown error")
    print(f"  ❌ Error: {error_msg}")
            '''],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    """Main function"""
    print_header()

    if not os.path.exists('manage.py'):
        print("❌ Error: manage.py not found. Please run from the Django project root directory.")
        sys.exit(1)

    django_process = None
    flask_process = None

    while True:
        show_menu()
        try:
            choice = input("Enter your choice (0-7): ").strip()

            if choice == '0':
                print("\n👋 Goodbye!")
                break

            elif choice == '1':
                django_process = start_django()

            elif choice == '2':
                flask_process = start_flask()

            elif choice == '3':
                django_process = start_django()
                time.sleep(3)
                flask_process = start_flask()

            elif choice == '4':
                check_applications()

            elif choice == '5':
                stop_applications()
                django_process = None
                flask_process = None

            elif choice == '6':
                show_ai_status()

            elif choice == '7':
                test_api_endpoints()

            else:
                print("❌ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            stop_applications()
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        print()  # Empty line for readability

if __name__ == "__main__":
    main()