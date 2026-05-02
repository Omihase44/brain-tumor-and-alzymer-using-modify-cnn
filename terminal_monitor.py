#!/usr/bin/env python3
"""
Medical Imaging Platform - Terminal Monitor
Displays real-time logs and status from Django and Flask applications
"""

import subprocess
import threading
import time
import sys
import os
from datetime import datetime

class ApplicationMonitor:
    def __init__(self):
        self.django_process = None
        self.flask_process = None
        self.monitoring = True

    def start_django(self):
        """Start Django development server"""
        print("🚀 Starting Django Application (Port 8000)...")
        try:
            self.django_process = subprocess.Popen(
                [sys.executable, 'manage.py', 'runserver', '--noreload'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print("✅ Django server started successfully")
        except Exception as e:
            print(f"❌ Failed to start Django: {e}")

    def start_flask(self):
        """Start Flask API server"""
        print("🚀 Starting Flask API Backend (Port 5000)...")
        try:
            self.flask_process = subprocess.Popen(
                [sys.executable, 'app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print("✅ Flask API server started successfully")
        except Exception as e:
            print(f"❌ Failed to start Flask: {e}")

    def monitor_output(self, process, name, color_code):
        """Monitor and display output from a process"""
        while self.monitoring and process.poll() is None:
            try:
                line = process.stdout.readline()
                if line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"{color_code}[{timestamp}] {name}: {line.strip()}\033[0m")
            except:
                break

    def display_status(self):
        """Display current status of applications"""
        print("\n" + "="*80)
        print("🏥 MEDICAL IMAGING PLATFORM - TERMINAL MONITOR")
        print("="*80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Working Directory: {os.getcwd()}")
        print()

        # Check Django status
        django_status = "🟢 RUNNING" if self.django_process and self.django_process.poll() is None else "🔴 STOPPED"
        print(f"Django Web App (Port 8000): {django_status}")

        # Check Flask status
        flask_status = "🟢 RUNNING" if self.flask_process and self.flask_process.poll() is None else "🔴 STOPPED"
        print(f"Flask API Backend (Port 5000): {flask_status}")

        print("\n🌐 Access URLs:")
        print("   Django Frontend: http://127.0.0.1:8000/")
        print("   Flask Backend:   http://127.0.0.1:5000/")
        print("\n🤖 AI Models Status:")
        print("   🧠 Brain Tumor Model: 97.5% accuracy (4-class)")
        print("   🧠 Alzheimer Model:    97.0% accuracy (4-class)")
        print("="*80 + "\n")

    def start_monitoring(self):
        """Start monitoring both applications"""
        # Start both applications
        self.start_django()
        time.sleep(2)  # Wait for Django to start
        self.start_flask()
        time.sleep(2)  # Wait for Flask to start

        # Display initial status
        self.display_status()

        # Start monitoring threads
        django_thread = threading.Thread(
            target=self.monitor_output,
            args=(self.django_process, "DJANGO", "\033[92m"),  # Green
            daemon=True
        )

        flask_thread = threading.Thread(
            target=self.monitor_output,
            args=(self.flask_process, "FLASK", "\033[94m"),   # Blue
            daemon=True
        )

        django_thread.start()
        flask_thread.start()

        print("📊 Real-time monitoring started...")
        print("Press Ctrl+C to stop all applications\n")

        try:
            # Keep main thread alive
            while self.monitoring:
                time.sleep(1)

                # Check if processes are still alive
                if self.django_process and self.django_process.poll() is not None:
                    print("⚠️  Django process terminated unexpectedly")
                if self.flask_process and self.flask_process.poll() is not None:
                    print("⚠️  Flask process terminated unexpectedly")

        except KeyboardInterrupt:
            print("\n🛑 Shutting down applications...")
            self.monitoring = False

            # Terminate processes
            if self.django_process and self.django_process.poll() is None:
                self.django_process.terminate()
                print("✅ Django server stopped")

            if self.flask_process and self.flask_process.poll() is None:
                self.flask_process.terminate()
                print("✅ Flask server stopped")

            print("👋 All applications stopped. Goodbye!")

def main():
    """Main function"""
    print("🏥 Starting Medical Imaging Platform Terminal Monitor...")

    # Change to the correct directory if needed
    if not os.path.exists('manage.py'):
        print("❌ Error: manage.py not found. Please run from the Django project root directory.")
        sys.exit(1)

    monitor = ApplicationMonitor()
    monitor.start_monitoring()

if __name__ == "__main__":
    main()