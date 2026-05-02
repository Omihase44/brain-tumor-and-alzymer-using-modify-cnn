# Medical Imaging Platform - Terminal Monitoring

## Overview
This medical imaging platform now includes comprehensive terminal output and monitoring capabilities for both Django and Flask applications. The system provides real-time logging, status monitoring, and control functions through an interactive terminal interface.

## Applications
- **Django Web App** (Port 8000): Frontend interface for medical image analysis
- **Flask API Backend** (Port 5000): REST API with Socket.IO for real-time features

## AI Models
- **Brain Tumor Detection**: 97.5% accuracy (4-class: Glioma, Meningioma, No Tumor, Pituitary)
- **Alzheimer Disease Staging**: 97.0% accuracy (4-class: NonDemented, Very Mild, Mild, Moderate)

## Terminal Control Features

### Interactive Control Script (`terminal_control.py`)
Run with: `python terminal_control.py`

**Available Commands:**
1. **Start Django Web App** - Launch Django server on port 8000
2. **Start Flask API Backend** - Launch Flask server on port 5000
3. **Start Both Applications** - Launch both servers simultaneously
4. **Check Application Status** - Verify if applications are running
5. **Stop All Applications** - Terminate running applications
6. **Show AI Model Status** - Display model information and test functionality
7. **Test API Endpoints** - Verify API connectivity and response codes

### Real-time Monitor (`terminal_monitor.py`)
Run with: `python terminal_monitor.py`

**Features:**
- Starts both applications automatically
- Displays real-time logs from both Django and Flask
- Color-coded output (Green for Django, Blue for Flask)
- Shows timestamps for all log entries
- Monitors application health and reports failures

### Quick Start Batch File (`start_platform.bat`)
Run with: `start_platform.bat`

**Features:**
- One-click startup for the entire platform
- Displays access URLs and status information
- Interactive control menu

## Terminal Output Examples

### Application Status Check
```
🔍 Checking application status...
✅ Django Web App: RUNNING (Port 8000)
✅ Flask API Backend: RUNNING (Port 5000)
```

### AI Model Testing
```
🤖 AI Models Status:
🧠 Brain Tumor Detection:
   • Model: EfficientNetB0 (4-class)
   • Accuracy: 97.5% validation
   • Status: ✅ Active

🧠 Alzheimer Disease Staging:
   • Accuracy: 97.0% validation
   • Status: ✅ Active
```

### API Endpoint Testing
```
🧪 Testing API Endpoints...
✅ Django Web App: HTTP 200
✅ Flask API Backend: HTTP 200
```

### Real-time Logs
```
[15:12:30] DJANGO: Starting development server at http://127.0.0.1:8000/
[15:12:32] FLASK:  * Running on http://127.0.0.1:5000/
[15:12:35] DJANGO: Quit the server with CTRL-BREAK.
```

## Usage Instructions

### Starting the Platform
1. **Interactive Mode**: `python terminal_control.py`
2. **Quick Start**: Double-click `start_platform.bat`
3. **Real-time Monitoring**: `python terminal_monitor.py`

### Monitoring Applications
- Use option 4 in the control script to check status
- Use option 6 to verify AI model functionality
- Use option 7 to test API connectivity
- Real-time logs show all application activity

### Troubleshooting
- If applications fail to start, check for port conflicts
- Use option 5 to stop all applications before restarting
- Check the real-time monitor for detailed error messages
- Ensure all Python dependencies are installed (`pip install -r requirements.txt`)

## Access URLs
- **Django Frontend**: http://127.0.0.1:8000/
- **Flask Backend**: http://127.0.0.1:5000/

## System Requirements
- Python 3.8+
- Windows PowerShell (for process monitoring)
- All dependencies from `requirements.txt`

## Files Created
- `terminal_control.py` - Interactive control and monitoring script
- `terminal_monitor.py` - Real-time logging monitor
- `start_platform.bat` - Quick start batch file