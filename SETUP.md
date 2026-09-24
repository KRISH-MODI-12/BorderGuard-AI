# ⚙️ Installation & Required Tools

BorderGuard AI is a Python-based AI video analytics project. The following tools and libraries are used to run the demonstration.

> **Recommended Python:** Python 3.12
> Using a virtual environment is recommended so that project dependencies remain isolated from other Python projects. Python's official documentation recommends `venv` for creating isolated environments. [Python virtual environments documentation](https://docs.python.org/3.12/tutorial/venv.html?utm_source=chatgpt.com)

---

## 🧰 Required Tools

| Tool                   | Purpose                              |
| ---------------------- | ------------------------------------ |
| Python 3.12            | Main programming/runtime environment |
| pip                    | Python package installation          |
| Git                    | Clone and manage the project         |
| FastAPI                | Backend API/server                   |
| Uvicorn                | Runs the FastAPI application         |
| Ultralytics YOLO       | Object detection and tracking        |
| OpenCV                 | Video and image processing           |
| NumPy                  | Numerical/image-data processing      |
| YuNet                  | Face detection                       |
| SFace                  | Face recognition/verification        |
| OCR / Plate Processing | Number-plate analysis                |
| Modern Web Browser     | Access the dashboard                 |

---

# 1️⃣ Install Python

Install **Python 3.12** on your computer.

After installation, open Command Prompt or PowerShell and check:

```bash
python --version
```

Expected output:

```text
Python 3.12.x
```

Also check pip:

```bash
python -m pip --version
```

If `python` is not recognized on Windows, make sure Python was added to PATH during installation.

---

# 2️⃣ Install Git

Git is used to download the project from GitHub.

Check whether Git is already installed:

```bash
git --version
```

Then clone the project:

```bash
git clone https://github.com/KRISH-MODI-12/BorderGuard-AI.git
```

Move into the project:

```bash
cd BorderGuard-AI
```

---

# 3️⃣ Create a Python Virtual Environment

Create an isolated environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

After activation, your terminal should show something similar to:

```text
(venv)
```

Verify the Python interpreter:

```bash
python --version
```

Deactivate the environment later with:

```bash
deactivate
```

---

# 4️⃣ Upgrade pip

Inside the virtual environment:

```bash
python -m pip install --upgrade pip setuptools wheel
```

This helps ensure the package installer and build tools are up to date.

---

# 5️⃣ Install Project Dependencies

If this repository contains `requirements.txt`, install the project's dependencies with:

```bash
python -m pip install -r requirements.txt
```

This is the **recommended method** because the project dependencies are kept together in one file.

---

# 6️⃣ Main Python Packages

The project uses several Python packages.

## FastAPI

FastAPI provides the backend API/application layer.

Install:

```bash
python -m pip install "fastapi[standard]"
```

FastAPI also provides official documentation for installation and running applications. [FastAPI documentation](https://fastapi.tiangolo.com/tutorial/?utm_source=chatgpt.com)

---

## Uvicorn

Uvicorn runs the FastAPI application.

Install:

```bash
python -m pip install uvicorn
```

Run the backend with:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

---

## Ultralytics YOLO

Ultralytics provides the YOLO framework used for object detection and tracking.

Install:

```bash
python -m pip install -U ultralytics
```

Check the installation:

```bash
yolo checks
```

Ultralytics officially supports installation through pip and documents both CLI and Python usage.

For example, the YOLO CLI can be used with:

```bash
yolo predict model=yolo26n.pt
```

> The exact model file used by BorderGuard AI may be different. Use the model included/configured by this project.

---

## OpenCV

OpenCV is used for video capture, frame processing, image processing, and computer-vision operations.

Install:

```bash
python -m pip install opencv-python
```

Verify:

```bash
python -c "import cv2; print(cv2.__version__)"
```

OpenCV's documentation recommends installing the Python package from PyPI for typical Python projects.

> Do not install multiple OpenCV variants unnecessarily in the same environment. For example, normally choose `opencv-python` OR an appropriate contrib/headless variant depending on the project requirements.

---

## NumPy

NumPy is used for numerical operations and image/frame data.

Install:

```bash
python -m pip install numpy
```

Verify:

```bash
python -c "import numpy; print(numpy.__version__)"
```

---

# 7️⃣ Face Detection & Recognition

BorderGuard AI can use computer-vision models/components for face detection and recognition.

The required model files should be placed in the project's configured model directory.

Example:

```text
models/
├── face_detection/
├── face_recognition/
└── object_detection/
```

> Use the exact model filenames and paths configured in the project source code. Do not rename model files unless the corresponding configuration/code is updated.

---

# 8️⃣ Number-Plate / OCR Components

Number-plate processing may require additional OCR or computer-vision dependencies depending on the implementation.

If the repository provides these dependencies in `requirements.txt`, install them using:

```bash
python -m pip install -r requirements.txt
```

If a required dependency is unavailable for your operating system or Python version, check that dependency's **official documentation** for the supported installation method and compatible versions.

Avoid downloading Python packages or model files from unknown third-party websites.

---

# 9️⃣ Verify the Python Environment

After installation, you can check the installed packages:

```bash
python -m pip list
```

You can also check individual packages:

```bash
python -m pip show fastapi
python -m pip show ultralytics
python -m pip show opencv-python
python -m pip show numpy
```

---

# 🔟 Run BorderGuard AI

Make sure the virtual environment is activated.

Windows:

```bash
venv\Scripts\activate
```

Then start the backend:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open the dashboard in your browser:

```text
http://127.0.0.1:8000
```

---

# 🪟 Windows Quick Start

If `run.bat` is included in the repository, you can use it as a shortcut to start the project.

```text
run.bat
```

The batch file should activate the required environment and start the application according to the commands configured in the project.

---

# 📦 requirements.txt

The repository includes a `requirements.txt` file so that users can install the project's Python dependencies with one command:

```bash
python -m pip install -r requirements.txt
```

When updating the project, dependencies can be recorded with:

```bash
python -m pip freeze > requirements.txt
```

Only include packages actually required by the project.

---

# 🔧 If Something Is Missing

BorderGuard AI is provided as an open-source educational project.

Most required Python dependencies should be listed in:

```text
requirements.txt
```

If a dependency is missing or installation fails:

1. Check the error message.
2. Confirm that Python 3.12 and the virtual environment are being used.
3. Upgrade pip:

```bash
python -m pip install --upgrade pip
```

4. Check the dependency's official documentation for supported Python versions and installation instructions.
5. Install the compatible version required by the project.
6. If necessary, open an issue in this GitHub repository with the complete error message.

> **Recommendation:** Prefer official project documentation and official package repositories instead of downloading executables, Python packages, or AI models from unknown websites.

---

# 🧪 Test the Environment

Before running the complete application, you can test the main libraries:

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

```bash
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

```bash
python -c "import fastapi; print('FastAPI installed successfully')"
```

If these commands complete without an import error, the corresponding packages are available in the active Python environment.

---

# 🌐 Official Documentation

For installation problems or version compatibility, use the official documentation:

* Python — https://www.python.org/
* Python virtual environments — https://docs.python.org/3.12/tutorial/venv.html
* FastAPI — https://fastapi.tiangolo.com/
* Ultralytics — https://docs.ultralytics.com/
* OpenCV — https://docs.opencv.org/
* NumPy — https://numpy.org/
* Git — https://git-scm.com/

The project does not guarantee that every third-party dependency will remain compatible with every future Python or operating-system version. Check the relevant project's official documentation when upgrading the environment.
