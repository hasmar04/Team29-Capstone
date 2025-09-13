import os
import sys
import subprocess
import platform

REQUIRED_PACKAGES = [
    "ultralytics",
    "matplotlib",
    "numpy",
    "opencv-python",
    "scikit-learn",
    "pydantic",
    "scipy"
]

def install_pip():
    try:
        import pip
        return True
    except ImportError:
        print("pip not found. Attempting to install pip...")
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"])
        try:
            import pip
            return True
        except ImportError:
            print("Failed to install pip.")
            return False

def install_packages(packages):
    for package in packages:
        print(f"\nInstalling {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package])

def check_tkinter():
    try:
        import tkinter
        print("Tkinter is already installed.")
    except ImportError:
        print("Tkinter is not installed. Attempting installation...")
        if platform.system() == "Linux":
            distro = platform.linux_distribution()[0].lower() if hasattr(platform, "linux_distribution") else ""
            if "ubuntu" in distro or "debian" in distro:
                subprocess.run(["sudo", "apt-get", "install", "-y", "python3-tk"])
            else:
                print("Please install Tkinter manually for your Linux distro.")
        elif platform.system() == "Darwin":
            print("On macOS, Tkinter is included with Python from python.org. Consider reinstalling Python from https://www.python.org")
        elif platform.system() == "Windows":
            print("Tkinter should be included in the Windows Python installer. Reinstall Python if it's missing.")
        else:
            print("Unknown OS — cannot install tkinter automatically.")

def main():
    print("Checking for pip...")
    if not install_pip():
        sys.exit("pip installation failed. Cannot continue.")

    print("\nInstalling required packages...")
    install_packages(REQUIRED_PACKAGES)

    print("\nChecking for tkinter...")
    check_tkinter()

    print("\n Setup complete.")

if __name__ == "__main__":
    main()
