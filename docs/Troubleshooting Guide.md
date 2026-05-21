# Troubleshooting Guide

Note: The project was run on a development platform called Visual Studio Code (VS Code). While you can run the program without VS Code, if you’d like to see the code for the programs, this is the integrated development platform (IDE) we recommend and is what is shown in our examples.

# Q: Nothing happens when I run the python file to start the program

## A: Are you running the correct file?

Make sure you are following the instructions in [GUI_README.md](../GUI_README.md)

## A: Check your Desktop

Sometimes when you run the file, if you have any other windows or applications open on your computer, the popup for the program appears underneath everything on your desktop. To quickly navigate to your desktop on Windows, click the very far right part of the taskbar (in this instance it is in the bottom right corner of the screen)

You should then be able to see the main program window.

# Q: Program appears to run then crashes

## A: Open in an IDE

If the program launches, then crashes, you might get an error message in the terminal of the development platform or IDE you are in. If you tried to launch the program (through File Explorer or Finder), the error message may be abstracted from view. We recommend opening in an IDE, namely VS Code, to see what may have caused the error.

## A: Check installations

Make sure you are following the instructions in [GUI_README.md](../GUI_README.md)



# Q: I’m closing the video window, but it keeps reappearing

## A: Hit ‘q’ to exit

If the analysis is finished and you’d like to close the analysis window that pops up but find that it reappears every time, instead of hitting the ‘X’ on the window or closing VS Code, hit the ‘q’ on your keyboard to stop it. In the terminal, you will get a message saying, ‘Exit video processing’.

# Q: When I try to run the program, it says I have items or modules missing

If you attempt to run the program within VS Code, and you get this error, here are some things you can check.

## A: Make sure you have followed the usage instructions

Make sure you are following the instructions in [GUI_README.md](../GUI_README.md)

## A: Confirm that your computer has downloaded modules during installation

If you need to manually install any packages, see this section for command line prompts.

After running the installation file for your computer, you should have the packages listed in [requirements.txt](../requirements.txt)

### **Check installed packages in computer or IDE terminal**

To display the list of your installed modules, type:

`pip list`

**Important**: the installation should have downloaded pip and Python for you, if you are missing pip or python or are unsure if those modules were installed, please read the following section.

### **Check that Python and its package manager ‘pip’ are installed**

`python –version`

If installed, you will see: `python <version>`

If not installed, you will see a ‘`not found`’ error

`pip –version`

### **Update pip or Python**

To update pip on all platforms:

`python -m pip install --upgrade pip`

To update Python (macOS):

`brew update`

`brew upgrade python`

To update Python (Windows): 

Officially, you’ll have to go to the Python website ([python.org](http://python.org)) and download a new version, or you can try use the interpreter in VS Code. 

## A: Check the version compatibility of the installed packages

Check that the version of each package you have installed matches with what is required in [requirements.txt](../requirements.txt).

Note: if you have upgraded a package where the version is a dependency, then you may have installed an earlier version. In this guide, packages and modules are used interchangeably.

### **How to check versions in terminal**

For instructions on how to access the terminal/command line in VS Code or on your computer: [click here.](#q-how-do-i-run-code-commands-in-terminal)

To check a specific module version, type:

`pip show <name of module>`

e.g.: `pip show ultralytics`

### **How to install a package**

If you need to install a specific version of a package, type: 

`pip install <name of module>`

e.g.: `pip install ultralytics`

### How to install an alternate version

If you need to install a specific version of a package, type:

`pip install <name of module> == <version>`

e.g.: `pip install opencv-contrib-python==4.8.0.74` , `pip install numpy==1.24.4`

### **How to uninstall a package**

To prevent any conflicts, you may choose to uninstall a version before reinstalling a different version. To do this, you have to uninstall the module.

To uninstall, type:

`pip uninstall <name of module>`

e.g.: `pip uninstall ultralytics`

### **How to update package**

Otherwise, if you’d just like to update your package, type: `pip install --upgrade <package-name>`

# Q: Message "Models are still loading" is shown

## A: Wait for YOLO models to load completely

## A: Check that model files exist in `models/` directory

# Q: Message "No video file selected" is shown

## A: Ensure you've selected a valid video file

## A: Check file format
The program supports .mp4, .avi, .mov, .gif file types.

# Q: Message "Processing failed" is shown

## A: Check the processing log for detailed error messages
This can be found in the application window. 

## A: Ensure all dependencies are installed
If you are running the code from the repository, run `pip install -r requirements.txt` in the terminal. 

# Q: How do I run code commands in terminal?

The project was run on a development platform called Visual Studio Code (VS Code). Some of the examples where terminal is shown, it is in VS Code however, most of the commands can be entered into Windows, Linux or Mac command line terminal.

## A: Windows PowerShell/Command Prompt

### **Method 1: Using the Start Menu**

1. Click the **Start** button (Windows icon) in the bottom-left corner of your screen. 
2. Type `cmd` or **Command Prompt** in the search bar.
3. Click on the **Command Prompt** app that appears in the search results.

### **Method 2: Using the Run Dialog**

1. Press **Windows key + R** on your keyboard to open the Run dialog box. 
2. Type `cmd` into the box.
3. Press **Enter** or click **OK**.

### **Method 3: From File Explorer**

1. Open **File Explorer** (folder icon on the taskbar or press **Windows key + E**). 
2. Click on the address bar at the top.
3. Type `cmd` and press **Enter**.
    1. This will open Command Prompt in the current folder location.

### **Open as Administrator (if needed)**

1. Search for **Command Prompt** in the Start Menu.
2. Right-click the **Command Prompt** app.
3. Select **Run as administrator**.
    1. This gives Command Prompt elevated permissions needed for some commands.

## A: Mac Terminal

### **Method 1: Using Spotlight Search**

1. Press **Command (**⌘**) + Spacebar** to open Spotlight Search. 
2. Type **Terminal**.
3. Press **Enter** when Terminal appears as the top result.

### **Method 2: Using Finder**

1. Open **Finder** (the smiling face icon in the Dock).
2. Go to **Applications** in the sidebar.
3. Open the **Utilities** folder.
4. Double-click **Terminal** to open it.

### **Method 3: Using Launchpad**

1. Click the **Launchpad** icon in the Dock (rocket icon).
2. Type **Terminal** in the search bar at the top.
3. Click the **Terminal** app icon to open it.

### **Run as Administrator (if needed)**

Prefix the command with `sudo` and enter your password when prompted if you need elevated/admin permissions.

## A: Linux Terminal

### Method 1: Using tty

To open a full-screen terminal on Linux, press **Ctrl + Alt + <F1-F12>.** 

To return to the Desktop Environment, press **Ctrl + Alt + <F1-F12>** until you find the correct screen (typically **F1** or **F2**)

### Method 2: Using a Terminal Emulator

Depending on your Desktop Environment and Linux Distribution, different terminal programs may be installed, such as Konsole (KDE), Terminal (Gnome), or XTerm. 

### **Run as Administrator (if needed)**

Prefix the command with `sudo` and enter your password when prompted if you need elevated/admin permissions.