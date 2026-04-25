# scripts/fix_macos_deps.py
import os
import subprocess
import sys
from pathlib import Path

def fix_mac_computer_library_paths():
    """
    Fixes a problem on Mac computers where the AI program cannot find its library files.
    It adds the correct library folder path to the AI program's internal search list.
    """
    # Only run this on Mac computers
    if sys.platform != "darwin":
        return

    # 1. Find the folder where the program's packages are kept
    virtual_environment_folder = Path(".venv")
    if not virtual_environment_folder.exists():
        print("❌ .venv folder not found. Please create it first.")
        return

    # Find the specific binary code file that is having the problem
    binary_code_files = list(virtual_environment_folder.glob("**/sherpa_onnx/lib/_sherpa_onnx*.so"))
    if not binary_code_files:
        print("❌ Could not find the AI program file to fix.")
        return

    file_to_patch = binary_code_files[0]
    
    # 2. Define the path to the library folder that holds onnxruntime
    # This is where the missing files are actually stored.
    library_folder_path = virtual_environment_folder.absolute() / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "onnxruntime" / "capi"

    if not library_folder_path.exists():
        print(f"❌ Library folder not found at {library_folder_path}")
        return

    print(f"🔧 Fixing the library search list for: {file_to_patch.name}...")
    
    # 3. Add the folder path to the file's internal search list using a Mac system tool
    try:
        # install_name_tool is a Mac tool that changes where a file looks for libraries
        subprocess.run([
            "install_name_tool", 
            "-add_rpath", str(library_folder_path), 
            str(file_to_patch)
        ], check=True, capture_output=True)
        print(f"✅ Successfully added the library path: {library_folder_path}")
    except subprocess.CalledProcessError as error_message:
        # If the path is already there, we don't need to do anything
        error_text = error_message.stderr.decode().lower()
        if "duplicate" in error_text and "path" in error_text:
            print("ℹ️  The library path is already fixed.")
        else:
            print(f"❌ Failed to fix the file: {error_message.stderr.decode()}")

if __name__ == "__main__":
    fix_mac_computer_library_paths()
