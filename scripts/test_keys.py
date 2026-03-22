from pynput import keyboard
import time
import sys

def on_press(key):
    print(f"✅ Key pressed: {key}")
    if key == keyboard.Key.cmd_r:
        print("🎯 Right Command key detected successfully!")
        
def on_release(key):
    if key == keyboard.Key.esc:
        return False # stop listener

print("Listening for keys... Press 'Right Command' to test.")
print("Press 'Esc' to exit.")

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except Exception as e:
    print(f"❌ Error starting listener: {e}")
    print("This is likely a macOS permissions issue. Please grant Input Monitoring or Accessibility permissions to your terminal.")
    sys.exit(1)
