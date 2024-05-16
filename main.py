import subprocess
import sys

def run_detect():
    result = subprocess.run(['python', 'detect.py'], capture_output=True, text=True)
    output = result.stdout.strip()
    print("Detect.py Output:", output)
    return output

def run_imgtostring():
    result = subprocess.run(['python', 'imgtostring.py'], capture_output=True, text=True)
    print("ImgToString.py Output:", result.stdout)

def main():
    detect_output = run_detect()
    if "You are authorised to vote from this centre" in detect_output:
        print("Authorization confirmed. Running ImgToString...")
        run_imgtostring()
    else:
        print("Authorization failed. Exiting.")

if __name__ == "__main__":
    main()
