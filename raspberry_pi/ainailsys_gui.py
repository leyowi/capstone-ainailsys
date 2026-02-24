#!/usr/bin/env python3
"""
AINAILSYS - FINAL PRODUCTION VERSION
Complete system with voice announcements and safe shutdown
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import traceback
import subprocess
import threading
import os

MODEL_DIR = Path.home() / "capstone" / "models"
STAGE1_MODEL = MODEL_DIR / "stage1_binary.onnx"
STAGE2_MODEL = MODEL_DIR / "stage2_multiclass.onnx"

# Display
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 280

# COLOR PALETTE
COLOR_LIGHTEST = "#F5EFE6"
COLOR_LIGHT = "#E8DFCA"
COLOR_BLUE = "#6D94C5"
COLOR_PALE_BLUE = "#CBDCEB"
COLOR_TEXT_DARK = "#2C3E50"
COLOR_WARNING = "#d63031"
COLOR_SUCCESS = "#00b894"
COLOR_MID = "#697565"

# Deficiency mapping
DEFICIENCY_MAP = {
    'spooning': 'Iron', 'onycholysis': 'Iron', 'onychorrhexis': 'Iron',
    'beaus_lines': 'Folate', 'onychoschizia': 'Folate',
    'melanonychia': 'B12', 'blue_nails': 'B12'
}

# ============================================
# IMAGE PROCESSING
# ============================================

def detect_nail_presence(image):
    """
    Binary image processing for nail detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    white_percentage = (white_pixels / total_pixels) * 100
    
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    texture_var = gray.std()
    brightness = gray.mean()
    
    print(f"\nNail Detection (Binary Image Processing):")
    print(f"  White pixels: {white_percentage:.2f}%")
    print(f"  Edge pixels: {edge_percentage:.2f}%")
    print(f"  Texture variance: {texture_var:.2f}")
    print(f"  Brightness: {brightness:.2f}")
    
    has_white = white_percentage > 30
    has_edges = edge_percentage > 1.2
    has_texture = texture_var > 20
    not_overexposed = brightness < 200
    
    has_nail = has_white and has_edges and has_texture and not_overexposed
    
    print(f"\nValidation:")
    print(f"  White > 30%: {has_white}")
    print(f"  Edges > 1.2%: {has_edges}")
    print(f"  Texture > 20: {has_texture}")
    print(f"  Brightness < 200: {not_overexposed}")
    
    if has_nail:
        print(f"  ✅ NAIL DETECTED")
    else:
        print(f"  ❌ NO NAIL")
    
    return has_nail

def preprocess_image(image):
    """Preprocess for AI inference"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_float = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_float - mean) / std
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_batch = np.expand_dims(image_transposed, axis=0)
    return image_batch

# ============================================
# GUI CLASS
# ============================================

class AINAILSYSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AINAILSYS")
        
        # Set fullscreen BEFORE configuring anything else
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)  # Keep on top
        
        self.root.configure(bg=COLOR_LIGHTEST)
        
        # Hidden exit: ESC 3 times
        self.esc_count = 0
        self.root.bind('<Escape>', self.handle_escape)
        
        self.cap = None
        self.current_frame = None
        self.is_analyzing = False
        self.on_preview_page = True
        
        self.preview_page = None
        self.results_page = None
        
        print("Initializing AINAILSYS...")
        self.load_models()
        self.start_camera()
        self.setup_ui()
        self.show_preview_page()
        self.update_preview()
        print("AINAILSYS ready!")
        
        # Force fullscreen again after 100ms (ensures it sticks)
        self.root.after(100, lambda: self.root.attributes('-fullscreen', True))
        
        # Startup announcement
        self.speak("System ready")
    
    def handle_escape(self, event):
        """Hidden exit - press ESC 3 times"""
        self.esc_count += 1
        print(f"ESC pressed {self.esc_count}/3")
        if self.esc_count >= 3:
            print("Exiting...")
            self.exit_app()
        self.root.after(2000, lambda: setattr(self, 'esc_count', 0))
    
    def speak(self, text):
        """
        Text-to-speech in background thread
        """
        def speak_in_background():
            try:
                print(f"🔊 Speaking: {text}")
                
                # Add leading pauses to prevent clipping
                fixed_text = ". . . ." + text
                
                # Set USB audio device
                env = os.environ.copy()
                env['AUDIODEV'] = 'hw:2,0'
                
                # Run espeak
                subprocess.run(
                    ['espeak', fixed_text, '-ven+f3', '-s', '130', '-a', '200', '-g', '10'],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                print(f"✅ Speech completed")
                
            except:
                print(f"❌ Speech failed")
        
        # Run in background thread (non-blocking)
        thread = threading.Thread(target=speak_in_background, daemon=True)
        thread.start()
    
    def load_models(self):
        """Load models"""
        print("Loading models...")
        self.stage1_session = ort.InferenceSession(str(STAGE1_MODEL))
        self.stage2_session = ort.InferenceSession(str(STAGE2_MODEL))
        
        with open(MODEL_DIR / "stage1_binary.json") as f:
            self.stage1_meta = json.load(f)
        with open(MODEL_DIR / "stage2_multiclass.json") as f:
            self.stage2_meta = json.load(f)
        print("Models loaded!")
    
    def start_camera(self):
        """Start webcam"""
        print("Starting camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera started!")
    
    def setup_ui(self):
        """Create UI"""
        
        # ==========================================
        # PAGE 1: PREVIEW PAGE
        # ==========================================
        
        self.preview_page = tk.Frame(self.root, bg=COLOR_LIGHTEST)

        # Title
        tk.Label(
            self.preview_page,
            text="AINAILSYS",
            font=("Arial", 30, "bold"),
            bg=COLOR_LIGHTEST,
            fg=COLOR_BLUE
        ).pack(pady=5)

        # Instruction - moved up, same size
        tk.Label(
            self.preview_page,
            text="Position fingernail clearly in camera view",
            font=("Arial", 10),
            bg=COLOR_LIGHTEST,
            fg=COLOR_TEXT_DARK
        ).pack(pady=2)

        # Camera preview frame
        preview_frame = tk.Frame(self.preview_page, bg=COLOR_LIGHT, relief=tk.RAISED, bd=2)
        preview_frame.pack(pady=10, padx=20)

        self.camera_label = tk.Label(preview_frame, bg=COLOR_LIGHT)
        self.camera_label.pack(padx=3, pady=3)

        # CAPTURE button - EXACT MATCH to NEW SCAN button
        self.capture_btn = tk.Button(
            self.preview_page,
            text="CAPTURE & ANALYZE",
            font=("Arial", 16, "bold"),
            bg=COLOR_BLUE,
            fg="white",
            activebackground=COLOR_PALE_BLUE,
            activeforeground=COLOR_TEXT_DARK,
            command=self.capture_and_analyze,
            height=2,
            width=20,
            relief=tk.RAISED,
            bd=4
        )
        self.capture_btn.pack(pady=13)
        
        # ==========================================
        # PAGE 2: RESULTS PAGE
        # ==========================================
        
        self.results_page = tk.Frame(self.root, bg=COLOR_LIGHTEST)
        
        tk.Label(
            self.results_page,
            text="AINAILSYS",
            font=("Arial", 30, "bold"),
            bg=COLOR_LIGHTEST,
            fg=COLOR_BLUE
        ).pack(pady=8)
        
        results_container = tk.Frame(self.results_page, bg=COLOR_LIGHT, relief=tk.RAISED, bd=3)
        results_container.pack(pady=10, padx=30, fill=tk.BOTH, expand=True)
        
        tk.Label(
            results_container,
            text="RESULTS",
            font=("Arial", 16, "bold"),
            bg=COLOR_LIGHT,
            fg=COLOR_BLUE
        ).pack(pady=10)
        
        self.results_content = tk.Frame(results_container, bg=COLOR_LIGHT)
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # NEW SCAN button
        tk.Button(
            self.results_page,
            text="NEW SCAN",
            font=("Arial", 16, "bold"),
            bg=COLOR_BLUE,
            fg="white",
            activebackground=COLOR_PALE_BLUE,
            command=self.show_preview_page,
            height=2,
            width=20,
            relief=tk.RAISED,
            bd=4
        ).pack(pady=10)
        
        # POWER OFF button
        power_off_btn = tk.Button(
            self.results_page,
            text="POWER OFF",
            font=("Arial", 10, "bold"),
            bg=COLOR_WARNING,
            fg="white",
            activebackground="#c0392b",
            command=self.shutdown_system,
            height=1,
            width=10,  # Reduced from 15
            relief=tk.RAISED,
            bd=3
        )
        # Position in lower right corner with padding
        power_off_btn.place(relx=1.0, rely=1.0, x=-20, y=-20, anchor="se")
    
    def show_preview_page(self):
        """Show preview page"""
        print("Switching to preview page")
        self.results_page.pack_forget()
        self.preview_page.pack(fill=tk.BOTH, expand=True)
        self.on_preview_page = True
    
    def show_results_page(self):
        """Show results page"""
        print("Switching to results page")
        self.preview_page.pack_forget()
        self.results_page.pack(fill=tk.BOTH, expand=True)
        self.on_preview_page = False
    
    def update_preview(self):
        """Update camera preview"""
        if self.on_preview_page and not self.is_analyzing:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if ret:
                    self.current_frame = frame.copy()
                    
                    display_frame = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.camera_label.imgtk = imgtk
                    self.camera_label.configure(image=imgtk)
        
        self.root.after(30, self.update_preview)
    
    def capture_and_analyze(self):
        """Capture and analyze"""
        if self.current_frame is None:
            print("No frame captured!")
            return
        
        self.is_analyzing = True
        self.capture_btn.config(state=tk.DISABLED, text="ANALYZING...")
        self.root.update()
        
        try:
            print("\n" + "="*50)
            print("ANALYSIS STARTING")
            print("="*50)
            
            # Binary preprocessing
            has_nail = detect_nail_presence(self.current_frame)
            
            if not has_nail:
                print("\n🚫 NO NAIL DETECTED")
                self.display_no_nail_result()
            else:
                print("\nRunning AI classification...")
                results = self.analyze_image(self.current_frame)
                
                stage1_confidence = results['stage1']['confidence']
                prediction = results['stage1']['prediction']
                
                print(f"\nSTAGE 1 RESULTS:")
                print(f"  Prediction: {prediction.upper()}")
                print(f"  Confidence: {stage1_confidence:.2%}")
                
                if prediction == 'anemic' and 'stage2' in results:
                    stage2 = results['stage2']
                    print(f"\nSTAGE 2 RESULTS:")
                    print(f"  Abnormality: {stage2['abnormality']}")
                    print(f"  Deficiency: {stage2['deficiency']}")
                    print(f"  Confidence: {stage2['confidence']:.2%}")
                
                self.display_results(results)
            
            print("="*50 + "\n")
        
        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()
            self.display_error_result(str(e))
        
        finally:
            self.is_analyzing = False
            self.capture_btn.config(state=tk.NORMAL, text="CAPTURE & ANALYZE")
            self.show_results_page()
    
    def analyze_image(self, image):
        """Two-stage inference"""
        image_tensor = preprocess_image(image)
        
        stage1_result = self.run_stage1(image_tensor)
        result = {'stage1': stage1_result}
        
        if stage1_result['prediction'] == 'anemic':
            stage2_result = self.run_stage2(image_tensor)
            result['stage2'] = stage2_result
        
        return result
    
    def run_stage1(self, image_tensor):
        """Stage 1"""
        input_name = self.stage1_session.get_inputs()[0].name
        output_name = self.stage1_session.get_outputs()[0].name
        outputs = self.stage1_session.run([output_name], {input_name: image_tensor})
        
        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_class = np.argmax(probs)
        
        class_names = self.stage1_meta['class_names']
        
        return {
            'prediction': class_names[pred_class],
            'confidence': float(probs[pred_class]),
            'probabilities': {name: float(prob) for name, prob in zip(class_names, probs)}
        }
    
    def run_stage2(self, image_tensor):
        """Stage 2"""
        input_name = self.stage2_session.get_inputs()[0].name
        output_name = self.stage2_session.get_outputs()[0].name
        outputs = self.stage2_session.run([output_name], {input_name: image_tensor})
        
        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_class = np.argmax(probs)
        
        abnormality = self.stage2_meta['class_names'][pred_class]
        
        return {
            'abnormality': abnormality,
            'deficiency': DEFICIENCY_MAP[abnormality],
            'confidence': float(probs[pred_class])
        }
    
    def display_no_nail_result(self):
        """No nail detected"""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.results_content,
            text="NO NAIL DETECTED",
            font=("Arial", 20, "bold"),
            bg=COLOR_LIGHT,
            fg=COLOR_WARNING
        ).pack(pady=20)
        
        tk.Label(
            self.results_content,
            text="Unable to detect fingernail\nin the image",
            font=("Arial", 12),
            bg=COLOR_LIGHT,
            fg=COLOR_TEXT_DARK,
            justify=tk.CENTER
        ).pack(pady=10)
        
        tk.Label(
            self.results_content,
            text="Please ensure:\n• Fingernail is visible\n• Good lighting\n• Nail in focus",
            font=("Arial", 10),
            bg=COLOR_LIGHT,
            fg=COLOR_TEXT_DARK,
            justify=tk.CENTER
        ).pack(pady=10)
        
        # Voice announcement
        self.speak("No nail detected. Please position fingernail and try again.")
    
    def display_error_result(self, error):
        """Display error"""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.results_content,
            text="ERROR",
            font=("Arial", 20, "bold"),
            bg=COLOR_LIGHT,
            fg=COLOR_WARNING
        ).pack(pady=20)
        
        tk.Label(
            self.results_content,
            text="Analysis failed\nPlease try again",
            font=("Arial", 12),
            bg=COLOR_LIGHT,
            fg=COLOR_TEXT_DARK,
            justify=tk.CENTER
        ).pack(pady=10)
    
    def display_results(self, results):
        """Display results"""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        stage1 = results['stage1']
        
        if stage1['prediction'] == 'healthy':
            # HEALTHY
            tk.Label(
                self.results_content,
                text="HEALTHY",
                font=("Arial", 26, "bold"),
                bg=COLOR_LIGHT,
                fg=COLOR_SUCCESS
            ).pack(pady=20)
            
            tk.Label(
                self.results_content,
                text="No signs of anemia detected",
                font=("Arial", 13),
                bg=COLOR_LIGHT,
                fg=COLOR_TEXT_DARK
            ).pack(pady=12)
            
            tk.Label(
                self.results_content,
                text=f"Confidence: {stage1['confidence']*100:.1f}%",
                font=("Arial", 11),
                bg=COLOR_LIGHT,
                fg=COLOR_BLUE
            ).pack(pady=8)
            
            # Voice announcement
            self.speak("Healthy Nail. No signs of anemia detected.")
        
        else:
            # ANEMIC
            tk.Label(
                self.results_content,
                text="ANEMIC",
                font=("Arial", 22, "bold"),
                bg=COLOR_LIGHT,
                fg=COLOR_WARNING
            ).pack(pady=10)
            
            if 'stage2' in results:
                stage2 = results['stage2']
                
                tk.Label(
                    self.results_content,
                    text=stage2['abnormality'].replace('_', ' ').title(),
                    font=("Arial", 13, "bold"),
                    bg=COLOR_LIGHT,
                    fg=COLOR_TEXT_DARK
                ).pack(pady=4)
                
                tk.Label(
                    self.results_content,
                    text=f"{stage2['deficiency']} Deficiency",
                    font=("Arial", 12),
                    bg=COLOR_LIGHT,
                    fg=COLOR_BLUE
                ).pack(pady=4)
                
                tk.Label(
                    self.results_content,
                    text=f"Confidence: {stage2['confidence']*100:.1f}%",
                    font=("Arial", 10),
                    bg=COLOR_LIGHT,
                    fg=COLOR_BLUE
                ).pack(pady=4)
                
                # Voice announcement
                abnormality = stage2['abnormality'].replace('_', ' ')
                deficiency = stage2['deficiency']
                self.speak(f"Anemic {abnormality} nail detected. {deficiency} deficiency.")
            
            tk.Label(
                self.results_content,
                text="Consult healthcare provider\nfor diagnosis",
                font=("Arial", 9, "italic"),
                bg=COLOR_LIGHT,
                fg=COLOR_TEXT_DARK,
                justify=tk.CENTER
            ).pack(pady=8)
    
    def shutdown_system(self):
        """Show shutdown confirmation dialog"""
        # Create confirmation popup
        confirm = tk.Toplevel(self.root)
        confirm.title("Power Off")
        confirm.geometry("450x250")
        confirm.configure(bg=COLOR_LIGHT)
        
        # Center the popup
        confirm.transient(self.root)
        confirm.grab_set()
        
        # Warning message
        tk.Label(
            confirm,
            text="POWER OFF DEVICE?",
            font=("Arial", 18, "bold"),
            bg=COLOR_LIGHT,
            fg=COLOR_WARNING
        ).pack(pady=20)
        
        tk.Label(
            confirm,
            text="This will shut down the system.\nWait 30 seconds before unplugging.",
            font=("Arial", 11),
            bg=COLOR_LIGHT,
            fg=COLOR_TEXT_DARK,
            justify=tk.CENTER
        ).pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(confirm, bg=COLOR_LIGHT)
        btn_frame.pack(pady=20)
        
        tk.Button(
            btn_frame,
            text="YES, POWER OFF",
            font=("Arial", 13, "bold"),
            bg=COLOR_WARNING,
            fg="white",
            command=lambda: self.do_shutdown(confirm),
            width=16,
            height=2,
            relief=tk.RAISED,
            bd=3
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text="CANCEL",
            font=("Arial", 13, "bold"),
            bg=COLOR_MID,
            fg="white",
            command=confirm.destroy,
            width=12,
            height=2,
            relief=tk.RAISED,
            bd=3
        ).pack(side=tk.LEFT, padx=10)
    
    def do_shutdown(self, confirm_window):
        """Execute system shutdown"""
        print("Shutting down system...")
        
        # Close confirmation window
        confirm_window.destroy()
        
        # Show shutdown message
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.results_content,
            text="POWERING OFF...",
            font=("Arial", 24, "bold"),
            bg=COLOR_LIGHT,
            fg=COLOR_WARNING
        ).pack(pady=30)
        
        tk.Label(
            self.results_content,
            text="Please wait 30 seconds\nbefore unplugging power",
            font=("Arial", 14),
            bg=COLOR_LIGHT,
            fg=COLOR_TEXT_DARK,
            justify=tk.CENTER
        ).pack(pady=20)
        
        self.root.update()
        
        # Voice announcement
        self.speak("Powering off.")
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Small delay for voice to start, then shutdown
        self.root.after(2000, lambda: subprocess.run(['sudo', 'shutdown', 'now']))
    
    def exit_app(self):
        """Exit (hidden ESC 3x)"""
        print("Exiting AINAILSYS...")
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        finally:
            self.root.destroy()

# ============================================
# MAIN
# ============================================

def main():
    root = tk.Tk()
    app = AINAILSYSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()