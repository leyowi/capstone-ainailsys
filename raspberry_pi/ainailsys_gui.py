#!/usr/bin/env python3
"""
AINAILSYS - FINAL PRODUCTION VERSION
Complete system with homepage, patient info, CSV database, voice, and safe shutdown
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
import csv
from datetime import datetime

MODEL_DIR    = Path.home() / "capstone" / "models"
STAGE1_MODEL = MODEL_DIR / "stage1_binary.onnx"
STAGE2_MODEL = MODEL_DIR / "stage2_multiclass.onnx"
DB_PATH      = Path.home() / "capstone" / "patient_records.csv"

# Display
PREVIEW_WIDTH  = 640
PREVIEW_HEIGHT = 280
SCREEN_W       = 800
SCREEN_H       = 480

# COLOR PALETTE
COLOR_LIGHTEST  = "#F5EFE6"
COLOR_LIGHT     = "#E8DFCA"
COLOR_BLUE      = "#6D94C5"
COLOR_PALE_BLUE = "#CBDCEB"
COLOR_TEXT_DARK = "#2C3E50"
COLOR_WARNING   = "#d63031"
COLOR_SUCCESS   = "#00b894"
COLOR_MID       = "#697565"

# Deficiency mapping
DEFICIENCY_MAP = {
    'spooning':      'Iron',
    'onycholysis':   'Iron',
    'onychorrhexis': 'Iron',
    'beaus_lines':   'Folate',
    'onychoschizia': 'Folate',
    'melanonychia':  'B12',
    'blue_nails':    'B12',
}

CSV_COLUMNS = [
    "Timestamp", "Patient_ID", "Name", "Age", "Sex",
    "Last_Menstrual_Cycle", "Fasting_Status",
    "Prediction", "Abnormality_Type", "Deficiency_Type",
    "Confidence_Stage1", "Confidence_Stage2",
]

# ============================================
# DATABASE
# ============================================

def ensure_csv():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        with open(DB_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
    else:
        _migrate_csv()

def _migrate_csv():
    """Add any missing columns to an existing CSV (e.g. Patient_ID added later)."""
    try:
        with open(DB_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_cols = reader.fieldnames or []
            rows = list(reader)

        missing = [c for c in CSV_COLUMNS if c not in existing_cols]
        if not missing:
            return   # nothing to do

        print(f"⚙️  Migrating CSV — adding columns: {missing}")
        with open(DB_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
        print("CSV migration complete")
    except Exception as e:
        print(f"CSV migration failed: {e}")

def append_record(record: dict):
    ensure_csv()
    with open(DB_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow({col: record.get(col, "") for col in CSV_COLUMNS})

def lookup_patient_by_id(patient_id: str):
    """Return the most recent CSV record matching Patient_ID, or None."""
    if not patient_id.strip() or not DB_PATH.exists():
        return None
    try:
        with open(DB_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "Patient_ID" not in (reader.fieldnames or []):
                print("Patient_ID column not found in CSV — run ensure_csv() first")
                return None
            matches = [
                row for row in reader
                if row.get("Patient_ID", "").strip().lower()
                   == patient_id.strip().lower()
            ]
        if matches:
            print(f"  Returning patient found: {matches[-1].get('Name','?')}")
        else:
            print(f"  No record for ID: {patient_id!r}")
        return matches[-1] if matches else None
    except Exception as e:
        print(f"  Lookup error: {e}")
        return None

# ============================================
# IMAGE PROCESSING
# ============================================

def detect_nail_presence(image):
    """Binary image processing for nail detection"""
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels     = np.sum(binary == 255)
    total_pixels     = binary.size
    white_percentage = (white_pixels / total_pixels) * 100

    edges            = cv2.Canny(gray, 50, 150)
    edge_pixels      = np.sum(edges > 0)
    edge_percentage  = (edge_pixels / total_pixels) * 100

    texture_var = gray.std()
    brightness  = gray.mean()

    print(f"\nNail Detection (Binary Image Processing):")
    print(f"  White pixels: {white_percentage:.2f}%")
    print(f"  Edge pixels: {edge_percentage:.2f}%")
    print(f"  Texture variance: {texture_var:.2f}")
    print(f"  Brightness: {brightness:.2f}")

    has_white       = white_percentage > 35
    has_edges       = edge_percentage > 1.5
    has_texture     = texture_var > 25
    not_overexposed = brightness < 200

    has_nail = has_white and has_edges and has_texture and not_overexposed

    print(f"\nValidation:")
    print(f"  White > 35%: {has_white}")
    print(f"  Edges > 1.5%: {has_edges}")
    print(f"  Texture > 25: {has_texture}")
    print(f"  Brightness < 200: {not_overexposed}")
    print(f"  {' NAIL DETECTED' if has_nail else '❌ NO NAIL'}")

    return has_nail


def preprocess_image(image):
    """Preprocess for AI inference"""
    image_rgb        = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized    = cv2.resize(image_rgb, (224, 224))
    image_float      = image_resized.astype(np.float32) / 255.0
    mean             = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std              = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_float - mean) / std
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_batch      = np.expand_dims(image_transposed, axis=0)
    return image_batch

# ============================================
# GUI CLASS
# ============================================

class AINAILSYSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AINAILSYS")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg=COLOR_LIGHTEST)

        # Hidden exit: ESC 3 times
        self.esc_count = 0
        self.root.bind('<Escape>', self.handle_escape)

        self.cap             = None
        self.current_frame   = None
        self.is_analyzing    = False
        self.on_preview_page = False   # controls camera update loop

        # Patient data variables
        self.patient_id        = tk.StringVar()
        self.patient_name      = tk.StringVar()
        self.patient_age       = tk.StringVar()
        self.patient_sex       = tk.StringVar(value="")
        self.patient_menstrual = tk.StringVar()
        self.patient_fasting   = tk.StringVar(value="")
        self.patient_consent   = tk.BooleanVar(value=False)

        # Returning patient state
        self.is_returning_patient = False

        # On-screen keyboard (built-in Tkinter OSK, built in setup_ui)
        self.osk         = None
        self.osk_target  = None
        self.osk_visible = False

        # Stores last analysis result for CSV save
        self.last_results = None

        # Pages (assigned in setup_ui)
        self.homepage_page     = None
        self.patient_info_page = None
        self.preview_page      = None
        self.results_page      = None

        print("Initializing AINAILSYS...")
        ensure_csv()
        self.load_models()
        self.start_camera()
        self.setup_ui()
        self.show_homepage()
        self.update_preview()
        print("AINAILSYS ready!")

        # Force fullscreen again after 100ms (ensures it sticks on Pi)
        self.root.after(100, lambda: self.root.attributes('-fullscreen', True))

        self.speak("System ready")

    # ==========================================
    # UTILITY
    # ==========================================

    def handle_escape(self, event):
        """Hidden exit — press ESC 3 times"""
        self.esc_count += 1
        print(f"ESC pressed {self.esc_count}/3")
        if self.esc_count >= 3:
            print("Exiting...")
            self.exit_app()
        self.root.after(2000, lambda: setattr(self, 'esc_count', 0))

    def speak(self, text):
        """Text-to-speech in background thread"""
        def speak_in_background():
            try:
                print(f" Speaking: {text}")
                fixed_text = ". . . . " + text
                env = os.environ.copy()
                env['AUDIODEV'] = 'hw:2,0'
                subprocess.run(
                    ['espeak', fixed_text, '-ven+f3', '-s', '130', '-a', '200', '-g', '5'],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(" Speech completed")
            except Exception:
                print(" Speech failed")
        threading.Thread(target=speak_in_background, daemon=True).start()

    def load_models(self):
        print("Loading models...")
        self.stage1_session = ort.InferenceSession(str(STAGE1_MODEL))
        self.stage2_session = ort.InferenceSession(str(STAGE2_MODEL))
        with open(MODEL_DIR / "stage1_binary.json") as f:
            self.stage1_meta = json.load(f)
        with open(MODEL_DIR / "stage2_multiclass.json") as f:
            self.stage2_meta = json.load(f)
        print("Models loaded!")

    def start_camera(self):
        print("Starting camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera started!")

    # ==========================================
    # ON-SCREEN KEYBOARD  (tk.Frame inside root — no Toplevel, no topmost issues)
    # ==========================================

    def _build_osk(self):
        """Build keyboard as a Frame placed at the bottom of the root window."""
        self.osk_target  = None
        self.osk_visible = False

        kh = 195
        self.osk = tk.Frame(self.root, bg="#2C3E50", height=kh)

        # Position at bottom of root — hidden initially via place_forget
        self._osk_kh = kh

        rows = [
            ['1','2','3','4','5','6','7','8','9','0','-','⌫'],
            ['Q','W','E','R','T','Y','U','I','O','P'],
            ['A','S','D','F','G','H','J','K','L'],
            ['Z','X','C','V','B','N','M',',','.'],
            ['SPACE', 'CLEAR', 'HIDE'],
        ]

        for row_keys in rows:
            row_frame = tk.Frame(self.osk, bg="#2C3E50")
            row_frame.pack(pady=1)
            for key in row_keys:
                if key == 'SPACE':
                    w, label, cmd = 14, 'SPACE', lambda: self._osk_press(' ')
                elif key == 'CLEAR':
                    w, label, cmd = 8,  'CLR',   self._osk_clear
                elif key == 'HIDE':
                    w, label, cmd = 8,  'HIDE',  self._do_hide_keyboard
                elif key == '⌫':
                    w, label, cmd = 5,  '⌫',    self._osk_backspace
                else:
                    w, label, cmd = 4, key, lambda k=key: self._osk_press(k)

                is_action = key in ('⌫', 'CLEAR', 'HIDE', 'SPACE')
                tk.Button(
                    row_frame, text=label, width=w,
                    font=("Arial", 11, "bold"),
                    bg="#697565" if is_action else "#4a7ab5",
                    fg="white",
                    activebackground="#6D94C5",
                    relief=tk.FLAT, bd=0, pady=4,
                    command=cmd
                ).pack(side=tk.LEFT, padx=2)

    def _osk_press(self, char):
        if self.osk_target:
            self.osk_target.insert(tk.INSERT, char)
            # Keep focus on the entry so next keypress works
            self.osk_target.focus_set()

    def _osk_backspace(self):
        if self.osk_target:
            pos = self.osk_target.index(tk.INSERT)
            if pos > 0:
                self.osk_target.delete(pos - 1, pos)
            self.osk_target.focus_set()

    def _osk_clear(self):
        if self.osk_target:
            self.osk_target.delete(0, tk.END)
            self.osk_target.focus_set()

    def _show_keyboard(self, event=None):
        """Show OSK and remember which Entry called it."""
        if event and isinstance(event.widget, tk.Entry):
            self.osk_target = event.widget
        if not self.osk_visible:
            # place() over everything else at the bottom of root
            self.osk.place(x=0, y=SCREEN_H - self._osk_kh,
                           width=SCREEN_W, height=self._osk_kh)
            self.osk.lift()
            self.osk_visible = True
        print(f"  Keyboard shown — target: {self.osk_target}")

    def _hide_keyboard(self, event=None):
        self._do_hide_keyboard()

    def _do_hide_keyboard(self):
        if self.osk_visible or self.osk.winfo_ismapped():
            self.osk.place_forget()
            self.osk_visible = False
            self.osk_target  = None
        print("  Keyboard hidden")

    def _destroy_keyboard(self):
        """Called on shutdown/exit — just hide it (Frame can't be re-created)."""
        try:
            self.osk.place_forget()
        except Exception:
            pass
        self.osk_visible = False
        self.osk_target  = None

    def _attach_keyboard(self, entry):
        """Show keyboard when field is tapped."""
        entry.bind("<FocusIn>", self._show_keyboard, add="+")

    # ==========================================
    # RETURNING PATIENT HELPERS
    # ==========================================

    def _populate_from_existing(self, record: dict):
        """Fill patient vars from a CSV record dict."""
        self.patient_name.set(record.get("Name", ""))
        self.patient_age.set(record.get("Age", ""))
        self.patient_sex.set(record.get("Sex", ""))
        self.patient_menstrual.set(record.get("Last_Menstrual_Cycle", ""))
        self.patient_fasting.set(record.get("Fasting_Status", ""))
        # Sync LMC entry state
        if record.get("Sex", "") == "Female":
            self.lmc_entry.configure(state=tk.NORMAL)
        else:
            self.lmc_entry.configure(state=tk.DISABLED)

    # ==========================================
    # PAGE NAVIGATION
    # ==========================================

    def _hide_all_pages(self):
        self._do_hide_keyboard()   # always hide OSK on any page change
        for page in (self.homepage_page, self.patient_info_page,
                     self.preview_page, self.results_page):
            if page:
                page.pack_forget()

    def show_homepage(self):
        print("Switching to homepage")
        self._hide_all_pages()
        self.on_preview_page = False
        self.homepage_page.pack(fill=tk.BOTH, expand=True)

    def show_patient_info_page(self):
        print("Switching to patient info page")
        self._hide_all_pages()
        self.on_preview_page = False
        self.patient_info_page.pack(fill=tk.BOTH, expand=True)

    def show_preview_page(self):
        print("Switching to preview page")
        self._hide_all_pages()
        self.on_preview_page = True
        self.preview_page.pack(fill=tk.BOTH, expand=True)

    def show_results_page(self):
        print("Switching to results page")
        self._hide_all_pages()
        self.on_preview_page = False
        self.results_page.pack(fill=tk.BOTH, expand=True)

    def _clear_patient_data(self):
        self.patient_id.set("")
        self.patient_name.set("")
        self.patient_age.set("")
        self.patient_sex.set("")
        self.patient_menstrual.set("")
        self.patient_fasting.set("")
        self.patient_consent.set(False)
        self.is_returning_patient = False
        self.last_results = None

    # ==========================================
    # SETUP UI
    # ==========================================

    def setup_ui(self):

        # ==========================================
        # PAGE 0: HOMEPAGE
        # ==========================================

        self.homepage_page = tk.Frame(self.root, bg=COLOR_LIGHTEST)

        # Top accent bar
        tk.Frame(
            self.homepage_page,
            height=100,
            bg=COLOR_LIGHTEST
        ).pack(pady=(22, 4))

        tk.Label(
            self.homepage_page,
            text="AINAILSYS",
            font=("Arial", 30, "bold"),
            bg=COLOR_LIGHTEST,
            fg=COLOR_BLUE
        ).pack()

        tk.Label(
            self.homepage_page,
            text="AI-Powered Anemia Detection via Nail Analysis",
            font=("Arial", 10),
            bg=COLOR_LIGHTEST,
            fg=COLOR_MID
        ).pack(pady=(2, 14))

        # Accuracy strip
        info_strip = tk.Frame(self.homepage_page, bg=COLOR_PALE_BLUE, padx=20, pady=8)
        info_strip.pack(padx=80)
        tk.Label(
            info_strip,
            text="Stage-1 Accuracy: 98.41%   |   Stage-2 Accuracy: 96.43%   |   Overall: 97.45%",
            font=("Arial", 9),
            bg=COLOR_PALE_BLUE,
            fg=COLOR_TEXT_DARK
        ).pack()

        # START button
        tk.Button(
            self.homepage_page,
            text="START",
            font=("Arial", 16, "bold"),
            bg=COLOR_BLUE,
            fg="white",
            activebackground=COLOR_PALE_BLUE,
            activeforeground=COLOR_TEXT_DARK,
            command=lambda: (self._clear_patient_data(),
                             self.show_patient_info_page()),
            height=2,
            width=20,
            relief=tk.RAISED,
            bd=4
        ).pack(pady=20)

        # Power off (bottom-right corner)
        tk.Button(
            self.homepage_page,
            text="POWER OFF",
            font=("Arial", 10, "bold"),
            bg=COLOR_WARNING,
            fg="white",
            activebackground="#c0392b",
            command=self.shutdown_system,
            height=1,
            width=10,
            relief=tk.RAISED,
            bd=3
        ).place(relx=1.0, rely=1.0, x=-20, y=-20, anchor="se")

        # ==========================================
        # PAGE 1: PATIENT INFORMATION
        # ==========================================

        self.patient_info_page = tk.Frame(self.root, bg=COLOR_LIGHTEST)

        # Header bar
        hdr = tk.Frame(self.patient_info_page, bg=COLOR_BLUE, pady=6)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Patient Information",
                 font=("Arial", 14, "bold"),
                 fg="white", bg=COLOR_BLUE).pack()

        # Two-column body
        body = tk.Frame(self.patient_info_page, bg=COLOR_LIGHTEST)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        left  = tk.Frame(body, bg=COLOR_LIGHTEST)
        right = tk.Frame(body, bg=COLOR_LIGHTEST)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # LEFT — Patient ID  (top, always required)
        tk.Label(left, text="Patient ID *",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                 anchor=tk.W).pack(fill=tk.X, pady=(4, 2))
        id_row = tk.Frame(left, bg=COLOR_LIGHTEST)
        id_row.pack(fill=tk.X, pady=(0, 2))
        id_entry = tk.Entry(id_row, textvariable=self.patient_id,
                            font=("Arial", 11), relief=tk.SUNKEN, bd=2)
        id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self._attach_keyboard(id_entry)

        # Returning-patient status badge (updates on ID change)
        self.id_status_lbl = tk.Label(left, text="",
                                       font=("Arial", 8),
                                       bg=COLOR_LIGHTEST, fg=COLOR_MID)
        self.id_status_lbl.pack(anchor=tk.W, pady=(0, 6))

        # LEFT — Name
        tk.Label(left, text="Full Name *",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                 anchor=tk.W).pack(fill=tk.X, pady=(0, 2))
        name_entry = tk.Entry(left, textvariable=self.patient_name,
                              font=("Arial", 11), relief=tk.SUNKEN, bd=2)
        name_entry.pack(fill=tk.X, ipady=5, pady=(0, 8))
        self._attach_keyboard(name_entry)

        # LEFT — Age
        tk.Label(left, text="Age *",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                 anchor=tk.W).pack(fill=tk.X, pady=(0, 2))
        age_entry = tk.Entry(left, textvariable=self.patient_age,
                             font=("Arial", 11), relief=tk.SUNKEN, bd=2, width=10)
        age_entry.pack(fill=tk.X, ipady=5, pady=(0, 8))
        self._attach_keyboard(age_entry)

        # LEFT — Sex
        tk.Label(left, text="Sex *",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                 anchor=tk.W).pack(fill=tk.X, pady=(0, 2))
        sex_frame = tk.Frame(left, bg=COLOR_LIGHTEST)
        sex_frame.pack(fill=tk.X, pady=(0, 8))
        for val in ("Male", "Female"):
            tk.Radiobutton(
                sex_frame, text=val,
                variable=self.patient_sex, value=val,
                font=("Arial", 11),
                bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                activebackground=COLOR_LIGHTEST,
                selectcolor=COLOR_PALE_BLUE,
                command=self._on_sex_change
            ).pack(side=tk.LEFT, padx=(0, 16))

        # LEFT — Last Menstrual Cycle (conditional)
        tk.Label(left, text="Last Menstrual Cycle",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                 anchor=tk.W).pack(fill=tk.X, pady=(0, 2))
        self.lmc_entry = tk.Entry(left, textvariable=self.patient_menstrual,
                                   font=("Arial", 11), relief=tk.SUNKEN, bd=2,
                                   state=tk.DISABLED)
        self.lmc_entry.pack(fill=tk.X, ipady=5, pady=(0, 2))
        self._attach_keyboard(self.lmc_entry)
        tk.Label(left, text="MM/DD/YYYY  —  required for Female only",
                 font=("Arial", 8),
                 bg=COLOR_LIGHTEST, fg=COLOR_MID).pack(anchor=tk.W)

        # RIGHT — Fasting
        tk.Label(right, text="Fasting Status *",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                 anchor=tk.W).pack(fill=tk.X, pady=(4, 0))
        tk.Label(right, text="Did not eat for the last 8 hours?",
                 font=("Arial", 9),
                 bg=COLOR_LIGHTEST, fg=COLOR_MID,
                 anchor=tk.W).pack(fill=tk.X, pady=(0, 4))
        fast_frame = tk.Frame(right, bg=COLOR_LIGHTEST)
        fast_frame.pack(fill=tk.X, pady=(0, 12))
        for val in ("Yes", "No"):
            tk.Radiobutton(
                fast_frame, text=val,
                variable=self.patient_fasting, value=val,
                font=("Arial", 11),
                bg=COLOR_LIGHTEST, fg=COLOR_TEXT_DARK,
                activebackground=COLOR_LIGHTEST,
                selectcolor=COLOR_PALE_BLUE,
                command=self._validate_patient_form
            ).pack(side=tk.LEFT, padx=(0, 16))

        # RIGHT — Consent box
        consent_frame = tk.Frame(right, bg=COLOR_PALE_BLUE, padx=10, pady=10)
        consent_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(consent_frame, text="Informed Consent",
                 font=("Arial", 10, "bold"),
                 bg=COLOR_PALE_BLUE, fg=COLOR_TEXT_DARK).pack(anchor=tk.W)
        tk.Label(consent_frame,
                 text="I consent to the analysis of my\nfingernail image for health screening.",
                 font=("Arial", 9),
                 bg=COLOR_PALE_BLUE, fg=COLOR_MID,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 6))
        tk.Checkbutton(
            consent_frame, text="I Consent *",
            variable=self.patient_consent,
            font=("Arial", 11, "bold"),
            bg=COLOR_PALE_BLUE, fg=COLOR_BLUE,
            activebackground=COLOR_PALE_BLUE,
            selectcolor=COLOR_PALE_BLUE,
            command=self._validate_patient_form
        ).pack(anchor=tk.W)

        tk.Label(right, text="* Required  |  ID alone sufficient for returning patients",
                 font=("Arial", 8),
                 bg=COLOR_LIGHTEST, fg=COLOR_MID).pack(anchor=tk.W)

        # Footer buttons
        pi_footer = tk.Frame(self.patient_info_page, bg=COLOR_LIGHT, pady=10)
        pi_footer.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(
            pi_footer, text="◀ BACK",
            font=("Arial", 11, "bold"),
            bg=COLOR_MID, fg="white",
            activebackground=COLOR_TEXT_DARK,
            relief=tk.RAISED, bd=3,
            padx=16, pady=6,
            command=self.show_homepage
        ).pack(side=tk.LEFT, padx=12)

        self.next_btn = tk.Button(
            pi_footer, text="NEXT ▶",
            font=("Arial", 11, "bold"),
            bg=COLOR_MID, fg="white",
            activebackground=COLOR_TEXT_DARK,
            relief=tk.RAISED, bd=3,
            padx=16, pady=6,
            state=tk.DISABLED,
            command=self.show_preview_page
        )
        self.next_btn.pack(side=tk.RIGHT, padx=12)

        # Trace fields for live validation
        self.patient_id.trace_add("write",   lambda *_: self._on_id_change())
        self.patient_name.trace_add("write", lambda *_: self._validate_patient_form())
        self.patient_age.trace_add("write",  lambda *_: self._validate_patient_form())

        # Build on-screen keyboard (must be after root is configured)
        self._build_osk()

        # ==========================================
        # PAGE 2: PREVIEW PAGE  (original layout preserved)
        # ==========================================

        self.preview_page = tk.Frame(self.root, bg=COLOR_LIGHTEST)

        tk.Label(
            self.preview_page,
            text="AINAILSYS",
            font=("Arial", 30, "bold"),
            bg=COLOR_LIGHTEST,
            fg=COLOR_BLUE
        ).pack(pady=5)

        tk.Label(
            self.preview_page,
            text="Position fingernail clearly in camera view",
            font=("Arial", 10),
            bg=COLOR_LIGHTEST,
            fg=COLOR_TEXT_DARK
        ).pack(pady=2)

        preview_frame = tk.Frame(self.preview_page, bg=COLOR_LIGHT,
                                  relief=tk.RAISED, bd=2)
        preview_frame.pack(pady=10, padx=20)

        self.camera_label = tk.Label(preview_frame, bg=COLOR_LIGHT)
        self.camera_label.pack(padx=3, pady=3)

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
        self.capture_btn.pack(pady=8)

        # Back to patient info
        tk.Button(
            self.preview_page,
            text="◀ BACK",
            font=("Arial", 10, "bold"),
            bg=COLOR_MID, fg="white",
            activebackground=COLOR_TEXT_DARK,
            relief=tk.RAISED, bd=3,
            padx=12, pady=4,
            command=self.show_patient_info_page
        ).place(relx=0.0, rely=1.0, x=16, y=-16, anchor="sw")

        # ==========================================
        # PAGE 3: RESULTS PAGE  (original layout + SAVE)
        # ==========================================

        self.results_page = tk.Frame(self.root, bg=COLOR_LIGHTEST)

        tk.Label(
            self.results_page,
            text="AINAILSYS",
            font=("Arial", 30, "bold"),
            bg=COLOR_LIGHTEST,
            fg=COLOR_BLUE
        ).pack(pady=8)

        results_container = tk.Frame(self.results_page, bg=COLOR_LIGHT,
                                      relief=tk.RAISED, bd=3)
        results_container.pack(pady=6, padx=30, fill=tk.BOTH, expand=True)

        tk.Label(
            results_container,
            text="RESULTS",
            font=("Arial", 16, "bold"),
            bg=COLOR_LIGHT,
            fg=COLOR_BLUE
        ).pack(pady=8)

        self.results_content = tk.Frame(results_container, bg=COLOR_LIGHT)
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # Save status label (inside container, above buttons)
        self.save_status_lbl = tk.Label(
            results_container, text="",
            font=("Arial", 9),
            bg=COLOR_LIGHT, fg=COLOR_SUCCESS
        )
        self.save_status_lbl.pack(pady=(0, 4))

        # Bottom button row — DONE + RE-SCAN + SAVE
        btn_row = tk.Frame(self.results_page, bg=COLOR_LIGHTEST)
        btn_row.pack(pady=8)

        # DONE → clears data and returns to homepage
        tk.Button(
            btn_row,
            text="DONE",
            font=("Arial", 16, "bold"),
            bg=COLOR_BLUE,
            fg="white",
            activebackground=COLOR_PALE_BLUE,
            activeforeground=COLOR_TEXT_DARK,
            command=self._done,
            height=2,
            width=10,
            relief=tk.RAISED,
            bd=4
        ).pack(side=tk.LEFT, padx=6)

        # RE-SCAN → back to preview, keeps patient data
        tk.Button(
            btn_row,
            text="RE-SCAN",
            font=("Arial", 16, "bold"),
            bg=COLOR_MID,
            fg="white",
            activebackground=COLOR_TEXT_DARK,
            command=self._re_scan,
            height=2,
            width=10,
            relief=tk.RAISED,
            bd=4
        ).pack(side=tk.LEFT, padx=6)

        # SAVE button
        self.save_btn = tk.Button(
            btn_row,
            text="SAVE",
            font=("Arial", 16, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            activebackground="#00a07e",
            command=self._save_record,
            height=2,
            width=10,
            relief=tk.RAISED,
            bd=4
        )
        self.save_btn.pack(side=tk.LEFT, padx=6)

        # POWER OFF — same position as original (bottom-right)
        power_off_btn = tk.Button(
            self.results_page,
            text="POWER OFF",
            font=("Arial", 10, "bold"),
            bg=COLOR_WARNING,
            fg="white",
            activebackground="#c0392b",
            command=self.shutdown_system,
            height=1,
            width=10,
            relief=tk.RAISED,
            bd=3
        )
        power_off_btn.place(relx=1.0, rely=1.0, x=-20, y=-20, anchor="se")

    # ==========================================
    # PATIENT FORM LOGIC
    # ==========================================

    def _on_id_change(self):
        """Real-time ID lookup — auto-fill if returning patient."""
        pid = self.patient_id.get().strip()
        if not pid:
            self.is_returning_patient = False
            self.id_status_lbl.configure(text="", fg=COLOR_MID)
            self._validate_patient_form()
            return

        record = lookup_patient_by_id(pid)
        if record:
            self.is_returning_patient = True
            name = record.get("Name", "")
            self.id_status_lbl.configure(
                text=f"  Returning patient: {name}  —  info loaded automatically",
                fg=COLOR_SUCCESS)
            self._populate_from_existing(record)
        else:
            self.is_returning_patient = False
            self.id_status_lbl.configure(
                text="  New patient — please fill all fields",
                fg=COLOR_MID)
        self._validate_patient_form()

    def _on_sex_change(self):
        if self.patient_sex.get() == "Female":
            self.lmc_entry.configure(state=tk.NORMAL)
        else:
            self.patient_menstrual.set("")
            self.lmc_entry.configure(state=tk.DISABLED)
        self._validate_patient_form()

    def _validate_patient_form(self):
        pid     = self.patient_id.get().strip()
        consent = self.patient_consent.get()

        if self.is_returning_patient:
            # Returning patient: ID already looked up, just need consent
            valid = bool(pid) and consent
        else:
            # New patient: all fields required
            name    = self.patient_name.get().strip()
            age_str = self.patient_age.get().strip()
            sex     = self.patient_sex.get()
            fasting = self.patient_fasting.get()
            age_ok  = age_str.isdigit() and 0 < int(age_str) < 130
            valid   = (bool(pid) and bool(name) and age_ok and
                       sex in ("Male", "Female") and
                       fasting in ("Yes", "No") and consent)

        if valid:
            self.next_btn.configure(state=tk.NORMAL, bg=COLOR_SUCCESS,
                                    activebackground="#00a07e")
        else:
            self.next_btn.configure(state=tk.DISABLED, bg=COLOR_MID,
                                    activebackground=COLOR_TEXT_DARK)

    # ==========================================
    # CAMERA
    # ==========================================

    def update_preview(self):
        """Update camera preview — runs every 30 ms"""
        if self.on_preview_page and not self.is_analyzing:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    display   = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                    frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    img       = Image.fromarray(frame_rgb)
                    imgtk     = ImageTk.PhotoImage(image=img)
                    self.camera_label.imgtk = imgtk
                    self.camera_label.configure(image=imgtk)
        self.root.after(30, self.update_preview)

    # ==========================================
    # CAPTURE & ANALYZE
    # ==========================================

    def capture_and_analyze(self):
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

            has_nail = detect_nail_presence(self.current_frame)

            if not has_nail:
                print("\n NO NAIL DETECTED")
                self.last_results = None
                self.display_no_nail_result()
            else:
                print("\nRunning AI classification...")
                results = self.analyze_image(self.current_frame)
                self.last_results = results

                stage1_confidence = results['stage1']['confidence']
                prediction        = results['stage1']['prediction']

                print(f"\nSTAGE 1 RESULTS:")
                print(f"  Prediction: {prediction.upper()}")
                print(f"  Confidence: {stage1_confidence:.2%}")

                if prediction == 'anemic' and 'stage2' in results:
                    stage2 = results['stage2']
                    print(f"\nSTAGE 2 RESULTS:")
                    print(f"  Abnormality: {stage2['abnormality']}")
                    print(f"  Deficiency:  {stage2['deficiency']}")
                    print(f"  Confidence:  {stage2['confidence']:.2%}")

                self.display_results(results)

            print("="*50 + "\n")

        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()
            self.last_results = None
            self.display_error_result(str(e))

        finally:
            self.is_analyzing = False
            self.capture_btn.config(state=tk.NORMAL, text="CAPTURE & ANALYZE")
            self.show_results_page()

    def analyze_image(self, image):
        image_tensor  = preprocess_image(image)
        stage1_result = self.run_stage1(image_tensor)
        result        = {'stage1': stage1_result}
        if stage1_result['prediction'] == 'anemic':
            stage2_result  = self.run_stage2(image_tensor)
            result['stage2'] = stage2_result
        return result

    def run_stage1(self, image_tensor):
        input_name  = self.stage1_session.get_inputs()[0].name
        output_name = self.stage1_session.get_outputs()[0].name
        outputs     = self.stage1_session.run([output_name], {input_name: image_tensor})
        logits      = outputs[0][0]
        probs       = np.exp(logits) / np.sum(np.exp(logits))
        pred_class  = np.argmax(probs)
        class_names = self.stage1_meta['class_names']
        return {
            'prediction':    class_names[pred_class],
            'confidence':    float(probs[pred_class]),
            'probabilities': {n: float(p) for n, p in zip(class_names, probs)}
        }

    def run_stage2(self, image_tensor):
        input_name  = self.stage2_session.get_inputs()[0].name
        output_name = self.stage2_session.get_outputs()[0].name
        outputs     = self.stage2_session.run([output_name], {input_name: image_tensor})
        logits      = outputs[0][0]
        probs       = np.exp(logits) / np.sum(np.exp(logits))
        pred_class  = np.argmax(probs)
        abnormality = self.stage2_meta['class_names'][pred_class]
        return {
            'abnormality': abnormality,
            'deficiency':  DEFICIENCY_MAP[abnormality],
            'confidence':  float(probs[pred_class])
        }

    # ==========================================
    # DISPLAY RESULTS  (clears results_content like original)
    # ==========================================

    def _prep_results_display(self):
        """Clear results_content and reset save UI before showing new result."""
        for w in self.results_content.winfo_children():
            w.destroy()
        self.save_status_lbl.configure(text="")
        self.save_btn.configure(
            state=tk.NORMAL, bg=COLOR_SUCCESS,
            activebackground="#00a07e", text="SAVE"
        )

    def display_no_nail_result(self):
        self._prep_results_display()

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

        self.speak("No nail detected. Please position fingernail and try again.")

    def display_error_result(self, error):
        self._prep_results_display()

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
        self._prep_results_display()

        stage1 = results['stage1']

        if stage1['prediction'] == 'healthy':
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

            self.speak("Healthy Nail. No signs of anemia detected.")

        else:
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

                abnormality = stage2['abnormality'].replace('_', ' ')
                deficiency  = stage2['deficiency']
                self.speak(f"Anemic {abnormality} nail detected. {deficiency} deficiency.")

            tk.Label(
                self.results_content,
                text="Consult healthcare provider\nfor diagnosis",
                font=("Arial", 9, "italic"),
                bg=COLOR_LIGHT,
                fg=COLOR_TEXT_DARK,
                justify=tk.CENTER
            ).pack(pady=8)

    # ==========================================
    # SAVE TO CSV
    # ==========================================

    def _save_record(self):
        if self.last_results is None:
            self.save_status_lbl.configure(
                text="Nothing to save (no valid analysis).", fg=COLOR_WARNING)
            return

        stage1     = self.last_results.get('stage1', {})
        stage2     = self.last_results.get('stage2')
        prediction = stage1.get('prediction', 'unknown').capitalize()
        conf1      = stage1.get('confidence', 0.0) * 100
        is_anemic  = prediction.lower() == 'anemic'

        abnorm_type = ""
        deficiency  = ""
        conf2_str   = ""
        if is_anemic and stage2:
            abnorm_type = stage2.get('abnormality', '')
            deficiency  = stage2.get('deficiency', '')
            conf2_str   = f"{stage2.get('confidence', 0.0)*100:.2f}%"

        record = {
            "Timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Patient_ID":           self.patient_id.get().strip(),
            "Name":                 self.patient_name.get().strip(),
            "Age":                  self.patient_age.get().strip(),
            "Sex":                  self.patient_sex.get(),
            "Last_Menstrual_Cycle": self.patient_menstrual.get().strip()
                                    if self.patient_sex.get() == "Female" else "",
            "Fasting_Status":       self.patient_fasting.get(),
            "Prediction":           prediction,
            "Abnormality_Type":     abnorm_type.replace('_', ' ').title()
                                    if abnorm_type else "",
            "Deficiency_Type":      deficiency if is_anemic else "",
            "Confidence_Stage1":    f"{conf1:.2f}%",
            "Confidence_Stage2":    conf2_str,
        }

        try:
            append_record(record)
            self.save_status_lbl.configure(
                text="  Record saved to patient_records.csv", fg=COLOR_SUCCESS)
            self.save_btn.configure(
                state=tk.DISABLED, bg=COLOR_MID,
                activebackground=COLOR_MID, text="SAVED"
            )
            self.speak("Record saved successfully.")
            print(f" Record saved: {record}")
        except Exception as e:
            self.save_status_lbl.configure(
                text=f"  Save failed: {e}", fg=COLOR_WARNING)
            self.speak("Error. Record could not be saved.")
            print(f" Save error: {e}")

    # ==========================================
    # DONE → homepage  /  RE-SCAN → preview (keeps patient data)
    # ==========================================

    def _done(self):
        """Finish session — clear all data and return to homepage."""
        self._do_hide_keyboard()
        self._clear_patient_data()
        self.show_homepage()

    def _re_scan(self):
        """Re-scan the same patient — go back to preview without clearing data."""
        self.last_results = None
        self.show_preview_page()

    # ==========================================
    # SHUTDOWN  (exact original pattern)
    # ==========================================

    def shutdown_system(self):
        """Show shutdown confirmation dialog"""
        confirm = tk.Toplevel(self.root)
        confirm.title("Power Off")
        confirm.geometry("450x250")
        confirm.configure(bg=COLOR_LIGHT)
        confirm.transient(self.root)
        confirm.grab_set()

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
        self._destroy_keyboard()   # hide OSK before anything else
        confirm_window.destroy()

        for w in self.results_content.winfo_children():
            w.destroy()

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
        self.speak("Powering off.")

        if self.cap:
            self.cap.release()

        self.root.after(2000, lambda: subprocess.run(['sudo', 'shutdown', 'now']))

    def exit_app(self):
        """Exit — hidden ESC 3×"""
        print("Exiting AINAILSYS...")
        self._destroy_keyboard()
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        finally:
            self.root.destroy()


# ============================================
# MAIN
# ============================================

def main():
    root = tk.Tk()
    app  = AINAILSYSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()