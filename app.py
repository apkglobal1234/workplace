"""
Smart Attendance System
=======================
Face-recognition attendance for a school, using PRE-TRAINED models and CSV files.

  * Face detection   : YuNet  (pre-trained, OpenCV)
  * Face recognition : SFace  (pre-trained, OpenCV)  -> a 128-number "face fingerprint"
  * Storage          : CSV files (open them in Excel)

SETUP (one time)
    pip install opencv-python numpy
    python smart_attendance.py
  The two model files (about 40 MB) download automatically on the first run.

FILES CREATED (inside the "data" folder next to this script)
    students.csv      roll_no, name, class, registered_on
    attendance.csv    date, time, roll_no, name, class, status
    reports/          attendance summary reports (CSV)
    embeddings/       face fingerprints (one .npy file per student)
    photos/           one small photo per student

HOW IT WORKS (for your project report)
    1. REGISTER : the webcam captures 10 pictures of a student. YuNet finds the face,
                  SFace converts each face into 128 numbers. These numbers are saved.
    2. ATTEND   : for every face the camera sees, we compute its 128 numbers and compare
                  them with all saved students using cosine similarity.
                  A score above MATCH_THRESHOLD = same person.
    3. MARK     : a student must be recognised in CONFIRM_FRAMES frames in a row (to avoid
                  mistakes). Then one row is written to attendance.csv - only once per day.
"""

import csv
import datetime
import os
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing libraries. Run this first:\n\n    pip install opencv-python numpy\n")
    sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ----------------------------------------------------------------------------
# 1. SETTINGS  (change these freely)
# ----------------------------------------------------------------------------
SCHOOL_NAME = "Sunrise Public School"
CAMERA_INDEX = 0            # 0 = default webcam, try 1 for an external camera
MATCH_THRESHOLD = 0.45      # higher = stricter (fewer wrong matches, more misses)
MATCH_MARGIN = 0.05         # best match must beat the 2nd best student by this much
CONFIRM_FRAMES = 3          # frames in a row a face must match before marking present
STOP_AFTER_MATCH = True     # True = show popup and stop the camera after each student
POPUP_SECONDS = 5           # the popup closes by itself after this many seconds
SAMPLES_NEEDED = 10         # face pictures captured per student during registration
MIN_FACE_WIDTH = 90         # pixels - faces smaller than this are too far away
REG_COUNTDOWN = 2.0         # seconds of "get ready" before capture starts
REG_INTERVAL = 0.7          # seconds between two captured pictures
LOW_ATTENDANCE_PERCENT = 75 # students below this are flagged in the report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EMB_DIR = DATA_DIR / "embeddings"
PHOTO_DIR = DATA_DIR / "photos"
REPORT_DIR = DATA_DIR / "reports"
STUDENTS_CSV = DATA_DIR / "students.csv"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

STUDENT_FIELDS = ["roll_no", "name", "class", "registered_on"]
ATT_FIELDS = ["date", "time", "roll_no", "name", "class", "status"]

POSES = [
    "Look straight at the camera", "Look straight at the camera",
    "Turn your head slightly LEFT", "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT", "Turn your head slightly RIGHT",
    "Tilt your head UP a little", "Tilt your head DOWN a little",
    "Smile 🙂", "Look straight again",
]

MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (100_000, [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ]),
    "face_recognition_sface_2021dec.onnx": (10_000_000, [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ]),
}


# ----------------------------------------------------------------------------
# 2. CSV STORAGE
# ----------------------------------------------------------------------------
def init_storage():
    for folder in (DATA_DIR, MODEL_DIR, EMB_DIR, PHOTO_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for path, fields in ((STUDENTS_CSV, STUDENT_FIELDS), (ATTENDANCE_CSV, ATT_FIELDS)):
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(fields)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f)]


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def today_str():
    return datetime.date.today().isoformat()


def valid_roll(roll):
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", roll))


# ---- students ----
def load_students():
    return read_csv(STUDENTS_CSV)


def add_student(roll, name, cls):
    if any(s["roll_no"].lower() == roll.lower() for s in load_students()):
        raise ValueError(f"Roll number {roll} already exists.")
    with open(STUDENTS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=STUDENT_FIELDS).writerow(
            {"roll_no": roll, "name": name, "class": cls, "registered_on": today_str()})


def delete_student(roll):
    rows = [s for s in load_students() if s["roll_no"] != roll]
    write_csv(STUDENTS_CSV, STUDENT_FIELDS, rows)
    for path in (EMB_DIR / f"{roll}.npy", PHOTO_DIR / f"{roll}.jpg"):
        if path.exists():
            path.unlink()


# ---- attendance ----
def load_attendance(date=None):
    rows = read_csv(ATTENDANCE_CSV)
    return [r for r in rows if r["date"] == date] if date else rows


def mark_attendance(student, status="Present"):
    """Write one attendance row. Returns 'new', 'updated' or 'exists'."""
    date, when = today_str(), datetime.datetime.now().strftime("%H:%M:%S")
    rows = read_csv(ATTENDANCE_CSV)
    for r in rows:
        if r["date"] == date and r["roll_no"] == student["roll_no"]:
            if r["status"] == "Absent" and status == "Present":   # came late, after absentees were marked
                r["time"], r["status"] = when, "Present"
                write_csv(ATTENDANCE_CSV, ATT_FIELDS, rows)
                return "updated"
            return "exists"
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=ATT_FIELDS).writerow({
            "date": date, "time": when, "roll_no": student["roll_no"],
            "name": student["name"], "class": student["class"], "status": status})
    return "new"


def mark_absentees():
    """Add an 'Absent' row for every student who has no record today. Returns the count."""
    date = today_str()
    rows = read_csv(ATTENDANCE_CSV)
    done = {r["roll_no"] for r in rows if r["date"] == date}
    added = 0
    for s in load_students():
        if s["roll_no"] not in done:
            rows.append({"date": date, "time": "", "roll_no": s["roll_no"], "name": s["name"],
                         "class": s["class"], "status": "Absent"})
            added += 1
    if added:
        write_csv(ATTENDANCE_CSV, ATT_FIELDS, rows)
    return added


def make_summary_report():
    """Create a CSV with every student's attendance percentage. Returns the file path."""
    rows = read_csv(ATTENDANCE_CSV)
    total_days = len({r["date"] for r in rows})
    present = defaultdict(int)
    for r in rows:
        if r["status"] == "Present":
            present[r["roll_no"]] += 1
    out = []
    for s in load_students():
        days = present[s["roll_no"]]
        percent = round(100 * days / total_days, 1) if total_days else 0.0
        out.append({"roll_no": s["roll_no"], "name": s["name"], "class": s["class"],
                    "days_present": days, "total_days": total_days, "percentage": percent,
                    "remark": "Low attendance" if total_days and percent < LOW_ATTENDANCE_PERCENT else ""})
    path = REPORT_DIR / f"attendance_summary_{today_str()}.csv"
    write_csv(path, ["roll_no", "name", "class", "days_present", "total_days", "percentage", "remark"], out)
    return path


# ----------------------------------------------------------------------------
# 3. PRE-TRAINED FACE MODELS (auto-download) AND FACE ENGINE
# ----------------------------------------------------------------------------
def ensure_models():
    """Download the two ONNX models the first time. Returns (yunet_path, sface_path)."""
    import urllib.request
    paths = []
    for filename, (min_size, urls) in MODEL_FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size >= min_size:
            paths.append(dest)
            continue
        print(f"Downloading {filename} (first run only)...")
        ok = False
        for url in urls:
            try:
                tmp = dest.with_suffix(".part")
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as out:
                    shutil.copyfileobj(resp, out)
                if tmp.stat().st_size < min_size:
                    raise IOError("downloaded file is too small")
                tmp.replace(dest)
                ok = True
                break
            except Exception as e:
                print(f"  failed from {url.split('/')[2]}: {e}")
        if not ok:
            links = "\n".join(f"  {u}" for u in urls)
            raise RuntimeError(
                f"Could not download {filename}.\n\nDownload it manually from either link:\n{links}\n\n"
                f"and put it in this folder:\n  {MODEL_DIR}")
        paths.append(dest)
    return paths


class FaceEngine:
    """Wraps the two pre-trained OpenCV models."""

    def __init__(self):
        yunet, sface = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.85, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect(self, frame):
        """Return a list of faces. Each face: x, y, w, h, 10 landmark values, score."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)

    def embed(self, frame, face):
        """Return the 128-number 'fingerprint' of a face (normalised)."""
        aligned = self.recognizer.alignCrop(frame, face)
        vec = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class FaceDB:
    """Holds every student's saved face fingerprints and finds the best match."""

    def __init__(self):
        self.reload()

    def reload(self):
        self.students = {s["roll_no"]: s for s in load_students()}
        mats, owners = [], []
        for roll in self.students:
            path = EMB_DIR / f"{roll}.npy"
            if path.exists():
                emb = np.load(path)
                mats.append(emb)
                owners += [roll] * len(emb)
        self.matrix = np.vstack(mats).astype(np.float32) if mats else np.empty((0, 128), np.float32)
        self.owners = owners

    def identify(self, emb):
        """Return (roll_no or None, best_score)."""
        if not self.owners:
            return None, 0.0
        scores = self.matrix @ emb                     # cosine similarity (vectors are normalised)
        best = {}
        for owner, sc in zip(self.owners, scores):
            if sc > best.get(owner, -1.0):
                best[owner] = float(sc)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        roll, score = ranked[0]
        if score < MATCH_THRESHOLD:
            return None, score
        if len(ranked) > 1 and score - ranked[1][1] < MATCH_MARGIN:
            return None, score                         # too close to another student: don't guess
        return roll, score


def beep():
    def _b():
        try:
            import winsound
            winsound.Beep(1200, 120)
        except Exception:
            pass
    threading.Thread(target=_b, daemon=True).start()


# ----------------------------------------------------------------------------
# 4. GUI
# ----------------------------------------------------------------------------
PRIMARY, PRIMARY_DARK = "#2f5bea", "#2447bd"
BG, CARD = "#eef1f8", "#ffffff"
GREEN, RED, GREY = "#1e9e55", "#d64545", "#6b7385"
FONT, FONT_B = ("Segoe UI", 10), ("Segoe UI", 10, "bold")


def make_button(parent, text, command, kind="primary", **kw):
    colors = {"primary": (PRIMARY, "white", PRIMARY_DARK), "danger": (RED, "white", "#b03030"),
              "success": (GREEN, "white", "#167a42"), "light": ("#e3e8f6", PRIMARY, "#d3dbf2")}
    bg, fg, active = colors[kind]
    return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=active,
                     activeforeground=fg, relief="flat", borderwidth=0, cursor="hand2",
                     font=FONT_B, padx=14, pady=8, **kw)


class App:
    def __init__(self, root, engine, db):
        self.root, self.engine, self.db = root, engine, db
        self.cap = None
        self.mode = None            # None | "attend" | "register"
        self.loop_running = False
        self.frame_no = 0
        self.results = []           # last recognised faces, for drawing
        self.streak = {}            # roll_no -> frames in a row recognised
        self.today_marked = set()
        self.reg = None             # current registration session
        self.popup = None           # the "Present" popup window
        self.popup_after = None
        self.stop_var = tk.BooleanVar(value=STOP_AFTER_MATCH)

        root.title(f"Smart Attendance System - {SCHOOL_NAME}")
        root.configure(bg=BG)
        # size the window from the real screen size and display scaling, so it always fits
        self.scale = root.winfo_fpixels("1i") / 96.0
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(int(1150 * self.scale), sw - 40), min(int(760 * self.scale), sh - 90)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
        root.minsize(min(int(900 * self.scale), w), min(int(600 * self.scale), h))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=FONT_B)
        style.configure("Treeview", rowheight=26, font=FONT)
        style.configure("Treeview.Heading", font=FONT_B)
        style.configure("Blue.Horizontal.TProgressbar", troughcolor="#dde3f3", background=PRIMARY)

        self.build_header()
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        self.tab_att, self.tab_reg = tk.Frame(self.nb, bg=BG), tk.Frame(self.nb, bg=BG)
        self.tab_rec, self.tab_stu = tk.Frame(self.nb, bg=BG), tk.Frame(self.nb, bg=BG)
        self.nb.add(self.tab_att, text="  📷 Take Attendance  ")
        self.nb.add(self.tab_reg, text="  ➕ Register Student  ")
        self.nb.add(self.tab_rec, text="  📋 Records  ")
        self.nb.add(self.tab_stu, text="  👥 Students  ")
        self.build_attendance_tab()
        self.build_register_tab()
        self.build_records_tab()
        self.build_students_tab()
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_all()
        self.tick()

    # ---------- header ----------
    def build_header(self):
        bar = tk.Frame(self.root, bg=PRIMARY)
        bar.pack(fill="x")
        tk.Label(bar, text="🎓", bg=PRIMARY, fg="white", font=("Segoe UI Emoji", 22)).pack(
            side="left", padx=(16, 8), pady=10)
        box = tk.Frame(bar, bg=PRIMARY)
        box.pack(side="left", pady=10)
        tk.Label(box, text="Smart Attendance System", bg=PRIMARY, fg="white",
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(box, text=f"{SCHOOL_NAME} • Face Recognition", bg=PRIMARY, fg="#d6e0ff",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.clock = tk.Label(bar, text="", bg=PRIMARY, fg="white", font=("Segoe UI", 11, "bold"))
        self.clock.pack(side="right", padx=18)

    def tick(self):
        self.clock.config(text=datetime.datetime.now().strftime("%a, %d %b %Y   %I:%M:%S %p"))
        self.root.after(1000, self.tick)

    def make_view(self, parent, text):
        """A camera area that fills the free space (never forces the window to grow)."""
        box = tk.Frame(parent, bg="#1b2030", width=320, height=240)
        box.pack_propagate(False)
        box.pack(fill="both", expand=True)
        label = tk.Label(box, text=text, bg="#1b2030", fg="#9aa3bd", font=("Segoe UI", 13))
        label.pack(fill="both", expand=True)
        return label

    # ---------- tab: attendance ----------
    def build_attendance_tab(self):
        left = tk.Frame(self.tab_att, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)
        row = tk.Frame(left, bg=BG)
        row.pack(side="bottom", fill="x", pady=(10, 0))
        self.att_view = self.make_view(left, "📷  Camera is off\n\nClick  ▶ Start Attendance")
        make_button(row, "▶  Start Attendance", self.start_attendance, "success").pack(side="left")
        make_button(row, "⏹  Stop", self.stop_camera, "light").pack(side="left", padx=8)
        tk.Checkbutton(row, text="Stop after each student", variable=self.stop_var, bg=BG,
                       activebackground=BG, font=FONT, cursor="hand2").pack(side="left", padx=6)
        self.att_status = tk.Label(row, text="", bg=BG, fg=GREY, font=FONT, anchor="w")
        self.att_status.pack(side="left", padx=10)

        right = tk.Frame(self.tab_att, bg=BG, width=int(330 * self.scale))
        right.pack(side="right", fill="y", padx=(6, 12), pady=12)
        right.pack_propagate(False)
        card = tk.Frame(right, bg=CARD)
        card.pack(fill="x")
        tk.Label(card, text="Present today", bg=CARD, fg=GREY, font=FONT).pack(pady=(14, 0))
        self.count_label = tk.Label(card, text="0 / 0", bg=CARD, fg=PRIMARY, font=("Segoe UI", 30, "bold"))
        self.count_label.pack(pady=(0, 14))
        tk.Label(right, text="Marked present", bg=BG, fg=GREY, font=FONT_B).pack(anchor="w", pady=(14, 4))
        self.marked_tree = ttk.Treeview(right, columns=("time", "roll", "name"), show="headings", height=12)
        for col, text, w in (("time", "Time", 70), ("roll", "Roll", 60), ("name", "Name", 150)):
            self.marked_tree.heading(col, text=text)
            self.marked_tree.column(col, width=w, anchor="w")
        self.marked_tree.pack(fill="both", expand=True)
        make_button(right, "Mark absentees (end of day)", self.do_mark_absentees, "danger").pack(
            fill="x", pady=(10, 0))

    # ---------- tab: register ----------
    def build_register_tab(self):
        left = tk.Frame(self.tab_reg, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)
        self.progress = ttk.Progressbar(left, style="Blue.Horizontal.TProgressbar", maximum=SAMPLES_NEEDED)
        self.progress.pack(side="bottom", fill="x", pady=(2, 0))
        self.reg_prompt = tk.Label(left, text="Fill the form, then click Start Face Capture", bg=BG,
                                   fg=PRIMARY, font=("Segoe UI", 14, "bold"))
        self.reg_prompt.pack(side="bottom", pady=(10, 2))
        self.reg_view = self.make_view(left, "📷  Camera preview appears here")

        right = tk.Frame(self.tab_reg, bg=CARD, width=int(340 * self.scale))
        right.pack(side="right", fill="y", padx=(6, 12), pady=12)
        right.pack_propagate(False)
        tk.Label(right, text="New student", bg=CARD, fg="#1f2430", font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=18, pady=(18, 8))
        self.entries = {}
        for key, label in (("roll", "Roll number"), ("name", "Full name"), ("cls", "Class (e.g. 10-A)")):
            tk.Label(right, text=label, bg=CARD, fg=GREY, font=FONT).pack(anchor="w", padx=18, pady=(8, 2))
            e = tk.Entry(right, font=("Segoe UI", 12), relief="solid", borderwidth=1)
            e.pack(fill="x", padx=18, ipady=5)
            self.entries[key] = e
        make_button(right, "📸  Start Face Capture", self.start_capture, "primary").pack(
            fill="x", padx=18, pady=(22, 6))
        make_button(right, "🖼  Add from photo files", self.register_from_photos, "light").pack(
            fill="x", padx=18, pady=6)
        make_button(right, "Cancel", self.cancel_capture, "light").pack(fill="x", padx=18, pady=6)
        tk.Label(right, text="Tips: good light on the face, no cap or mask,\nonly one person in front of the camera.",
                 bg=CARD, fg=GREY, font=("Segoe UI", 9), justify="left").pack(anchor="w", padx=18, pady=14)

    # ---------- tab: records ----------
    def build_records_tab(self):
        top = tk.Frame(self.tab_rec, bg=BG)
        top.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(top, text="Date (YYYY-MM-DD):", bg=BG, font=FONT_B).pack(side="left")
        self.date_entry = tk.Entry(top, font=("Segoe UI", 11), width=12, relief="solid", borderwidth=1)
        self.date_entry.pack(side="left", padx=8, ipady=3)
        self.date_entry.insert(0, today_str())
        make_button(top, "Show", self.refresh_records, "primary").pack(side="left")
        make_button(top, "Today", self.records_today, "light").pack(side="left", padx=6)
        make_button(top, "📊 Export summary report", self.export_report, "success").pack(side="right")
        make_button(top, "📂 Open data folder", self.open_folder, "light").pack(side="right", padx=6)

        self.rec_summary = tk.Label(self.tab_rec, text="", bg=BG, fg=GREY, font=FONT_B, anchor="w")
        self.rec_summary.pack(fill="x", padx=14)
        frame = tk.Frame(self.tab_rec, bg=BG)
        frame.pack(fill="both", expand=True, padx=12, pady=8)
        self.rec_tree = ttk.Treeview(frame, columns=("time", "roll", "name", "cls", "status"),
                                     show="headings")
        for col, text, w in (("time", "Time", 100), ("roll", "Roll no", 100), ("name", "Name", 280),
                             ("cls", "Class", 100), ("status", "Status", 110)):
            self.rec_tree.heading(col, text=text)
            self.rec_tree.column(col, width=w, anchor="w")
        self.rec_tree.tag_configure("Present", foreground=GREEN)
        self.rec_tree.tag_configure("Absent", foreground=RED)
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.rec_tree.pack(fill="both", expand=True)

    # ---------- tab: students ----------
    def build_students_tab(self):
        top = tk.Frame(self.tab_stu, bg=BG)
        top.pack(fill="x", padx=12, pady=(12, 6))
        self.stu_count = tk.Label(top, text="", bg=BG, fg=GREY, font=FONT_B)
        self.stu_count.pack(side="left")
        make_button(top, "🗑 Delete selected student", self.delete_selected, "danger").pack(side="right")
        frame = tk.Frame(self.tab_stu, bg=BG)
        frame.pack(fill="both", expand=True, padx=12, pady=8)
        self.stu_tree = ttk.Treeview(frame, columns=("roll", "name", "cls", "date", "face"), show="headings")
        for col, text, w in (("roll", "Roll no", 100), ("name", "Name", 280), ("cls", "Class", 100),
                             ("date", "Registered on", 130), ("face", "Face data", 100)):
            self.stu_tree.heading(col, text=text)
            self.stu_tree.column(col, width=w, anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.stu_tree.yview)
        self.stu_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.stu_tree.pack(fill="both", expand=True)

    # ---------- refreshing tables ----------
    def refresh_all(self):
        self.refresh_students()
        self.refresh_marked()
        self.refresh_records()

    def refresh_students(self):
        self.stu_tree.delete(*self.stu_tree.get_children())
        students = load_students()
        for s in students:
            has_face = "✔" if (EMB_DIR / f"{s['roll_no']}.npy").exists() else "✘ missing"
            self.stu_tree.insert("", "end", values=(s["roll_no"], s["name"], s["class"],
                                                    s["registered_on"], has_face))
        self.stu_count.config(text=f"{len(students)} students registered")

    def refresh_marked(self):
        self.marked_tree.delete(*self.marked_tree.get_children())
        current = {s["roll_no"] for s in load_students()}      # ignore students who were deleted
        rows = [r for r in load_attendance(today_str())
                if r["status"] == "Present" and r["roll_no"] in current]
        self.today_marked = {r["roll_no"] for r in rows}
        for r in sorted(rows, key=lambda r: r["time"], reverse=True):
            self.marked_tree.insert("", "end", values=(r["time"][:5], r["roll_no"], r["name"]))
        self.count_label.config(text=f"{len(rows)} / {len(load_students())}")

    def refresh_records(self):
        date = self.date_entry.get().strip()
        try:
            datetime.date.fromisoformat(date)
        except ValueError:
            messagebox.showwarning("Date", "Please type the date like 2026-10-01")
            return
        rows = load_attendance(date)
        self.rec_tree.delete(*self.rec_tree.get_children())
        for r in sorted(rows, key=lambda r: (r["status"] != "Present", r["name"])):
            self.rec_tree.insert("", "end", tags=(r["status"],),
                                 values=(r["time"], r["roll_no"], r["name"], r["class"], r["status"]))
        present = sum(r["status"] == "Present" for r in rows)
        absent = sum(r["status"] == "Absent" for r in rows)
        self.rec_summary.config(text=f"{date}   •   Present: {present}   •   Absent: {absent}   •   "
                                     f"Total students: {len(load_students())}")

    def records_today(self):
        self.date_entry.delete(0, "end")
        self.date_entry.insert(0, today_str())
        self.refresh_records()

    # ---------- camera ----------
    def open_camera(self):
        if self.cap is not None:
            return True
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Camera", "Could not open the camera.\n\n• Is it connected?\n"
                                           "• Is another app (Zoom, Teams) using it?\n"
                                           "• Windows Settings → Privacy → Camera → allow apps.")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap = cap
        return True

    def stop_camera(self):
        self.mode = None
        self.reg = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.results, self.streak = [], {}
        for view, text in ((self.att_view, "📷  Camera is off\n\nClick  ▶ Start Attendance"),
                           (self.reg_view, "📷  Camera preview appears here")):
            view.configure(image="", text=text)
            view.image = None
        self.att_status.config(text="")

    def ensure_loop(self):
        if not self.loop_running:
            self.loop_running = True
            self.loop()

    def loop(self):
        if self.cap is None or self.mode is None:
            self.loop_running = False
            return
        ok, frame = self.cap.read()
        if ok:
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)          # mirror view feels natural
            try:
                if self.mode == "attend":
                    self.process_attendance(frame)
                else:
                    self.process_register(frame)
            except Exception as e:              # never let one bad frame kill the camera loop
                print("Frame error:", e)
        self.root.after(10, self.loop)

    def show(self, frame, view):
        cw, ch = view.winfo_width(), view.winfo_height()
        if cw > 50 and ch > 50:
            h, w = frame.shape[:2]
            f = min(cw / w, ch / h, 1.5)
            if abs(f - 1) > 0.02:
                frame = cv2.resize(frame, (max(1, int(w * f)), max(1, int(h * f))))
        data = cv2.imencode(".ppm", frame)[1].tobytes()
        img = tk.PhotoImage(data=data)
        view.configure(image=img, text="")
        view.image = img

    @staticmethod
    def label_text(student):
        return student["name"] if student["name"].isascii() else student["roll_no"]

    # ---------- attendance mode ----------
    def start_attendance(self):
        if not self.db.owners:
            messagebox.showinfo("No students", "Register at least one student first.")
            return
        if not self.open_camera():
            return
        self.mode = "attend"
        self.streak = {}
        self.att_status.config(text="Looking for faces...", fg=GREY)
        self.ensure_loop()

    def process_attendance(self, frame):
        self.frame_no += 1
        if self.frame_no % 2 == 0:              # recognise every 2nd frame to stay smooth
            seen, results, pending = set(), [], None
            for face in self.engine.detect(frame)[:5]:
                box = [int(v) for v in face[:4]]
                emb = self.engine.embed(frame, face)
                roll, score = self.db.identify(emb)
                if roll:
                    seen.add(roll)
                    self.streak[roll] = self.streak.get(roll, 0) + 1
                    student = self.db.students[roll]
                    results.append((box, f"{self.label_text(student)} {score:.2f}", (60, 200, 90)))
                    if self.streak[roll] >= CONFIRM_FRAMES and pending is None:
                        pending = student
                else:
                    results.append((box, "Unknown", (70, 70, 230)))
            for roll in list(self.streak):
                if roll not in seen:
                    self.streak[roll] = 0
            self.results = results
            if pending is not None:
                self.handle_match(pending)
                if self.mode is None:              # camera was stopped by handle_match
                    return
        for (x, y, w, h), text, color in self.results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 24), (x + max(w, 120), y), color, -1)
            cv2.putText(frame, text, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        self.show(frame, self.att_view)

    def mark_present(self, student):
        """Save Present to the CSV. Returns 'new', 'updated', 'exists' - or None if saving failed."""
        try:
            result = mark_attendance(student, "Present")
        except PermissionError:
            self.att_status.config(text="⚠ Close attendance.csv in Excel - cannot save!", fg=RED)
            return None
        self.today_marked.add(student["roll_no"])
        if result in ("new", "updated"):
            beep()
            self.att_status.config(text=f"✔ {student['name']} marked present", fg=GREEN)
            self.refresh_marked()
            self.refresh_records()
        return result

    def handle_match(self, student):
        """A face was confirmed: save attendance, show the popup, and (optionally) stop the camera."""
        stop = self.stop_var.get()
        roll = student["roll_no"]
        if roll in self.today_marked:
            if not stop:
                return                              # continuous mode: stay quiet for repeat faces
            result = "exists"
        else:
            result = self.mark_present(student)
            if result is None:
                return
        when = datetime.datetime.now().strftime("%I:%M:%S %p")
        if result == "exists":
            for r in load_attendance(today_str()):
                if r["roll_no"] == roll and r["time"]:
                    when = r["time"]
        if stop:
            self.stop_camera()
        self.show_popup(student, result, when, restart=stop)

    # ---------- the "Present" popup ----------
    def show_popup(self, student, result, when, restart):
        self.close_popup()
        ok = result in ("new", "updated")
        color = GREEN if ok else "#d98f00"
        top = tk.Toplevel(self.root)
        self.popup = top
        top.title("Attendance")
        top.configure(bg=CARD)
        top.transient(self.root)
        top.resizable(False, False)
        try:
            top.attributes("-topmost", True)
        except tk.TclError:
            pass

        band = tk.Frame(top, bg=color)
        band.pack(fill="x")
        tk.Label(band, text="✔  PRESENT" if ok else "ℹ  ALREADY PRESENT", bg=color, fg="white",
                 font=("Segoe UI", 22, "bold"), padx=60, pady=14).pack()
        body = tk.Frame(top, bg=CARD)
        body.pack(padx=40, pady=(18, 8))

        photo_path = PHOTO_DIR / f"{student['roll_no']}.jpg"
        if photo_path.exists():
            img = cv2.imdecode(np.fromfile(str(photo_path), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                pic = tk.Label(body, bg=CARD)
                pic.image = tk.PhotoImage(data=cv2.imencode(".ppm", img)[1].tobytes())
                pic.configure(image=pic.image)
                pic.pack(pady=(0, 10))
        tk.Label(body, text=student["name"], bg=CARD, fg="#1f2430", font=("Segoe UI", 24, "bold")).pack()
        tk.Label(body, text=f"Roll no {student['roll_no']}   •   Class {student['class']}", bg=CARD,
                 fg=GREY, font=("Segoe UI", 12)).pack(pady=(4, 0))
        tk.Label(body, text=(f"Marked at {when}" if ok else f"Already marked today at {when}"),
                 bg=CARD, fg=color, font=("Segoe UI", 12, "bold")).pack(pady=(8, 0))

        btns = tk.Frame(top, bg=CARD)
        btns.pack(pady=(8, 20))
        if restart:
            make_button(btns, "Next student  ▶", self.next_student, "success").pack(side="left", padx=6)
        make_button(btns, "Close", self.close_popup, "light").pack(side="left", padx=6)

        top.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - top.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - top.winfo_height()) // 2
        top.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.popup_after = self.root.after(int(POPUP_SECONDS * 1000), self.close_popup)

    def close_popup(self):
        if self.popup_after is not None:
            try:
                self.root.after_cancel(self.popup_after)
            except Exception:
                pass
            self.popup_after = None
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
            self.popup = None

    def next_student(self):
        self.close_popup()
        self.start_attendance()

    def do_mark_absentees(self):
        if not messagebox.askyesno("Mark absentees",
                                   "Mark every student who has NOT been recognised today as Absent?\n\n"
                                   "Do this at the end of the attendance session."):
            return
        try:
            count = mark_absentees()
        except PermissionError:
            messagebox.showerror("File locked", "Close attendance.csv in Excel and try again.")
            return
        self.refresh_all()
        messagebox.showinfo("Done", f"{count} student(s) marked Absent for {today_str()}.")

    # ---------- registration mode ----------
    def read_form(self):
        roll = self.entries["roll"].get().strip()
        name = self.entries["name"].get().strip()
        cls = self.entries["cls"].get().strip()
        if not roll or not name or not cls:
            messagebox.showwarning("Missing details", "Please fill roll number, name and class.")
            return None
        if not valid_roll(roll):
            messagebox.showwarning("Roll number", "Roll number can contain only letters, numbers, - and _")
            return None
        if any(s["roll_no"].lower() == roll.lower() for s in load_students()):
            messagebox.showwarning("Duplicate", f"Roll number {roll} is already registered.")
            return None
        return {"roll": roll, "name": name, "cls": cls}

    def start_capture(self):
        info = self.read_form()
        if not info or not self.open_camera():
            return
        self.mode = "register"
        self.reg = {"info": info, "samples": [], "start": time.time() + REG_COUNTDOWN,
                    "last": 0.0, "photo": None}
        self.progress["value"] = 0
        self.ensure_loop()

    def cancel_capture(self):
        self.stop_camera()
        self.progress["value"] = 0
        self.reg_prompt.config(text="Fill the form, then click Start Face Capture", fg=PRIMARY)

    def process_register(self, frame):
        reg, now = self.reg, time.time()
        faces = self.engine.detect(frame)
        prompt, color = "", PRIMARY
        if reg is not None:
            done = len(reg["samples"])
            if now < reg["start"]:
                prompt = f"Get ready...  {int(reg['start'] - now) + 1}"
            elif not faces:
                prompt, color = "No face found - face the camera", RED
            elif len(faces) > 1:
                prompt, color = "Only one person please", RED
            else:
                face = faces[0]
                if face[2] < MIN_FACE_WIDTH:
                    prompt, color = "Come a little closer", RED
                elif face[14] < 0.9:
                    prompt, color = "Face not clear - improve the light", RED
                else:
                    prompt = POSES[min(done, len(POSES) - 1)]
                    if now - reg["last"] >= REG_INTERVAL:
                        reg["samples"].append(self.engine.embed(frame, face))
                        reg["last"] = now
                        if reg["photo"] is None and len(reg["samples"]) >= 3:
                            reg["photo"] = self.crop_face(frame, face)
            self.progress["value"] = len(reg["samples"])
            self.reg_prompt.config(text=prompt, fg=color)
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 200, 90), 2)
        self.show(frame, self.reg_view)

        if reg is not None and len(reg["samples"]) >= SAMPLES_NEEDED:
            info, samples, photo = reg["info"], reg["samples"], reg["photo"]
            self.stop_camera()
            self.root.after(50, lambda: self.finish_registration(info, samples, photo))

    @staticmethod
    def crop_face(frame, face, margin=0.35):
        x, y, w, h = [int(v) for v in face[:4]]
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        crop = frame[y0:y + h + my, x0:x + w + mx]
        if crop.size == 0:
            return None
        scale = 160 / max(crop.shape[:2])
        return cv2.resize(crop, None, fx=scale, fy=scale)

    def finish_registration(self, info, samples, photo):
        emb = np.array(samples, dtype=np.float32)
        mean = emb.mean(axis=0)
        mean /= np.linalg.norm(mean)
        match, _ = self.db.identify(mean)
        if match:
            other = self.db.students[match]
            messagebox.showwarning("Already registered",
                                   f"This face already matches {other['name']} ({match}).\n"
                                   "Registration cancelled.")
            self.reg_prompt.config(text="Registration cancelled", fg=RED)
            return False
        try:
            add_student(info["roll"], info["name"], info["cls"])
        except (ValueError, PermissionError) as e:
            messagebox.showerror("Could not save", str(e))
            return False
        np.save(EMB_DIR / f"{info['roll']}.npy", emb)
        if photo is not None:
            cv2.imencode(".jpg", photo)[1].tofile(str(PHOTO_DIR / f"{info['roll']}.jpg"))
        self.db.reload()
        for e in self.entries.values():
            e.delete(0, "end")
        self.progress["value"] = 0
        self.reg_prompt.config(text=f"✔ {info['name']} registered!", fg=GREEN)
        self.refresh_all()
        messagebox.showinfo("Registered", f"{info['name']} ({info['roll']}) registered successfully.")
        return True

    def register_from_photos(self):
        info = self.read_form()
        if not info:
            return
        files = filedialog.askopenfilenames(title="Choose 1 or more clear photos of the student",
                                            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not files:
            return
        samples, photo = [], None
        for path in files:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = self.engine.detect(img)
            if len(faces) != 1:
                continue
            samples.append(self.engine.embed(img, faces[0]))
            if photo is None:
                photo = self.crop_face(img, faces[0])
        if not samples:
            messagebox.showwarning("No face", "No photo had exactly one clear face. Try other photos.")
            return
        self.finish_registration(info, samples, photo)

    # ---------- records / students actions ----------
    def export_report(self):
        try:
            path = make_summary_report()
        except PermissionError:
            messagebox.showerror("File locked", "Close the report file in Excel and try again.")
            return
        messagebox.showinfo("Report created", f"Saved:\n{path}")
        self.open_folder(REPORT_DIR)

    def open_folder(self, folder=None):
        folder = folder if isinstance(folder, Path) else DATA_DIR
        try:
            os.startfile(folder)                 # Windows
        except AttributeError:
            messagebox.showinfo("Folder", f"Your files are in:\n{folder}")

    def delete_selected(self):
        sel = self.stu_tree.selection()
        if not sel:
            messagebox.showinfo("Delete", "Select a student in the list first.")
            return
        roll, name = self.stu_tree.item(sel[0])["values"][:2]
        if not messagebox.askyesno("Delete student",
                                   f"Delete {name} ({roll})?\nTheir old attendance records are kept."):
            return
        delete_student(str(roll))
        self.db.reload()
        self.refresh_all()

    def on_tab_changed(self, _event=None):
        self.close_popup()
        self.stop_camera()                       # camera is only used by the first two tabs
        self.refresh_all()

    def on_close(self):
        self.close_popup()
        self.stop_camera()
        self.root.destroy()


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # sharper text on Windows
    except Exception:
        pass
    init_storage()
    print("Loading face models...")
    try:
        engine = FaceEngine()
    except Exception as e:
        print("\n" + str(e))
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("Face models missing", str(e))
        except Exception:
            pass
        sys.exit(1)
    root = tk.Tk()
    App(root, engine, FaceDB())
    root.mainloop()


if __name__ == "__main__":
    main()
