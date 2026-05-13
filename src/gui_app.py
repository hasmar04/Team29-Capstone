"""
gui_app.py
----------
GUI application for the Queensland Reds Rugby Offside Detection System.
Capstone project - Team 29

Authors: Team 29
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import threading
import os
import sys
import queue
import time

import ui_functions as UI
import yolo_functions as YOLO
import field_functions as field
import ruck_functions as ruck
import lineout_functions as lineout
import drawing_functions as draw
import line_functions as line
import general_functions as general
import offside_functions as offside
import point_functions as points
import batch_processor as batch
from constants import RUCK_MODEL_CLASS_NUMBERS, LINEOUT_MODEL_CLASS_NUMBERS
from events.session_stats import DetectionSessionStats
from events.detection_event import DetectionEvent

import main

# Colour scheme
MAROON = "#800020"
MAROON_DARK = "#5c0018"
MAROON_LIGHT = "#a03050"
WHITE = "#ffffff"
LIGHT_GREY = "#f0f0f0"
CARD_BG = "#ffffff"
DARK_GREY = "#333333"
MID_GREY = "#666666"
BORDER_COLOUR = "#e0e0e0"

# Configure customtkinter appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class HeaderBar(ctk.CTkFrame):
    """Maroon header bar with title and subtitle."""

    def __init__(self, parent):
        super().__init__(parent, fg_color=MAROON, corner_radius=0, height=80)
        self.pack_propagate(False)

        self.title_label = ctk.CTkLabel(
            self, text="Queensland Reds - Rugby Offside Detection System",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            text_color=WHITE
        )
        self.title_label.pack(side="left", padx=25, pady=20)


class CardFrame(ctk.CTkFrame):
    """Reusable card container with rounded corners and white background."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=CARD_BG, corner_radius=15,
                         border_width=1, border_color=BORDER_COLOUR, **kwargs)


class SectionTitle(ctk.CTkLabel):
    """Reusable section title label in maroon."""

    def __init__(self, parent, text):
        super().__init__(parent, text=text,
                         font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
                         text_color=MAROON)


class LogTextbox(ctk.CTkTextbox):
    """Styled log textbox with monospace font."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, font=ctk.CTkFont(family="Consolas", size=9),
                         fg_color=WHITE, text_color=DARK_GREY,
                         border_width=1, border_color=BORDER_COLOUR,
                         corner_radius=8, **kwargs)


class ModeCard(CardFrame):
    """Processing mode selection card."""

    def __init__(self, parent, mode_var):
        super().__init__(parent)

        SectionTitle(self, "Processing Mode").pack(anchor="w", padx=20, pady=(15, 10))

        modes = [
            ("manual", "Manual Mode - Manually select when to detect rucks/lineouts"),
            ("auto", "Auto Mode - Automatic detection of rucks/lineouts"),
            ("batch", "Batch Mode - Process multiple videos automatically"),
        ]
        for value, label in modes:
            ctk.CTkRadioButton(
                self, text=label, variable=mode_var, value=value,
                font=ctk.CTkFont(family="Arial", size=10),
                fg_color=MAROON, hover_color=MAROON_LIGHT,
                border_color=MID_GREY, text_color=DARK_GREY
            ).pack(anchor="w", padx=25, pady=3)

        # Bottom padding
        ctk.CTkLabel(self, text="", height=5).pack()


class FileCard(CardFrame):
    """File/directory selection card."""

    def __init__(self, parent, video_var, output_var, on_browse_video, on_browse_output):
        super().__init__(parent)

        SectionTitle(self, "File Selection").pack(anchor="w", padx=20, pady=(15, 10))

        self._build_row("Video File:", video_var, on_browse_video)
        self._build_row("Output Directory:", output_var, on_browse_output)

        # Bottom padding
        ctk.CTkLabel(self, text="", height=5).pack()

    def _build_row(self, label_text, variable, command):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(
            row, text=label_text, width=120, anchor="w",
            font=ctk.CTkFont(family="Arial", size=10, weight="bold"),
            text_color=DARK_GREY
        ).pack(side="left")

        ctk.CTkEntry(
            row, textvariable=variable, state="readonly",
            font=ctk.CTkFont(family="Arial", size=10),
            fg_color=WHITE, text_color=DARK_GREY,
            border_color=BORDER_COLOUR, corner_radius=8
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkButton(
            row, text="Browse...", command=command, width=100, height=32,
            fg_color=MAROON, hover_color=MAROON_DARK, corner_radius=16,
            font=ctk.CTkFont(family="Arial", size=10, weight="bold")
        ).pack(side="right")


class ProgressCard(CardFrame):
    """Progress bar and status display card."""

    def __init__(self, parent):
        super().__init__(parent)

        SectionTitle(self, "Progress").pack(anchor="w", padx=20, pady=(15, 10))

        self.progress_bar = ctk.CTkProgressBar(
            self, progress_color=MAROON, fg_color=BORDER_COLOUR,
            corner_radius=10, height=20
        )
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 8))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            self, text="Ready to start processing",
            font=ctk.CTkFont(family="Arial", size=10), text_color=MID_GREY
        )
        self.status_label.pack(anchor="w", padx=20, pady=(0, 15))

    def set_progress(self, value):
        """Set progress 0-100."""
        self.progress_bar.set(max(0, min(100, value)) / 100.0)

    def set_status(self, text, colour=MID_GREY):
        self.status_label.configure(text=text, text_color=colour)


class ModelStatusCard(CardFrame):
    """Model loading status card."""

    def __init__(self, parent):
        super().__init__(parent)

        SectionTitle(self, "Model Status").pack(anchor="w", padx=20, pady=(15, 5))

        self.status_label = ctk.CTkLabel(
            self, text="Loading YOLO models...",
            font=ctk.CTkFont(family="Arial", size=10), text_color=MID_GREY
        )
        self.status_label.pack(anchor="w", padx=20, pady=(0, 15))

    def set_status(self, text, colour=MID_GREY):
        self.status_label.configure(text=text, text_color=colour)


class ReportsTreeCard(CardFrame):
    """Generated reports treeview card."""

    def __init__(self, parent, on_select):
        super().__init__(parent)

        SectionTitle(self, "Generated Reports").pack(anchor="w", padx=20, pady=(15, 10))

        columns = ("Video", "Event Type", "Confidence", "Offsides")

        tree_frame = ctk.CTkFrame(self, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # Use ttk Treeview (no CTk equivalent)
        style = ttk.Style()
        style.configure("Treeview", background=WHITE, foreground=DARK_GREY,
                         rowheight=25, fieldbackground=WHITE)
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"),
                         background=MAROON, foreground=WHITE)
        style.map("Treeview",
                   background=[("selected", MAROON_LIGHT)],
                   foreground=[("selected", WHITE)])

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=5)
        for col, w in zip(columns, [120, 80, 80, 80]):
            self.tree.heading(col, text=col if col != "Video" else "Video Name")
            self.tree.column(col, width=w)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", on_select)


class RugbyOffsideGUI:
    """Main application controller."""

    def __init__(self, root):
        self.root = root
        self.root.title("Queensland Reds - Rugby Offside Detection System")
        self.root.geometry("1600x900")
        self.root.minsize(1400, 800)
        self.root.configure(fg_color=LIGHT_GREY)

        # State
        self.video_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.mode = ctk.StringVar(value="auto")
        self.processing = False
        self.progress_queue = queue.Queue()
        self.batch_reports = []
        self.models_loaded = False
        self.models = {}

        # Paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.models_dir = os.path.join(self.project_root, "models")

        # Build UI
        self._build_ui()

        # Load models and start monitoring
        self.load_models_async()
        self.monitor_progress()

    def _build_ui(self):
        """Construct the full interface."""
        # Header
        HeaderBar(self.root).pack(fill="x")

        # Main content area
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # Left panel
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right panel
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # --- Left panel cards ---
        ModeCard(left, self.mode).pack(fill="x", pady=(0, 12))

        FileCard(left, self.video_path, self.output_dir,
                 self.select_video, self.select_output).pack(fill="x", pady=(0, 12))

        # Control buttons
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", pady=12)

        self.start_button = ctk.CTkButton(
            btn_row, text="Start Processing", command=self.start_processing,
            fg_color=MAROON, hover_color=MAROON_DARK, width=180, height=45,
            corner_radius=22, font=ctk.CTkFont(family="Arial", size=12, weight="bold")
        )
        self.start_button.pack(side="left", padx=(0, 15))

        self.stop_button = ctk.CTkButton(
            btn_row, text="Stop", command=self.stop_processing,
            fg_color="#c0392b", hover_color="#a93226", width=100, height=45,
            corner_radius=22, font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            state="disabled"
        )
        self.stop_button.pack(side="left")

        # Progress
        self.progress_card = ProgressCard(left)
        self.progress_card.pack(fill="x", pady=(0, 12))

        # Model status
        self.model_card = ModelStatusCard(left)
        self.model_card.pack(fill="x", pady=(0, 12))

        # Processing log
        log_card = CardFrame(left)
        log_card.pack(fill="both", expand=True)
        SectionTitle(log_card, "Processing Log").pack(anchor="w", padx=20, pady=(15, 10))
        self.log_text = LogTextbox(log_card)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # --- Right panel cards ---

        # Event log
        event_card = CardFrame(right)
        event_card.pack(fill="both", expand=True, pady=(0, 12))
        SectionTitle(event_card, "Event Log & Reports").pack(anchor="w", padx=20, pady=(15, 10))
        self.event_log_text = LogTextbox(event_card, state="disabled")
        self.event_log_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # Reports tree
        self.reports_card = ReportsTreeCard(right, self.on_report_select)
        self.reports_card.pack(fill="x", pady=(0, 12))
        self.reports_tree = self.reports_card.tree

        # Report details
        details_card = CardFrame(right)
        details_card.pack(fill="both", expand=True, pady=(0, 12))
        SectionTitle(details_card, "Report Details").pack(anchor="w", padx=20, pady=(15, 10))
        self.report_details_text = LogTextbox(details_card, state="disabled")
        self.report_details_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # Clear log button
        ctk.CTkButton(
            right, text="Clear Log", command=self.clear_event_log,
            fg_color=MID_GREY, hover_color=DARK_GREY, width=120, height=35,
            corner_radius=17, font=ctk.CTkFont(family="Arial", size=10, weight="bold")
        ).pack(pady=(0, 5))

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_message(self, message, level="INFO"):
        """Add a timestamped message to the processing log and event log."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}\n"

        self.log_text.insert("end", entry)
        self.log_text.see("end")

        colour_map = {"ERROR": "#c0392b", "SUCCESS": "#27ae60", "WARNING": "#d35400"}
        if level in colour_map:
            tag = level.lower()
            self.log_text.tag_add(tag, "end-2l", "end-1l")
            self.log_text.tag_config(tag, foreground=colour_map[level])

        self.add_to_event_log(message, level)

    def add_to_event_log(self, message, level="INFO"):
        """Add a timestamped message to the event log."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}\n"

        self.event_log_text.configure(state="normal")
        self.event_log_text.insert("end", entry)
        self.event_log_text.see("end")

        colour_map = {"ERROR": "#c0392b", "SUCCESS": "#27ae60", "WARNING": "#d35400"}
        if level in colour_map:
            tag = f"evt_{level.lower()}"
            self.event_log_text.tag_add(tag, "end-2l", "end-1l")
            self.event_log_text.tag_config(tag, foreground=colour_map[level])

        self.event_log_text.configure(state="disabled")

    def add_batch_event_log(self, message):
        """Add a raw batch-processing message to the event log."""
        self.event_log_text.configure(state="normal")
        self.event_log_text.insert("end", message + "\n")
        self.event_log_text.see("end")

        low = message.lower()
        if "detected" in low and "confidence" in low:
            self.event_log_text.tag_add("detection", "end-2l", "end-1l")
            self.event_log_text.tag_config("detection", foreground=MAROON)
        elif "processing complete" in low or "[done]" in low:
            self.event_log_text.tag_add("complete", "end-2l", "end-1l")
            self.event_log_text.tag_config("complete", foreground="#27ae60")
        elif "no detections found" in low:
            self.event_log_text.tag_add("no_det", "end-2l", "end-1l")
            self.event_log_text.tag_config("no_det", foreground="#d35400")

        self.event_log_text.configure(state="disabled")

    def clear_event_log(self):
        self.event_log_text.configure(state="normal")
        self.event_log_text.delete("1.0", "end")
        self.event_log_text.configure(state="disabled")

    def update_progress(self, progress_percent):
        self.progress_queue.put(("progress_update", progress_percent))

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def select_video(self):
        if self.mode.get() == "batch":
            directory = filedialog.askdirectory(title="Select Directory Containing Videos")
            if directory:
                self.video_path.set(directory)
                self.log_message(f"Selected batch directory: {directory}")
        else:
            filetypes = [("Video Files", "*.mp4 *.avi *.mov *.gif"), ("All Files", "*.*")]
            filename = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
            if filename:
                self.video_path.set(filename)
                self.log_message(f"Selected video: {os.path.basename(filename)}")

    def select_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.log_message(f"Selected output directory: {directory}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models_async(self):
        def _load():
            try:
                self.log_message("Loading YOLO models...")
                self.models["ruck"] = YOLO.load_model(os.path.join(self.models_dir, "ruck.pt"))
                self.models["lineout"] = YOLO.load_model(os.path.join(self.models_dir, "lineout.pt"))
                self.models["ball"] = YOLO.load_model(os.path.join(self.models_dir, "ball.pt"))
                self.models["player"] = YOLO.load_model(os.path.join(self.models_dir, "player-id.pt"))
                self.models_loaded = True
                self.progress_queue.put(("model_loaded", "YOLO models loaded successfully"))
            except Exception as e:
                self.progress_queue.put(("error", f"Failed to load models: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------------
    # Progress monitoring
    # ------------------------------------------------------------------

    def monitor_progress(self):
        try:
            while True:
                msg_type, msg = self.progress_queue.get_nowait()

                if msg_type == "model_loaded":
                    self.model_card.set_status("YOLO models loaded successfully", "#27ae60")
                    self.log_message("YOLO models loaded successfully", "SUCCESS")
                    self.start_button.configure(state="normal")

                elif msg_type == "error":
                    self.log_message(msg, "ERROR")
                    messagebox.showerror("Error", msg)

                elif msg_type == "status":
                    self.progress_card.set_status(msg)
                    self.log_message(msg)

                elif msg_type == "progress_update":
                    self.progress_card.set_progress(msg)
                    self.progress_card.set_status(f"Processing... {msg:.1f}%")

                elif msg_type == "progress":
                    self.log_message(msg)

                elif msg_type == "parse_reports":
                    self.parse_batch_reports(msg)
                    self.log_message(f"Parsed {len(self.batch_reports)} reports", "SUCCESS")

                elif msg_type == "complete":
                    self.processing = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.progress_card.set_progress(100)
                    self.progress_card.set_status("Processing complete!", "#27ae60")
                    self.log_message("Processing completed successfully!", "SUCCESS")

                    if self.mode.get() == "batch":
                        messagebox.showinfo("Complete",
                            f"Batch processing completed!\n\nProcessed {len(self.batch_reports)} videos\n"
                            f"Output saved to: {self.output_dir.get()}")
                    else:
                        messagebox.showinfo("Complete",
                            f"Processing completed!\n\nOutput saved to: {self.output_dir.get()}")

        except queue.Empty:
            pass

        self.root.after(100, self.monitor_progress)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def start_processing(self):
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait...")
            return
        if not self.video_path.get():
            messagebox.showwarning("Warning", "Please select a video file or directory first.")
            return

        if not self.output_dir.get():
            if self.mode.get() == "batch":
                default_output = os.path.join(self.project_root, "batch_output")
            else:
                name = os.path.splitext(os.path.basename(self.video_path.get()))[0]
                default_output = os.path.join(self.project_root, f"{name}_output")
            self.output_dir.set(default_output)
            os.makedirs(default_output, exist_ok=True)

        self.processing = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress_card.set_progress(0)
        self.progress_card.set_status("Starting processing...")

        threading.Thread(target=self.run_processing, daemon=True).start()

    def stop_processing(self):
        self.processing = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_card.set_progress(0)
        self.progress_card.set_status("Processing stopped", "#d35400")
        self.log_message("Processing stopped by user", "WARNING")

    def run_processing(self):
        try:
            mode = self.mode.get()
            video_path = self.video_path.get()
            output_dir = self.output_dir.get()

            self.progress_queue.put(("status", f"Starting {mode} mode processing..."))

            if mode == "batch":
                self.run_batch_mode(video_path, output_dir)
            else:
                self.run_single_video_mode(video_path, output_dir, mode)

            self.progress_queue.put(("complete", "Processing completed successfully"))

        except Exception as e:
            self.progress_queue.put(("error", f"Processing failed: {e}"))
            self.processing = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.progress_card.set_progress(0)

    def run_batch_mode(self, input_path, output_dir):
        input_path = os.path.normpath(input_path).replace("\\", "/")
        output_dir = os.path.normpath(output_dir).replace("\\", "/")

        self.progress_queue.put(("progress", f"Processing batch directory: {input_path}"))

        self.batch_reports = []
        for item in self.reports_tree.get_children():
            self.reports_tree.delete(item)
        self.report_details_text.configure(state="normal")
        self.report_details_text.delete("1.0", "end")
        self.report_details_text.configure(state="disabled")

        self.clear_event_log()
        self.add_batch_event_log("=" * 80)
        self.add_batch_event_log("BATCH PROCESSING STARTED")
        self.add_batch_event_log("=" * 80)
        self.add_batch_event_log("")

        self.run_batch_with_logging(input_path, output_dir)

        self.add_batch_event_log("")
        self.add_batch_event_log("=" * 80)
        self.add_batch_event_log("BATCH PROCESSING COMPLETE")
        self.add_batch_event_log(f"Output directory: {output_dir}")
        self.add_batch_event_log("=" * 80)

        self.progress_queue.put(("progress", f"Batch processing complete. Output saved to: {output_dir}"))
        self.progress_queue.put(("parse_reports", output_dir))

    def run_batch_with_logging(self, input_path, output_dir):
        from io import StringIO

        video_extensions = [".mp4", ".avi", ".mov", ".gif"]
        video_files = sorted(
            os.path.join(input_path, f).replace("\\", "/")
            for f in os.listdir(input_path)
            if any(f.lower().endswith(ext) for ext in video_extensions)
        )
        total_videos = len(video_files)

        self.add_batch_event_log(f"BATCH PROCESSING: {total_videos} video(s) found")
        self.add_batch_event_log("=" * 80)
        self.add_batch_event_log("")
        self.update_progress(0)

        class OutputCapture:
            def __init__(self, callback, progress_cb, total):
                self.callback = callback
                self.progress_cb = progress_cb
                self.total = total
                self.completed = 0
                self.buffer = StringIO()

            def write(self, text):
                self.buffer.write(text)
                if text.strip():
                    if "Summary report saved" in text:
                        self.completed += 1
                        pct = (self.completed / self.total) * 100
                        self.progress_cb(pct)
                        self.callback(f"[DONE] Video {self.completed}/{self.total} completed ({pct:.0f}%)")
                    elif "Batch processing complete" in text:
                        self.progress_cb(100)
                    elif "Starting batch processing" in text:
                        self.progress_cb(0)
                    self.callback(text.strip())

            def flush(self):
                pass

        original_stdout = sys.stdout
        sys.stdout = OutputCapture(self.add_batch_event_log, self.update_progress, total_videos)

        try:
            batch.process_video_batch(
                input_path, output_dir,
                self.models["ruck"], self.models["lineout"],
                self.models["ball"], self.models["player"]
            )
        finally:
            sys.stdout = original_stdout

    def run_single_video_mode(self, video_path, output_dir, mode):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        fps = general.get_video_fps(video_path)
        self.progress_queue.put(("progress", f"Video FPS: {fps}"))

        if mode == "manual":
            self.run_manual_mode(video_path, fps, output_dir)
        else:
            self.run_auto_mode(video_path, fps, output_dir)

    def run_manual_mode(self, video_path, fps, output_dir):
        self.progress_queue.put(("progress", "Starting manual mode processing..."))
        self.progress_queue.put(("progress", "Manual mode requires user interaction - check console windows"))
        self.progress_queue.put(("progress_update", 0))
        self.progress_queue.put(("progress_update", 50))
        main.manual_mode(video_path, fps, self.models["ruck"], self.models["lineout"], self.models["player"])
        self.progress_queue.put(("progress_update", 100))

    def run_auto_mode(self, video_path, fps, output_dir):
        self.progress_queue.put(("progress", "Starting automatic mode processing..."))
        self.progress_queue.put(("progress_update", 0))
        self.progress_queue.put(("progress_update", 50))
        main.auto_mode(video_path, fps, self.models["ruck"], self.models["lineout"],
                       self.models["ball"], self.models["player"])
        self.progress_queue.put(("progress_update", 100))
        self.progress_queue.put(("progress", "Automatic processing complete"))

    # ------------------------------------------------------------------
    # Report handling
    # ------------------------------------------------------------------

    def on_report_select(self, event):
        selection = self.reports_tree.selection()
        if selection:
            video_name = self.reports_tree.item(selection[0])["values"][0]
            for report in self.batch_reports:
                if report.get("video_name", "").startswith(str(video_name)):
                    self.display_report_details(report)
                    break

    def display_report_details(self, data):
        self.report_details_text.configure(state="normal")
        self.report_details_text.delete("1.0", "end")

        lines = [
            f"Video: {data.get('video_name', 'Unknown')}",
            f"Processing Date: {data.get('processing_date', 'Unknown')}",
            f"Total Frames: {data.get('total_frames', 'Unknown')}",
            f"Total Events: {data.get('total_events', 'Unknown')}",
            "", "DETECTION EVENTS:", "=" * 50,
        ]
        for i, evt in enumerate(data.get("events", []), 1):
            lines += [
                f"\nEvent #{i}",
                f"  Type: {evt.get('type', 'Unknown')}",
                f"  Frame: {evt.get('frame', 'Unknown')}",
                f"  Timestamp: {evt.get('timestamp', 'Unknown')}",
                f"  Confidence: {evt.get('confidence', 'Unknown')}",
                f"  Offside Players: {evt.get('offside_count', 'Unknown')}",
            ]
        lines += [
            f"\nSUMMARY STATISTICS:", "=" * 50,
            f"Total Ruck Events: {data.get('ruck_count', 0)}",
            f"Total Lineout Events: {data.get('lineout_count', 0)}",
            f"Total Offside Players: {data.get('total_offside', 0)}",
            f"Average Confidence: {data.get('avg_confidence', 'N/A')}",
        ]

        self.report_details_text.insert("1.0", "\n".join(lines))
        self.report_details_text.configure(state="disabled")

    def parse_batch_reports(self, output_dir):
        output_dir = os.path.normpath(output_dir).replace("\\", "/")
        self.batch_reports = []

        for item in self.reports_tree.get_children():
            self.reports_tree.delete(item)

        if not os.path.exists(output_dir):
            return

        self.log_message(f"Looking for reports in: {output_dir}")
        files_found = os.listdir(output_dir)
        self.log_message(f"Files in directory: {files_found}")

        for filename in files_found:
            if not filename.endswith("_analysis_report.txt"):
                continue

            self.log_message(f"Found report file: {filename}")
            report_path = os.path.join(output_dir, filename).replace("\\", "/")
            try:
                self.log_message(f"Parsing report: {report_path}")
                report_data = self.parse_report_file(report_path)
                if not report_data:
                    self.log_message(f"Failed to parse report data from: {filename}")
                    continue

                self.log_message(f"Successfully parsed report: {filename}")
                self.batch_reports.append(report_data)

                video_name = report_data.get("video_name", filename)
                events = report_data.get("events", [])
                if events:
                    event_type = events[0].get("type", "Unknown")
                    confidence = events[0].get("confidence", "N/A")
                else:
                    event_type = "No Events"
                    confidence = "N/A"

                self.reports_tree.insert("", "end", values=(
                    video_name, event_type, confidence,
                    report_data.get("total_offside", 0)
                ))
            except Exception as e:
                self.log_message(f"Error parsing report {filename}: {e}", "ERROR")

    def parse_report_file(self, report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            data = {
                "video_name": "Unknown", "processing_date": "Unknown",
                "total_frames": 0, "total_events": 0, "events": [],
                "ruck_count": 0, "lineout_count": 0,
                "total_offside": 0, "avg_confidence": "N/A",
            }

            current_event = None
            for raw_line in content.split("\n"):
                ln = raw_line.strip()

                if ln.startswith("Video File:"):
                    data["video_name"] = ln.split(":", 1)[1].strip()
                elif ln.startswith("Processing Date:"):
                    data["processing_date"] = ln.split(":", 1)[1].strip()
                elif ln.startswith("Total Frames Analysed:"):
                    try: data["total_frames"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif ln.startswith("Total Events Detected:"):
                    try: data["total_events"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif ln.startswith("Event #"):
                    if current_event:
                        data["events"].append(current_event)
                    current_event = {"type": "Unknown", "frame": 0,
                                     "timestamp": "Unknown", "confidence": "Unknown",
                                     "offside_count": 0}
                elif current_event and ln.startswith("Type:"):
                    current_event["type"] = ln.split(":", 1)[1].strip()
                elif current_event and ln.startswith("Frame Number:"):
                    try: current_event["frame"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif current_event and ln.startswith("Timestamp:"):
                    current_event["timestamp"] = ln.split(":", 1)[1].strip()
                elif current_event and ln.startswith("Detection Confidence:"):
                    current_event["confidence"] = ln.split(":", 1)[1].strip()
                elif current_event and ln.startswith("Offside Players Detected:"):
                    try: current_event["offside_count"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif ln.startswith("Total Ruck Events:"):
                    try: data["ruck_count"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif ln.startswith("Total Lineout Events:"):
                    try: data["lineout_count"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif ln.startswith("Total Offside Players Across All Events:"):
                    try: data["total_offside"] = int(ln.split(":", 1)[1].strip())
                    except ValueError: pass
                elif ln.startswith("Average Ruck Detection Confidence:"):
                    data["avg_confidence"] = ln.split(":", 1)[1].strip()

            if current_event:
                data["events"].append(current_event)

            return data

        except Exception as e:
            self.log_message(f"Error parsing report file {report_path}: {e}", "ERROR")
            return None


def run_gui():
    """Main entry point for the GUI application."""
    root = ctk.CTk()
    app = RugbyOffsideGUI(root)

    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = root.winfo_width()
    wh = root.winfo_height()
    root.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")

    root.mainloop()


if __name__ == "__main__":
    run_gui()
