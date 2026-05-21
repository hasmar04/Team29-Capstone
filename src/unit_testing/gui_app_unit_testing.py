"""
gui_app_unit_testing.py

Unit tests for the gui_app.py module, which provides the GUI application for the
Queensland Reds Rugby Offside Detection System.

Tested Functionality:
    - Widget classes: HeaderBar, CardFrame, SectionTitle, LogTextbox, ModeCard,
      FileCard, ProgressCard, ModelStatusCard, ReportsTreeCard
    - RugbyOffsideGUI controller: initial state, mode switching, file selection,
      logging, progress updates, report parsing, processing guards, stop processing

Test Structure:
    - Uses unittest with a hidden root window (root.withdraw()) to avoid
      requiring a visible display during testing.
    - Uses unittest.mock to mock file dialogs, model loading, and messagebox calls.
    - Each test verifies widget state, variable bindings, or callback behaviour.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pytest

ctk = pytest.importorskip("customtkinter")


class TestWidgetClasses(unittest.TestCase):
    """Tests for individual custom widget classes."""

    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def test_header_bar_creation(self):
        """HeaderBar should display the correct title text."""
        from src.gui_app import HeaderBar
        header = HeaderBar(self.root)
        self.assertEqual(
            header.title_label.cget("text"),
            "Queensland Reds - Rugby Offside Detection System"
        )
        header.destroy()

    def test_card_frame_creation(self):
        """CardFrame should be created with a white background and rounded corners."""
        from src.gui_app import CardFrame, CARD_BG
        card = CardFrame(self.root)
        self.assertEqual(card.cget("fg_color"), CARD_BG)
        self.assertEqual(card.cget("corner_radius"), 15)
        card.destroy()

    def test_section_title_text(self):
        """SectionTitle should display the given text in maroon colour."""
        from src.gui_app import SectionTitle, MAROON
        title = SectionTitle(self.root, "Test Section")
        self.assertEqual(title.cget("text"), "Test Section")
        self.assertEqual(title.cget("text_color"), MAROON)
        title.destroy()

    def test_progress_card_initial_state(self):
        """ProgressCard should initialise with progress at 0 and 'Ready' status."""
        from src.gui_app import ProgressCard
        card = ProgressCard(self.root)
        self.assertAlmostEqual(card.progress_bar.get(), 0.0)
        self.assertEqual(card.status_label.cget("text"), "Ready to start processing")
        card.destroy()

    def test_progress_card_set_progress(self):
        """set_progress should clamp value between 0-100 and scale to 0-1."""
        from src.gui_app import ProgressCard
        card = ProgressCard(self.root)

        card.set_progress(50)
        self.assertAlmostEqual(card.progress_bar.get(), 0.5)

        card.set_progress(100)
        self.assertAlmostEqual(card.progress_bar.get(), 1.0)

        # Clamp below 0
        card.set_progress(-10)
        self.assertAlmostEqual(card.progress_bar.get(), 0.0)

        # Clamp above 100
        card.set_progress(200)
        self.assertAlmostEqual(card.progress_bar.get(), 1.0)

        card.destroy()

    def test_progress_card_set_status(self):
        """set_status should update the status label text."""
        from src.gui_app import ProgressCard
        card = ProgressCard(self.root)

        card.set_status("Processing...")
        self.assertEqual(card.status_label.cget("text"), "Processing...")

        card.set_status("Done!", "#27ae60")
        self.assertEqual(card.status_label.cget("text"), "Done!")
        self.assertEqual(card.status_label.cget("text_color"), "#27ae60")

        card.destroy()

    def test_model_status_card_initial_state(self):
        """ModelStatusCard should show 'Loading YOLO models...' on init."""
        from src.gui_app import ModelStatusCard
        card = ModelStatusCard(self.root)
        self.assertEqual(card.status_label.cget("text"), "Loading YOLO models...")
        card.destroy()

    def test_model_status_card_set_status(self):
        """set_status should update the model status text and colour."""
        from src.gui_app import ModelStatusCard
        card = ModelStatusCard(self.root)

        card.set_status("Models loaded", "#27ae60")
        self.assertEqual(card.status_label.cget("text"), "Models loaded")
        self.assertEqual(card.status_label.cget("text_color"), "#27ae60")

        card.destroy()

    def test_mode_card_creation(self):
        """ModeCard should create radio buttons for manual, auto, and batch modes."""
        from src.gui_app import ModeCard
        mode_var = ctk.StringVar(value="auto")
        card = ModeCard(self.root, mode_var)
        # The mode variable should retain its initial value
        self.assertEqual(mode_var.get(), "auto")
        card.destroy()

    def test_file_card_creation(self):
        """FileCard should create with the provided string variables."""
        from src.gui_app import FileCard
        video_var = ctk.StringVar(value="")
        output_var = ctk.StringVar(value="")
        card = FileCard(self.root, video_var, output_var, lambda: None, lambda: None)
        self.assertEqual(video_var.get(), "")
        self.assertEqual(output_var.get(), "")
        card.destroy()

    def test_reports_tree_card_columns(self):
        """ReportsTreeCard should have the correct treeview columns."""
        from src.gui_app import ReportsTreeCard
        card = ReportsTreeCard(self.root, lambda e: None)
        expected_columns = ("Video", "Event Type", "Confidence", "Offsides")
        self.assertEqual(card.tree["columns"], expected_columns)
        card.destroy()


class TestRugbyOffsideGUI(unittest.TestCase):
    """Tests for the main RugbyOffsideGUI controller class."""

    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def _create_gui(self):
        """Helper to create a GUI instance with mocked model loading."""
        with patch('gui_app.RugbyOffsideGUI.load_models_async'):
            with patch('gui_app.RugbyOffsideGUI.monitor_progress'):
                from src.gui_app import RugbyOffsideGUI
                gui = RugbyOffsideGUI(self.root)
        return gui

    def test_initial_mode_is_auto(self):
        """Default processing mode should be 'auto'."""
        gui = self._create_gui()
        self.assertEqual(gui.mode.get(), "auto")

    def test_initial_paths_empty(self):
        """Video path and output directory should be empty on init."""
        gui = self._create_gui()
        self.assertEqual(gui.video_path.get(), "")
        self.assertEqual(gui.output_dir.get(), "")

    def test_initial_processing_state(self):
        """GUI should not be processing and models should not be loaded on init."""
        gui = self._create_gui()
        self.assertFalse(gui.processing)
        self.assertFalse(gui.models_loaded)
        self.assertEqual(gui.models, {})
        self.assertEqual(gui.batch_reports, [])

    def test_mode_switching(self):
        """Changing the mode variable should reflect correctly."""
        gui = self._create_gui()

        gui.mode.set("manual")
        self.assertEqual(gui.mode.get(), "manual")

        gui.mode.set("batch")
        self.assertEqual(gui.mode.get(), "batch")

        gui.mode.set("auto")
        self.assertEqual(gui.mode.get(), "auto")

    def test_log_message_adds_text(self):
        """log_message should insert text into the processing log."""
        gui = self._create_gui()
        gui.log_message("Test log entry", "INFO")
        content = gui.log_text.get("1.0", "end").strip()
        self.assertIn("Test log entry", content)
        self.assertIn("INFO", content)

    def test_log_message_error_level(self):
        """log_message with ERROR level should add text to the log."""
        gui = self._create_gui()
        gui.log_message("Something failed", "ERROR")
        content = gui.log_text.get("1.0", "end").strip()
        self.assertIn("Something failed", content)
        self.assertIn("ERROR", content)

    def test_log_message_success_level(self):
        """log_message with SUCCESS level should add text to the log."""
        gui = self._create_gui()
        gui.log_message("Operation succeeded", "SUCCESS")
        content = gui.log_text.get("1.0", "end").strip()
        self.assertIn("Operation succeeded", content)

    def test_add_to_event_log(self):
        """add_to_event_log should insert text into the event log textbox."""
        gui = self._create_gui()
        gui.add_to_event_log("Event occurred", "INFO")
        gui.event_log_text.configure(state="normal")
        content = gui.event_log_text.get("1.0", "end").strip()
        gui.event_log_text.configure(state="disabled")
        self.assertIn("Event occurred", content)

    def test_add_batch_event_log(self):
        """add_batch_event_log should insert raw messages into the event log."""
        gui = self._create_gui()
        gui.add_batch_event_log("Detected ruck with confidence 0.95")
        gui.event_log_text.configure(state="normal")
        content = gui.event_log_text.get("1.0", "end").strip()
        gui.event_log_text.configure(state="disabled")
        self.assertIn("Detected ruck with confidence 0.95", content)

    def test_clear_event_log(self):
        """clear_event_log should empty the event log textbox."""
        gui = self._create_gui()
        gui.add_to_event_log("Some event", "INFO")
        gui.clear_event_log()
        gui.event_log_text.configure(state="normal")
        content = gui.event_log_text.get("1.0", "end").strip()
        gui.event_log_text.configure(state="disabled")
        self.assertEqual(content, "")

    @patch('src.gui_app.filedialog.askopenfilename', return_value="/fake/path/video.mp4")
    def test_select_video_single_mode(self, mock_dialog):
        """select_video in non-batch mode should set the video path via file dialog."""
        gui = self._create_gui()
        gui.mode.set("auto")
        gui.select_video()
        self.assertEqual(gui.video_path.get(), "/fake/path/video.mp4")

    @patch('src.gui_app.filedialog.askopenfilename', return_value="")
    def test_select_video_cancelled(self, mock_dialog):
        """select_video should not update path if user cancels the dialog."""
        gui = self._create_gui()
        gui.mode.set("auto")
        gui.video_path.set("/existing/video.mp4")
        gui.select_video()
        self.assertEqual(gui.video_path.get(), "/existing/video.mp4")

    @patch('src.gui_app.filedialog.askdirectory', return_value="/fake/batch/dir")
    def test_select_video_batch_mode(self, mock_dialog):
        """select_video in batch mode should open a directory dialog."""
        gui = self._create_gui()
        gui.mode.set("batch")
        gui.select_video()
        self.assertEqual(gui.video_path.get(), "/fake/batch/dir")

    @patch('src.gui_app.filedialog.askdirectory', return_value="/fake/output")
    def test_select_output(self, mock_dialog):
        """select_output should set the output directory path."""
        gui = self._create_gui()
        gui.select_output()
        self.assertEqual(gui.output_dir.get(), "/fake/output")

    @patch('src.gui_app.filedialog.askdirectory', return_value="")
    def test_select_output_cancelled(self, mock_dialog):
        """select_output should not update path if user cancels the dialog."""
        gui = self._create_gui()
        gui.output_dir.set("/existing/output")
        gui.select_output()
        self.assertEqual(gui.output_dir.get(), "/existing/output")

    @patch('src.gui_app.messagebox.showwarning')
    def test_start_processing_no_models(self, mock_warn):
        """start_processing should warn if models are not loaded yet."""
        gui = self._create_gui()
        gui.models_loaded = False
        gui.start_processing()
        mock_warn.assert_called_once()
        self.assertIn("loading", mock_warn.call_args[0][1].lower())

    @patch('src.gui_app.messagebox.showwarning')
    def test_start_processing_no_video(self, mock_warn):
        """start_processing should warn if no video file is selected."""
        gui = self._create_gui()
        gui.models_loaded = True
        gui.video_path.set("")
        gui.start_processing()
        mock_warn.assert_called_once()
        self.assertIn("video", mock_warn.call_args[0][1].lower())

    def test_stop_processing(self):
        """stop_processing should reset processing state and update UI."""
        gui = self._create_gui()
        gui.processing = True
        gui.stop_processing()
        self.assertFalse(gui.processing)

    def test_stop_processing_updates_progress(self):
        """stop_processing should reset the progress bar to 0."""
        gui = self._create_gui()
        gui.processing = True
        gui.progress_card.set_progress(75)
        gui.stop_processing()
        self.assertAlmostEqual(gui.progress_card.progress_bar.get(), 0.0)

    def test_update_progress_queues_message(self):
        """update_progress should put a progress_update message in the queue."""
        gui = self._create_gui()
        gui.update_progress(42.5)
        msg_type, msg = gui.progress_queue.get_nowait()
        self.assertEqual(msg_type, "progress_update")
        self.assertEqual(msg, 42.5)


class TestParseReportFile(unittest.TestCase):
    """Tests for the parse_report_file method."""

    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def _create_gui(self):
        with patch('src.gui_app.RugbyOffsideGUI.load_models_async'):
            with patch('src.gui_app.RugbyOffsideGUI.monitor_progress'):
                from src.gui_app import RugbyOffsideGUI
                gui = RugbyOffsideGUI(self.root)
        return gui

    def test_parse_report_file_basic(self):
        """parse_report_file should correctly extract data from a well-formed report."""
        gui = self._create_gui()

        report_content = """Video File: test_video.mp4
Processing Date: 2026-03-29
Total Frames Analysed: 1500
Total Events Detected: 2

Event #1
Type: Ruck
Frame Number: 100
Timestamp: 00:03.33
Detection Confidence: 0.92
Offside Players Detected: 3

Event #2
Type: Lineout
Frame Number: 500
Timestamp: 00:16.67
Detection Confidence: 0.87
Offside Players Detected: 1

Total Ruck Events: 1
Total Lineout Events: 1
Total Offside Players Across All Events: 4
Average Ruck Detection Confidence: 0.92
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='_analysis_report.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write(report_content)
            report_path = f.name

        try:
            data = gui.parse_report_file(report_path)

            self.assertIsNotNone(data)
            self.assertEqual(data["video_name"], "test_video.mp4")
            self.assertEqual(data["processing_date"], "2026-03-29")
            self.assertEqual(data["total_frames"], 1500)
            self.assertEqual(data["total_events"], 2)
            self.assertEqual(len(data["events"]), 2)

            # First event
            self.assertEqual(data["events"][0]["type"], "Ruck")
            self.assertEqual(data["events"][0]["frame"], 100)
            self.assertEqual(data["events"][0]["confidence"], "0.92")
            self.assertEqual(data["events"][0]["offside_count"], 3)

            # Second event
            self.assertEqual(data["events"][1]["type"], "Lineout")
            self.assertEqual(data["events"][1]["frame"], 500)
            self.assertEqual(data["events"][1]["offside_count"], 1)

            # Summary stats
            self.assertEqual(data["ruck_count"], 1)
            self.assertEqual(data["lineout_count"], 1)
            self.assertEqual(data["total_offside"], 4)
            self.assertEqual(data["avg_confidence"], "0.92")
        finally:
            os.unlink(report_path)

    def test_parse_report_file_empty(self):
        """parse_report_file should return defaults for an empty file."""
        gui = self._create_gui()

        with tempfile.NamedTemporaryFile(mode='w', suffix='_analysis_report.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write("")
            report_path = f.name

        try:
            data = gui.parse_report_file(report_path)
            self.assertIsNotNone(data)
            self.assertEqual(data["video_name"], "Unknown")
            self.assertEqual(data["total_frames"], 0)
            self.assertEqual(data["events"], [])
        finally:
            os.unlink(report_path)

    def test_parse_report_file_no_events(self):
        """parse_report_file should handle reports with no events."""
        gui = self._create_gui()

        report_content = """Video File: no_events.mp4
Processing Date: 2026-03-29
Total Frames Analysed: 500
Total Events Detected: 0
Total Ruck Events: 0
Total Lineout Events: 0
Total Offside Players Across All Events: 0
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='_analysis_report.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write(report_content)
            report_path = f.name

        try:
            data = gui.parse_report_file(report_path)
            self.assertIsNotNone(data)
            self.assertEqual(data["video_name"], "no_events.mp4")
            self.assertEqual(data["total_events"], 0)
            self.assertEqual(data["events"], [])
            self.assertEqual(data["total_offside"], 0)
        finally:
            os.unlink(report_path)

    def test_parse_report_file_nonexistent(self):
        """parse_report_file should return None for a nonexistent file."""
        gui = self._create_gui()
        result = gui.parse_report_file("/nonexistent/path/report.txt")
        self.assertIsNone(result)

    def test_parse_report_file_malformed_numbers(self):
        """parse_report_file should handle non-numeric values gracefully."""
        gui = self._create_gui()

        report_content = """Video File: malformed.mp4
Total Frames Analysed: not_a_number
Total Events Detected: also_bad
Total Ruck Events: nope
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='_analysis_report.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write(report_content)
            report_path = f.name

        try:
            data = gui.parse_report_file(report_path)
            self.assertIsNotNone(data)
            self.assertEqual(data["video_name"], "malformed.mp4")
            # Should fall back to defaults when parsing fails
            self.assertEqual(data["total_frames"], 0)
            self.assertEqual(data["total_events"], 0)
            self.assertEqual(data["ruck_count"], 0)
        finally:
            os.unlink(report_path)


class TestDisplayReportDetails(unittest.TestCase):
    """Tests for the display_report_details method."""

    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def _create_gui(self):
        with patch('src.gui_app.RugbyOffsideGUI.load_models_async'):
            with patch('src.gui_app.RugbyOffsideGUI.monitor_progress'):
                from gui_app import RugbyOffsideGUI
                gui = RugbyOffsideGUI(self.root)
        return gui

    def test_display_report_details_populates_text(self):
        """display_report_details should populate the report details textbox."""
        gui = self._create_gui()

        report_data = {
            "video_name": "test.mp4",
            "processing_date": "2026-03-29",
            "total_frames": 1000,
            "total_events": 1,
            "events": [{
                "type": "Ruck",
                "frame": 200,
                "timestamp": "00:06.67",
                "confidence": "0.95",
                "offside_count": 2,
            }],
            "ruck_count": 1,
            "lineout_count": 0,
            "total_offside": 2,
            "avg_confidence": "0.95",
        }

        gui.display_report_details(report_data)
        gui.report_details_text.configure(state="normal")
        content = gui.report_details_text.get("1.0", "end").strip()
        gui.report_details_text.configure(state="disabled")

        self.assertIn("test.mp4", content)
        self.assertIn("Ruck", content)
        self.assertIn("0.95", content)
        self.assertIn("Total Ruck Events: 1", content)
        self.assertIn("Total Offside Players: 2", content)

    def test_display_report_details_no_events(self):
        """display_report_details should handle reports with no events."""
        gui = self._create_gui()

        report_data = {
            "video_name": "empty.mp4",
            "processing_date": "2026-03-29",
            "total_frames": 0,
            "total_events": 0,
            "events": [],
            "ruck_count": 0,
            "lineout_count": 0,
            "total_offside": 0,
            "avg_confidence": "N/A",
        }

        gui.display_report_details(report_data)
        gui.report_details_text.configure(state="normal")
        content = gui.report_details_text.get("1.0", "end").strip()
        gui.report_details_text.configure(state="disabled")

        self.assertIn("empty.mp4", content)
        self.assertIn("Total Ruck Events: 0", content)


if __name__ == "__main__":
    unittest.main()
