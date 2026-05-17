package userInterface.controllers;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.PieChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.ResourceBundle;

public class statsController implements Initializable {

    @FXML
    private Label totalDetections;
    @FXML
    private Label lineoutsLabel;
    @FXML
    private Label rucksLabel;
    @FXML
    private Label confidenceLabel;
    @FXML
    private Label totalFramesLabel;
    @FXML
    private Label totalOffsideLabel;
    @FXML
    private Label videoNameLabel;
    @FXML
    private Label videoDurationLabel;
    @FXML
    private Label frameRateLabel;

    @FXML
    private PieChart pieChart;
    @FXML
    private LineChart<String, Number> lineChart;

    @Override
    public void initialize(URL url, ResourceBundle rb) {
        try {
            File file = new File(AppData.reportPath);

            if (!file.exists()) {
                System.err.println("Report not found at: " + file.getAbsolutePath());
                return;
            }

            int lineoutCount = 0;
            int ruckCount = 0;
            int totalFrames = 0;
            int totalEvents = 0;
            int totalOffside = 0;
            String avgConf = "N/A";
            String videoName = "N/A";

            List<String> lines = Files.readAllLines(file.toPath());

            for (String line : lines) {
                String t = line.trim();

                if (t.startsWith("Video File:")) {
                    videoName = t.split(":", 2)[1].trim();
                } else if (t.startsWith("Total Lineout Events:")) {
                    lineoutCount = Integer.parseInt(t.split(":", 2)[1].trim());
                } else if (t.startsWith("Total Ruck Events:")) {
                    ruckCount = Integer.parseInt(t.split(":", 2)[1].trim());
                } else if (t.startsWith("Total Frames Analysed:")) {
                    totalFrames = Integer.parseInt(t.split(":", 2)[1].trim());
                } else if (t.startsWith("Total Events Detected:")) {
                    totalEvents = Integer.parseInt(t.split(":", 2)[1].trim());
                } else if (t.startsWith("Total Offside Players Across All Events:")) {
                    totalOffside = Integer.parseInt(t.split(":", 2)[1].trim());
                } else if (t.startsWith("Average Lineout Detection Confidence:") ||
                        t.startsWith("Average Ruck Detection Confidence:") ||
                        t.startsWith("Average Detection Confidence:")) {
                    avgConf = t.split(":", 2)[1].trim();
                }
            }

            // Set all labels
            if (totalDetections != null)
                totalDetections.setText(String.valueOf(totalEvents));
            if (lineoutsLabel != null)
                lineoutsLabel.setText(String.valueOf(lineoutCount));
            if (rucksLabel != null)
                rucksLabel.setText(String.valueOf(ruckCount));
            if (confidenceLabel != null)
                confidenceLabel.setText(avgConf);
            if (totalFramesLabel != null)
                totalFramesLabel.setText(String.valueOf(totalFrames));
            if (totalOffsideLabel != null)
                totalOffsideLabel.setText(String.valueOf(totalOffside));
            if (videoNameLabel != null)
                videoNameLabel.setText(videoName);

            // Pie chart
            ObservableList<PieChart.Data> pieData = FXCollections.observableArrayList(
                    new PieChart.Data("Lineouts", lineoutCount),
                    new PieChart.Data("Rucks", ruckCount));
            pieChart.setData(pieData);

            // Line chart
            if (lineChart != null) {
                XYChart.Series<String, Number> series = new XYChart.Series<>();
                series.setName("Detection Count");
                series.getData().add(new XYChart.Data<>("Lineouts", lineoutCount));
                series.getData().add(new XYChart.Data<>("Rucks", ruckCount));
                lineChart.getData().add(series);
            }

        } catch (NumberFormatException e) {
            System.err.println("Parse error: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @FXML
    public void returnToMain() {
        SceneManager.switchScene("mainScreen");
    }
}