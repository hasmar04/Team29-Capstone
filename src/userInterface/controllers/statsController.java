package userInterface.controllers;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.PieChart;
import javafx.scene.control.Label;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.ResourceBundle;

public class statsController implements Initializable {

    @FXML private Label totalDetections;
    @FXML private Label lineouts;
    @FXML private Label rucks;
    @FXML private Label confidence;

    @FXML private PieChart pieChart;

    @Override
    public void initialize(URL url, ResourceBundle rb) {

        try {
            File file = new File(AppData.reportPath);

            int lineoutCount = 0;
            int ruckCount = 0;

            List<String> lines = Files.readAllLines(file.toPath());

            for (String line : lines) {
                if (line.contains("Total Lineout Events")) {
                    lineoutCount = Integer.parseInt(line.split(":")[1].trim());
                }
                if (line.contains("Total Ruck Events")) {
                    ruckCount = Integer.parseInt(line.split(":")[1].trim());
                }
            }

            // labels
            lineouts.setText(String.valueOf(lineoutCount));
            rucks.setText(String.valueOf(ruckCount));
            totalDetections.setText(String.valueOf(lineoutCount + ruckCount));

            // pie chart
            ObservableList<PieChart.Data> pieData =
                    FXCollections.observableArrayList(
                            new PieChart.Data("Lineouts", lineoutCount),
                            new PieChart.Data("Rucks", ruckCount)
                    );

            pieChart.setData(pieData);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
