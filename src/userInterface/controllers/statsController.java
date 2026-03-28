package userInterface.controllers;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.PieChart;
import javafx.beans.binding.Bindings;

import java.net.URL;
import java.util.ResourceBundle;

public class statsController implements Initializable {

    @FXML
    private PieChart pieChart;

    //Manual graph data adding - Replace with dynamic (testing)
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        ObservableList<PieChart.Data> pieChartData =
                FXCollections.observableArrayList(
                        new PieChart.Data("Lineouts", 4),
                        new PieChart.Data("Rucks", 3),
                        new PieChart.Data("Other", 4)
                );

        pieChartData.forEach(data ->
                data.nameProperty().bind(
                        Bindings.concat(data.getName(), " amount: ", data.pieValueProperty())
                )
        );

        pieChart.setData(pieChartData);
    }
}