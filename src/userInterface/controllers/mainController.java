package userInterface.controllers;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.PieChart;
import javafx.beans.binding.Bindings;

import java.net.URL;
import java.util.ResourceBundle;


public abstract class mainController extends pageSwitchingController{

    @FXML
    public void switchToStats(){
        try{
            MainApplication.setRoot("statsScreen");
        }catch (Exception e){
            e.printStackTrace();
        }
    }

}