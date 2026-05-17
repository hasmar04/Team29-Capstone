package userInterface.controllers;

import javafx.fxml.FXML;

public class mainController {

    @FXML
    public void switchToStats() {
        SceneManager.switchScene("statsScreen");
    }
}