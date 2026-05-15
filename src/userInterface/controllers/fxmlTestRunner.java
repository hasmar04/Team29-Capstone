package userInterface.controllers;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;


public class fxmlTestRunner extends Application {

    private static Scene scene;

    

    public static void main(String[] args) {
        launch();
    }

    private static Parent loadFXML(String fxml) throws IOException {
        return FXMLLoader.load(
            fxmlTestRunner.class.getResource("/userInterface/" + fxml + ".fxml")
        );
    }
}

