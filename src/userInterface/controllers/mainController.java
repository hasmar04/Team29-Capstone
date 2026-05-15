package userInterface.controllers;

public class mainController extends pageSwitchingController {

    public void switchToStats() {
        try {
            MainApplication.setRoot("statsScreen");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
