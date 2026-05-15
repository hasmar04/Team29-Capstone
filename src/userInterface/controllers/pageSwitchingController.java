package userInterface.controllers;

public abstract class pageSwitchingController {

    public void switchToMain() {
        try {
            MainApplication.setRoot("mainScreen");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

}

