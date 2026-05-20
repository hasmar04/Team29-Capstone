# Developed front-end user interface design concept to
improve usability, aesthetic and provide extra features

<!-- PURPOSE

This design was intended to be the final artefact UI
but we rescoped the project, proceeding with the 
standard GUI after we were unable to organise stakeholder
user testing and obtain requriements. We prioritised
essential features and have developed this as conceptual
as a potential for handover future development.

-->

## ELEMENTS

## Main Screen - Pause, play, skip buttons, title, video,
colour legend, button linking to statistics page.

## Statistis Page - Offside detection summary (total rucks, lineouts, player detections, average confidence), Video anaylsis (duration, frames), detailed log, line graph (rucks and lineouts across game), pie chart (rucks compared to lineouts).

## Controller appData.java - Directly draws dynamic data from batch output - lineout reports (generated as videos progress and model outputs data)

## MainController.java - Connects dynamic data and button functionality on main screen.

## statsController.java - Connects dynamic data from lineout report, displaying across text and automatically generating pie chart and line chart.

### All of the code has been developed including the FXML lineout. This can be displayed through scenebuilder or see the 2025-2026/Design handover section for wireframes, Figma link, user testing and future integration.