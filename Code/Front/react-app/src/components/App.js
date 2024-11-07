import React from "react";
import Welcome from "./Welcome";
import TrainingPage from "./TrainingPage";
import InteractionPage from "./InteractionPage";
import { connect } from "react-redux";

function App({ activeComponent }) {
  switch (activeComponent) {
    case "Welcome":
      return <Welcome />;
    case "TrainingPage":
      return <TrainingPage />;
    case "InteractionPage":
      return <InteractionPage />;
    default:
      return <Welcome />;
  }
}

const mapStateToProps = ({ activeComponent }) => ({
  activeComponent
});

export default connect(mapStateToProps)(App);
