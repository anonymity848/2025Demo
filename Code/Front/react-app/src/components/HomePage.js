import React from "react";
import { connect } from "react-redux";
import { setActiveComponent } from "../actions";
import "../css/HomePage.css"; // 样式文件

// 简单的欢迎界面，包含标题和两个按钮
class Welcome extends React.Component {
  handleNavigate = (path) => {
    if (path === "/training") {
      this.props.startTraining();
    } else if (path === "/interaction") {
      this.props.startInteraction();
    }
  };

  render() {
    return (
      <div className="container">
        <h1>Interactive Search With Reinforcement Learning</h1>
        <div className="button-group">
          <button onClick={() => this.handleNavigate("/training")}>Training</button>
          <button onClick={() => this.handleNavigate("/interaction")}>Inference</button>
        </div>
      </div>
    );
  }
}

const mapDispatchToProps = (dispatch) => ({
  startTraining: () => dispatch(setActiveComponent("TrainingPage")),
  startInteraction: () => dispatch(setActiveComponent("InteractionPage")),
});

export default connect(
  null,
  mapDispatchToProps
)(Welcome);