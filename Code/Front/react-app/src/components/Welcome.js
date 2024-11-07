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
    console.log("Welcome")
    return (
      <div className="text-center m-auto" style={{}}>
        <div className="containerr">
          <h1>Interactive Search With Reinforcement Learning</h1>
          <br/>
          <div className="button-group">
            <button className="styled-button" onClick={() => this.handleNavigate("/training")}>Training</button>
            <button className="styled-button" onClick={() => this.handleNavigate("/interaction")}>Inference</button>
          </div>
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