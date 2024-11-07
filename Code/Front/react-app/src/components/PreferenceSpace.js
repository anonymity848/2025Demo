import React from "react";
import { connect } from "react-redux";
import ConvexHull from "./ConvexHull";
import Legend from "./Legend";

// to visualize the preference space.
class PreferenceSpace extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      index: 0,
      convexHullVertices: props.vertices.map(vert => vert.slice(0, 3))
    };
  }

  updateContextHullVertices = index => {

    const convexHullVertices = this.props.vertices.map(vertext => {
      const vert = [];
      for (let i = 0; i < 3; ++i) {
        vert.push(vertext[(i + index) % vertext.length]);
      }
      return vert;
    });
    this.setState({ convexHullVertices });
  };

  componentDidUpdate(prevProps) {
    if (this.props.vertices !== prevProps.vertices) {
      this.updateContextHullVertices(this.state.index);
    }
  }

  changeStartingIndex = index => {
    this.setState({ index });
    // explicitly pass index to this.updateContextHullVertices
    // because this.state.index is not updated immediately,
    // i.e., this.state.index has still the old value at this line.
    this.updateContextHullVertices(index);
  };

  render() {
    return (
      <div style={{width: '40rem' }}>
        <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-start", width: '40rem' }}>
          <ConvexHull vertices={this.state.convexHullVertices} />
          &nbsp;&nbsp;&nbsp;&nbsp;
          <Legend changeStartingIndex={this.changeStartingIndex} />
        </div>
      </div>
    );
  }
}

const mapStateToProps = ({ vertices }) => ({ vertices });

export default connect(mapStateToProps)(PreferenceSpace);
