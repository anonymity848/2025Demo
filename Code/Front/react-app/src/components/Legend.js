import React from "react";
import Slider from "react-slick";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAngleUp, faAngleDown } from "@fortawesome/free-solid-svg-icons";
import { connect } from "react-redux";


function Line({ color }) {
  return (
    <div
      style={{
        width: "100%",
        height: "2rem",
        alignItems: "center",
        display: "flex"
      }}
    >
      <div style={{ width: "100%", height: "2px", backgroundColor: color }} />
    </div>
  );
}

function SlideItem({ text }) {
  return (
    <div
      style={{
        height: "2rem",
        alignItems: "center",
        display: "flex",
        justifyContent: "center"
      }}
    >
      {text}
    </div>
  );
}

// show/change the selected attributes for displaying the convex hull
class Legend extends React.Component {
    constructor(props) {
        super(props);

        let attrData = [
            { id: 1, name: "Year"},
            { id: 2, name: "Price"},
            { id: 3, name: "Mileage"},
            { id: 4, name: "Tax"}
        ];

        this.attributes = [];
        attrData.forEach((attr) => {
            this.attributes.push(attr.name);
        });
        console.log(this.attributes);
    }
  render() {
    const settings = {
      dots: false,
      infinite: true,
      speed: 500,
      slidesToShow: 3,
      slidesToScroll: 1,
      vertical: true,
      verticalSwiping: true,
      arrows: false,
      swipeToSlide: true,
      afterChange: this.props.changeStartingIndex
    };
    return (
      <div
        title="Click the arrows to change the selected attributes for displaying the convex hull."
        style={{
          width: "12rem"
        }}
      >
          <br /><br /><br /><br />
        <div
          style={{
            alignItems: "center",
            width: "4rem",
            display: "inline-block",
            verticalAlign: "middle"
          }}
        >
          <Line color="#ff0000" />
          <Line color="#00ff00" />
          <Line color="#0000ff" />
        </div>
        <div
          style={{
            width: "8rem",
            display: "inline-block",
            verticalAlign: "middle"
          }}
        >
          <div
            onClick={() => this.slider.slickPrev()}
            style={{ cursor: "pointer" }}
          >
            <FontAwesomeIcon icon={faAngleUp} />
          </div>
            <Slider {...settings} ref={self => (this.slider = self)}>
                {this.attributes.map((attr, key) => (<SlideItem key={key} text={attr}/>))}
                {this.attributes.map((attr, key) => (<SlideItem key={key} text={attr}/>))}
            </Slider>
          <div
            onClick={() => this.slider.slickNext()}
            style={{ cursor: "pointer" }}
          >
            <FontAwesomeIcon icon={faAngleDown} />
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = ({ mask, attributes, selectedDataset }) => ({
    mask,
    attributes,
    selectedDataset
});

export default connect(mapStateToProps)(Legend);
