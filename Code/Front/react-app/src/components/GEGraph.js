import React from "react";
import {
    setActiveComponent,
    setCandidates,
    toggleMask,
    changeMode,
    setLeftPoints,
    prunePoints,
    restart,
    updateConvexHull,
    setRadius
} from "../actions";
import { connect } from "react-redux";
import * as d3 from 'd3';
import io from 'socket.io-client';
import '../css/Graph.css'
import { normalized } from "../utils";
import Histogram from "./Histogram";
import HistogramForR from "./HistogramForR";
import PreferenceSpace from "./PreferenceSpace";
import Stats from "./Stats";
import "../css/textStyle.css"
import TitleGraph from "../imgs/TitleGraph.png";

let dcat = 3, dnum = 4;
let attrData = [];
let numOfQuestion = 0;
let prevLeftPoints = [];


class GEGraph extends React.Component {
    constructor(props) {
        super(props);
        this.ref = React.createRef();

        dcat = Object.values(this.props.mask).slice(0, 3).filter((i) => i === 1).length;
        dnum = Object.values(this.props.mask).slice(3).filter((i) => i === 1).length;
        numOfQuestion = 0; prevLeftPoints = [];
        attrData = [
            { id: 1, name: "Type"},
            { id: 2, name: "Power"},
            { id: 3, name: "Transmission"},
            { id: 4, name: "Price (USD)"},
            { id: 5, name: "Year"},
            { id: 6, name: "Power (HP)"},
            { id: 7, name: "Used KM"}
        ];

        if(this.props.selectedDataset === "nba") {
            attrData = [
                {id: 1, name: "Position"},
                {id: 2, name: "Style"},
                {id: 3, name: "Division"},
                {id: 4, name: "Score"},
                {id: 5, name: "Foul"},
                {id: 6, name: "Rebound"},
                {id: 7, name: "Turnover"}
            ]
        }


        attrData = attrData.filter((attr, index) => this.props.mask[attr.name] === 1);

        let indexes = [];
        for(let i = 0; i < this.props.candidates.length; ++i)
        {
            indexes.push(i);
            prevLeftPoints.push(i);
        }
        this.props.setLeftPoints(indexes);
        this.props.setRadius([1.414213]);

        this.state = {
            pair: [],
            nodeVec: [],
            upper: [],
            lower: [],
            relationVec: [],
            extVec: [],
            numUtilityVec: [],
            showMoreGraph: false,
            showMoreRange: false
        };

        // Initialize socket connection
        this.socket = io.connect('http://39.108.168.228:5000');

        // Set up event listeners for the socket
        this.socket.on('initialized', (data) => {
            this.setState({
                pair: [data.pair1, data.pair2],
                nodeVec: data.node_vectors,
                upper: data.node_upperBound,
                lower: data.node_lowerBound,
                relationVec: data.relation_vectors,
                extVec: data.ext_vectors
            });
            this.props.updateConvexHull(this.state.extVec);
            this.setUtilityVec(this.state.extVec);
            this.drawGraph();
        });

        this.socket.on('send_integer', (data) => {
            this.setState({
                pair: [data.pair1, data.pair2],
                nodeVec: data.node_vectors,
                upper: data.node_upperBound,
                lower: data.node_lowerBound,
                relationVec: data.relation_vectors,
                extVec: data.ext_vectors
            });
            this.props.updateConvexHull(this.state.extVec);
            console.log(this.state.extVec);
            let raArry = this.props.radius;
            raArry.push(this.findMaxDistance(this.state.extVec).toFixed(6));
            this.props.setRadius(raArry);
            console.log(this.props.radius);
            this.setUtilityVec(this.state.extVec);
            let indexes = data.leftpoints;
            const pruneIndexes = prevLeftPoints.filter(point => !indexes.includes(point));
            prevLeftPoints = indexes;
            this.props.prunePoints(pruneIndexes, numOfQuestion);
            if(data.pair1 === data.pair2)
            {
                this.socket.disconnect();
                this.props.showResult();
            }
            else
                this.drawGraph();
        });

        // Example: initialize immediately after setting up the socket
        this.initialize();

    }


    toggleShowMoreGraph = () => {
        this.setState(prevState => ({
            showMoreGraph: !prevState.showMoreGraph
        }));
    };

    toggleShowMoreRange = () => {
        this.setState(prevState => ({
            showMoreRange: !prevState.showMoreRange
        }));
    };

    calculateDistance(point1, point2) {
        return Math.sqrt(Math.pow(point2[0] - point1[0], 2) + Math.pow(point2[1] - point1[1], 2));
    }

    findMaxDistance(points) {
        let maxDistance = 0;

        for (let i = 0; i < points.length; i++) {
            for (let j = i + 1; j < points.length; j++) {
                const distance = this.calculateDistance(points[i], points[j]);
                maxDistance = Math.max(maxDistance, distance);
            }
        }
        return maxDistance;
    }

    // Emit 'initialize' event to the server
    initialize() {
        let smallerBetter = [];
        this.props.attributes.slice(3).map(([attr, config]) => {
            if(this.props.mask[attr])
                smallerBetter.push(config.smallerBetter ? 1 : 0);
        })
        console.log(smallerBetter);
        const normCandidate  = normalized(this.props.candidates, smallerBetter);
        console.log(normCandidate);
        this.socket.emit('initialize', {
            array: normCandidate,
            int1: this.props.candidates.length,
            int2: dcat,
            int3: dnum
        });
    }


    startAgain = () =>
    {
        this.socket.disconnect();
        this.props.restartedAgain();
    }

    drawGraph() {
        const svg = d3.select(this.ref.current);

        svg.selectAll("*").remove();
        console.log(this.state.relationVec);

        const x = this.state.nodeVec.length;  // large nodes count
        const radius = 250; // radius of the circle on which nodes are positioned
        const center = { x: 300, y: 270 };  // center of the SVG and circle

        const largeNodeWidth = 30;
        const largeNodeHeight = 30;

        console.log(this.state.upper);
        console.log(this.state.lower);


        let upperForRec = [], lowerForRec = [];
        for(let i = 0; i < this.state.upper.length; ++i) //for each node
        {
            let bound = 999999;
            for(let j = 0; j < this.state.upper[i].length; ++j)//for each bound
            {
                //calculate the bound
                let bb = 0;
                for(let d = 0; d < this.state.numUtilityVec.length; ++d)
                    bb += this.state.numUtilityVec[d] * this.state.upper[i][j][d];
                if(bb < bound)
                    bound = bb.toFixed(6);
            }
            upperForRec.push(bound);
        }
        for(let i = 0; i < this.state.lower.length; ++i) //for each node
        {
            let bound = -999999;
            for(let j = 0; j < this.state.lower[i].length; ++j)//for each bound
            {
                //calculate the bound
                let bb = 0;
                for(let d = 0; d < this.state.numUtilityVec.length; ++d)
                    bb += this.state.numUtilityVec[d] * this.state.lower[i][j][d];
                if(bb > bound)
                    bound = bb.toFixed(6);
            }
            lowerForRec.push(bound);
        }

        const getCirclePosition = (center, angle, r) => {
            return {
                x: center.x + r * Math.cos(angle),
                y: center.y + r * Math.sin(angle)
            };
        }

        svg.append("text")
            .attr("x", 950)  // Adjust x value to fit your SVG width.
            .attr("y", 150)   // Adjust y value to fit your SVG height.
            .attr("text-anchor", "end")  // Right align text.
            .style("font-size", "15px")  // Font size of the text.
            .text("Put your mouse on the node to see its information.");

        // Draw a rectangle in the center of the SVG.
        const rectangleWidth = 270;
        const rectangleHeight = 180;
        svg.append("rect")
            .attr("x", 780 - rectangleWidth/2)
            .attr("y", center.y - rectangleHeight/2 + 10)
            .attr("width", rectangleWidth)
            .attr("height", rectangleHeight)
            .attr("fill", "#FEFFFE")
            .attr("stroke", "#ccc")
            .attr("stroke-width", 2);

        // Add "Node Information" text in the upper half of the rectangle.
        svg.append("text")
            .attr("x", 780)
            .attr("y", center.y - rectangleHeight/4)  // Adjust y value to place the text in the upper half of the rectangle.
            .attr("text-anchor", "middle")
            .style("font-size", "20px")
            .text("Node Information");

        const largeRectNodes = Array.from({ length: x }).map((_, i) => {
            const angle = (2 * Math.PI / x) * i;  // equally spaced on the circle
            const position = getCirclePosition(center, angle, radius);

            const firstHalf = this.state.nodeVec[i].slice(0, this.state.nodeVec[i].length/2).join(", ");
            const secondHalf = this.state.nodeVec[i].slice(this.state.nodeVec[i].length/2).join(", ");
            let upshow, lwshow;
            if(upperForRec[i] === 999999) upshow = "unknown";
            else upshow = upperForRec[i];
            if(lowerForRec[i] === -999999) lwshow = "unknown";
            else lwshow = lowerForRec[i];
            return {
                x: position.x - largeNodeWidth / 2,  // center the rectangle on the calculated position
                y: position.y - largeNodeHeight / 2,
                label1: firstHalf,
                label2: secondHalf,
                up: upshow,
                lw: lwshow
            };
        });

        const getLineEndpoints = (node1, node2, width, height) => {
            return {
                x1: node1.x + width / 2,
                y1: node1.y + height / 2,
                x2: node2.x + width / 2,
                y2: node2.y + height / 2
            };
        };

        this.state.relationVec.forEach((relation) => {
            const points = getLineEndpoints(largeRectNodes[relation[0]], largeRectNodes[relation[1]], largeNodeWidth, largeNodeHeight);
            svg.append("line")
                .attr("x1", points.x1)
                .attr("y1", points.y1)
                .attr("x2", points.x2)
                .attr("y2", points.y2)
                .attr("stroke", "#ccc")
                .attr("stroke-width", 1);
        });

        const showLabel = (d, show) => {
            if (show) {
                const textElement = svg.append("text")
                    .attr("class", "tempLabel")
                    .attr("x", 660)
                    .attr("y", 110)
                    .attr("text-anchor", "left")
                    .style("font-size", "15px")

                // First line
                textElement.append("tspan")
                    .text("CV1 = {" + d.label2 + "}")
                    .attr("x", 660)  // reset x to keep text centered
                    .attr("dy", "10em");  // use 'dy' to adjust the y position for each line

                // Second line
                textElement.append("tspan")
                    .text("CV2 = {" + d.label1 + "}")
                    .attr("x", 660)  // reset x to keep text centered
                    .attr("dy", "1.2em");  // use 'dy' to adjust the y position for each line

                // Third line
                textElement.append("tspan")
                    .text("S1: Score(" + d.label2 + ")")
                    .attr("x", 660)  // reset x to keep text centered
                    .attr("dy", "1.2em");  // use 'dy' to adjust the y position for each line

                // Fourth line
                textElement.append("tspan")
                    .text("S2: Score(" + d.label1 + ")")
                    .attr("x", 660)  // reset x to keep text centered
                    .attr("dy", "1.2em");  // use 'dy' to adjust the y position for each line

                // Fifth line
                textElement.append("tspan")
                    .text("Upper Bound of S1 - S2: " + d.up)
                    .attr("x", 660)
                    .attr("dy", "1.2em");  // '1.2em' will position this line 1.2 times the font-size below the previous line

                // Sixth line
                textElement.append("tspan")
                    .text("Lower Bound of S1 - S2: " + d.lw)
                    .attr("x", 660)
                    .attr("dy", "1.2em");  // '1.2em' will position this line 1.2 times the font-size below the previous line

            } else {
                svg.selectAll(".tempLabel, .tempLabelBg").remove();
            }
        }

        const relationVecTP = this.state.relationVec;

        let largeRects = [];
        largeRectNodes.forEach((node, idx) => {
            const largeRect = svg.append("rect")
                .attr("x", node.x)
                .attr("y", node.y)
                .attr("width", largeNodeWidth)
                .attr("height", largeNodeHeight)
                .attr("data-id", idx)
                .attr("fill", "#FEFFFE")
                .attr("stroke", "#ccc")
                .attr("stroke-width", 2)
                .on("mouseover", function () {
                    d3.select(this).attr("fill", "#C8D0F8");
                    showLabel(node, true);

                    // Highlight connected nodes
                    const connectedNodes = relationVecTP
                        .filter(rel => rel.includes(idx))
                        .flat()
                        .filter(i => i !== idx);

                    connectedNodes.forEach(j => {
                        largeRects[j].attr("fill", "#c8e4f8");
                    });

                    // Highlight connected lines
                    relationVecTP.forEach((relation) => {
                        if (relation.includes(idx)) {
                            const points = getLineEndpoints(largeRectNodes[relation[0]], largeRectNodes[relation[1]], largeNodeWidth, largeNodeHeight);
                            svg.select(`line[x1="${points.x1}"][y1="${points.y1}"][x2="${points.x2}"][y2="${points.y2}"]`)
                                .attr("stroke", "#94a0b4")
                                .attr("stroke-width", 1.5);
                        }
                    });
                })
                .on("mouseout", function () {
                    d3.select(this).attr("fill", "#FEFFFE");
                    showLabel(node, false);

                    // Reset color of connected nodes
                    const connectedNodes = relationVecTP
                        .filter(rel => rel.includes(idx))
                        .flat()
                        .filter(i => i !== idx);

                    connectedNodes.forEach(j => {
                        largeRects[j].attr("fill", "#FEFFFE");
                    });

                    // Reset color of connected lines
                    relationVecTP.forEach((relation) => {
                        if (relation.includes(idx)) {
                            const points = getLineEndpoints(largeRectNodes[relation[0]], largeRectNodes[relation[1]], largeNodeWidth, largeNodeHeight);
                            svg.select(`line[x1="${points.x1}"][y1="${points.y1}"][x2="${points.x2}"][y2="${points.y2}"]`)
                                .attr("stroke", "#ccc")
                                .attr("stroke-width", 1);
                        }
                    });

                });
            largeRects.push(largeRect);
        });

        const bbox = svg.node().getBBox();
        svg.attr("viewBox", "0 0 1000 560");
        //svg.attr("viewBox", `${bbox.x - 50} ${bbox.y - 50} ${bbox.width + 100} ${bbox.height + 100}`);
    }

    choose = (choice) => {
        numOfQuestion++;
        this.socket.emit('send_integer', {
            integer: choice
        });
    }

    generateRandomNumbers = (x) => {
        let numbers = [0];

        // Generate x - 1 random numbers
        for (let i = 0; i < x - 1; i++) {
            numbers.push(Math.random());
        }

        numbers.push(1);
        numbers.sort((a, b) => a - b);

        let result = [];

        for (let i = 1; i < numbers.length; i++) {
            result.push(numbers[i] - numbers[i - 1]);
        }

        return result;
    }
    setUtilityVec = (Vec) =>
    {
        const coff = this.generateRandomNumbers(Vec.length);
        let resultU = [];
        let sum = 0
        for(let i = 0; i < dnum - 1; ++i)
        {
            let coord = 0
            for(let j = 0; j < Vec.length; ++j)
                coord += coff[j] * Vec[j][i];
            sum += coord;
            resultU.push(coord);
        }
        resultU.push(1 - sum);
        this.setState({
            numUtilityVec: resultU
        });
    }

    regenerateUtilityVec = () =>
    {
        this.setUtilityVec(this.state.extVec);
        this.drawGraph();
    }

    componentDidMount() {
        const container = document.querySelector('.scrollable-container');
        container.scrollLeft = (container.scrollWidth - container.clientWidth) / 2;
    }

    render() {


        let ths = null, trs = null;
        ths = [<th key="Option No.">Option</th>];
        attrData.forEach(attr => {
            ths.push(<th key={attr.id}>{attr.name}</th>);
        });
        ths.push(<th key="chooseButton"/>);

        trs = this.state.pair.map((idx, i) => {
                const tds = [<td key="Option No.">{i + 1}</td>];
                this.props.candidates[idx].forEach((x, j) => {
                    tds.push(<td key={j}>{x}</td>);
                });

                tds.push(
                    <td key="chooseButton">
                        <button type="button"
                                className="choose-btn"
                                style={{ width: "6rem", height: "1.5rem" }}
                                onClick={() => this.choose(i)}>
                            Choose
                        </button>
                    </td>
                );

                return (
                    <tr key={i} data-toggle="tooltip">
                        {tds}
                    </tr>
                );
            }
        );

        let vecForTable = [];
        if(this.state.numUtilityVec.length > 0) {
            const vvvvv = this.state.numUtilityVec;
            console.log(vvvvv);
            attrData.slice(dcat).map((attr, i) => {
                const ele = vvvvv[i].toFixed(2);
                vecForTable.push(
                    <tr>
                        <th key={attr.id}>{attr.name}</th>
                        <th key={attr.id}>{ele}</th>
                    </tr>
                );
            });
        }

        const { showMoreGraph, showMoreRange } = this.state;

        const moreTextGraph = "The following shows a node information template. \n" +
            "Suppose that Tuple 1 has value 'C1' on attribute A and value 'D1' on attribute B. \n" +
            "Suppose that Tuple 2 has value 'C2' on attribute A and value 'D2' on attribute B. \n" +
            "We define: \n" +
            "CV1 = {C1, D1}. \n" +
            "CV2 = {C2, D2}. \n" +
            "S1 = the score of value 'C1' on attribute A + the score of value 'D1' on attribute B, i.e., Score(CV1). \n"+
            "S2 = the score of value 'C2' on attribute A + the score of value 'D2' on attribute B, i.e., Score(CV2). \n\n"+
            "S1 - S2 indicates to what extend you prefer CV1 to CV2. \n" +
            "If S1 - S2 is positive, you prefer CV1 to CV2 (a larger value means larger extend). \n" +
            "If S1 - S2 is negative, you prefer CV1 to CV2 (a smaller value means smaller extend).";



        const moreTextRange = "The weights in the utility vector indicate the importance of numerical attributes to you.";

        return (
            <div className="text-center m-auto" style={{}}>
                <img alt='' src={TitleGraph} style={{ 'width': '100%' }} />
                <p style={{ 'background': 'gainsboro', 'borderRadius': '5px', 'padding': '10px',
                    'fontSize': '16px', 'textAlign': 'left'}}>
                    <span style={{ 'fontSize': '20px'}}><strong>Instruction:</strong></span> You will be asked with
                    multiple questions so that we can learn your preference on tuples, and then, find your favorite one. There are three parts in the following.
                    <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>Part 1:</em> It interacts with you by asking questions.
                    <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>Part 2:</em> It shows the middle results of algorithm GE-Graph.
                    <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>Part 3:</em> It shows the statistic of the performance of algorithm GE-Graph.
                </p>
                <br/>
                <h4 style={{
                    'background': 'linear-gradient(to right, #89bfe9, #5271a6)', // 添加从左到右的颜色渐变
                    'borderRadius': '5px',
                    'textAlign': 'left',
                    'padding': '5px',
                    'boxShadow': '3px 3px 5px rgba(0, 0, 0, 0.5)'  // 添加稍微的阴影效果
                }}>
                    <strong><em>&nbsp;PART 1:</em></strong> Interaction
                </h4>
                <p style={{ 'background': 'gainsboro', 'borderRadius': '5px', 'padding': '5px',
                    'fontSize': '16px', 'textAlign': 'left'}}>
                    <strong>&nbsp;NOTE: </strong>Choose the Tuple you favor more between the following options
                </p>
                <div className="row justify-content-center align-items-center">
                    <div className="col-md-8">
                        <table className="table table-hover text-center">
                            <thead>
                            <tr>{ths}</tr>
                            </thead>
                            <tbody>{trs}</tbody>
                        </table>
                    </div>
                </div>

                <h4 style={{
                    'background': 'linear-gradient(to right, #89bfe9, #5271a6)', // 添加从左到右的颜色渐变
                    'borderRadius': '5px',
                    'textAlign': 'left',
                    'padding': '5px',
                    'boxShadow': '3px 3px 5px rgba(0, 0, 0, 0.5)'  // 添加稍微的阴影效果
                }}>
                    <strong><em>&nbsp;PART 2:</em></strong> Visuals
                </h4>
                <p style={{ 'background': 'gainsboro', 'borderRadius': '5px', 'padding': '10px',
                    'fontSize': '16px', 'textAlign': 'left'}}>
                    <strong>NOTE: </strong>Here shows the relational graph and the numerical utility range. The relational
                    graph stores the information of your preference on the categorical attributes. The numerical utility range
                    stores the information of your preference on the numerical attributes.
                </p>


                <div className="justify-content-center">
                    <div style={{width: '100%', height: '500px', overflow: 'auto', background: '#F6F6F6'}}>
                        <svg ref={this.ref}></svg>
                    </div>
                    <div>
                    <h4 style={{'padding': '10px'}}>Relational Graph</h4>
                    <p className="preserve-newlines">
                        Each node in the relational graph stores your preference on some
                        categorical values. {showMoreGraph ? moreTextGraph : ' '}
                        <button className="toggleButton" onClick={this.toggleShowMoreGraph}>
                            {showMoreGraph ? <span className="foldText">&nbsp;fold</span> : <span className="foldText">...unfold</span>}
                        </button>
                    </p>
                    </div>
                </div>


                <div className="scrollable2-container">
                    &nbsp;&nbsp;&nbsp;&nbsp;
                    <div className="row justify-content-center" style={{width: '40rem' }}>
                        <PreferenceSpace />
                        <div>
                        <h4 style={{'padding': '10px'}}>Numerical Utility Range</h4>
                        <p className="text-t-align">
                            &nbsp;&nbsp;&nbsp;&nbsp; The numerical utility range is the possible domain of your utility
                            vector. {showMoreRange ? moreTextRange: ' '}
                            <button className="toggleButton" onClick={this.toggleShowMoreRange}>
                                {showMoreRange ? <span className="foldText">&nbsp;fold</span> : <span className="foldText">...unfold</span>}
                            </button>
                        </p>
                        </div>
                    </div>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    <div className="row justify-content-center">
                        <div style={{ width: "22rem" }}>
                            <div>
                                <h4><strong>Numerical Utility Vector</strong></h4>
                            </div>
                            <div>
                                <h4 style={{
                                    fontFamily: "'Arial', sans-serif",  // Use any preferred font.
                                    color: '#0cc0df',  // Text color.
                                    backgroundColor: '#f5f5f5',  // Background color.
                                    padding: '10px',  // Padding around text.
                                    border: '2px solid #0cc0df',  // Border around text.
                                    borderRadius: '5px',  // Rounded corners.
                                    display: 'inline-block',  // To make the background color only wrap the text.
                                    boxShadow: '3px 3px 5px rgba(0, 0, 0, 0.2)'  // A subtle shadow.
                                }}>({this.state.numUtilityVec.map(num => num.toFixed(2)).join(", ")})</h4>
                            </div>

                            <table className="table table-hover text-center">
                                <tbody>{vecForTable}</tbody>
                            </table>

                            <div>
                            <button type="button"
                            className="modern-btn"
                            style={{ height: "3rem", width: "12rem" }}
                            onClick={this.regenerateUtilityVec}>
                                Randomly Generate
                            </button>
                            </div>
                            <p className="text-t-align" style={{ 'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', width: "22rem"}}>
                                Click the button to randomly generate another utility vector in the numerical
                                utility range. The bounds in the relational graph will be updated based on the
                                generated utility vector.
                            </p>
                        </div>
                    </div>
                </div>




                <br />
                <h4 style={{
                    'background': 'linear-gradient(to right, #89bfe9, #5271a6)', // 添加从左到右的颜色渐变
                    'borderRadius': '5px',
                    'textAlign': 'left',
                    'padding': '5px',
                    'boxShadow': '3px 3px 5px rgba(0, 0, 0, 0.5)'  // 添加稍微的阴影效果
                }}>
                    <strong><em>&nbsp;PART 3:</em></strong> Statistics
                </h4>
                <p style={{ 'background': 'gainsboro', 'borderRadius': '5px', 'padding': '10px',
                    'fontSize': '16px', 'textAlign': 'left'}}>
                    <strong>NOTE: </strong>Here shows two figures and two tables that demonstrate the Diameter of the
                    numerical utility range, the Candidate Tuple, and the Tuples pruned during the interaction process.
                </p>
                <br/>
                <div>
                    <div className="row justify-content-center" >
                        <div className="col-md-auto">
                            <HistogramForR />
                        </div>
                        <div className="col-md-auto">
                            <Histogram />
                        </div>
                    </div>
                </div>
                <br />
                <p style={{ 'background': 'gainsboro', 'borderRadius': '5px', 'padding': '10px',
                    'fontSize': '16px', 'textAlign': 'left'}}>
                    &nbsp;
                </p>
                <br />
                <div className="scrollable-container">
                <div>
                    <div className="row justify-content-center">
                        <div>
                            <Stats />
                        </div>
                    </div>
                </div>
                </div>

                <br />
                <div>
                    <button type="button"
                            className="modern-btn"
                            style={{ width: "12rem" }}
                            onClick={this.startAgain}>
                        Return to Home
                    </button>
                </div>
                <br />

            </div>
        );
    }
}

const mapStateToProps = ({ candidates, attributes, mask, points, mode, radius, selectedDataset }) => ({
    attributes,
    mask,
    points,
    mode,
    candidates,
    radius,
    selectedDataset
});

const mapDispatchToProps = dispatch => ({
    showResult: () => {
        dispatch(setActiveComponent("ResultGraph"));
    },
    setLeftPoints: indices => {
        dispatch(setLeftPoints(indices));
    },
    setRadius: radius => {
        dispatch(setRadius(radius));
    },
    restartedAgain: () => {
        dispatch(setActiveComponent("Welcome"));
        dispatch(restart());
    },
    prunePoints: (indices, step) => {
        dispatch(prunePoints(indices, step));
    },
    updateConvexHull: vertices => {
        dispatch(updateConvexHull(vertices));
    }
});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(GEGraph);
