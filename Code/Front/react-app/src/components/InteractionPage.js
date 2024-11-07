// src/components/InteractionPage.js
import React, { Component } from "react";
import Chart from "chart.js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import "../css/InteractionPage.css";
import PreferenceSpace from "./PreferenceSpace";
import {
    setActiveComponent,
    updateConvexHull
} from "../actions";
import { connect } from "react-redux";


const TreeNode = ({ node, path, highlightPath }) => {
    if (!node) return null;

    // 检查当前节点路径是否是高亮路径的前缀
    const isHighlighted = highlightPath.slice(0, path.length).join(",") === path.join(",");

    if (node.left && node.right){
        return (
            <ul>
                <li>
                    <a
                        href="#"
                        style={{
                            backgroundColor: isHighlighted ? "#ffb581" : "",
                            color: isHighlighted ? "#000" : ""
                        }}
                    >
                    {node.p1 && node.p2 ? (
                        <>
                        <p>Question</p>
                        <table className="car-table">
                            <thead>
                                <tr>
                                    <th>Car</th>
                                    <th>Year</th>
                                    <th>Price</th>
                                    <th>Mileage</th>
                                    <th>Tax</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>1</td>
                                    {node.p1.map((attribute, index) => (
                                        <td key={index}>{attribute}</td>
                                    ))}
                                </tr>
                                <tr>
                                    <td>2</td>
                                    {node.p2.map((attribute, index) => (
                                        <td key={index}>{attribute}</td>
                                    ))}
                                </tr>
                            </tbody>
                        </table>
                        </>
                    ) : node.p1 ? (
                        <>
                        <p>Returned Car</p>
                        <table className="car-table">
                            <thead>
                                <tr>
                                    <th>Car</th>
                                    <th>Year</th>
                                    <th>Price</th>
                                    <th>Mileage</th>
                                    <th>Tax</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>1</td>
                                    {node.p1.map((attribute, index) => (
                                        <td key={index}>{attribute}</td>
                                    ))}
                                </tr>
                            </tbody>
                        </table>
                        </>
                    ) : (
                        "No Car Information Available"
                    )}
                    </a>
                    <ul>
                        {node.left && (
                            <li>
                                <TreeNode
                                    node={node.left}
                                    path={[...path, "left"]}
                                    highlightPath={highlightPath}
                                />
                            </li>
                        )}
                        {node.right && (
                            <li>
                                <TreeNode
                                    node={node.right}
                                    path={[...path, "right"]}
                                    highlightPath={highlightPath}
                                />
                            </li>
                        )}
                    </ul>
                </li>
            </ul>
        );
    }
    else
    {
        return (
            <ul>
                <li>
                    <a
                        href="#"
                        style={{
                            backgroundColor: isHighlighted ? "#ffb581" : "",
                            color: isHighlighted ? "#000" : ""
                        }}
                    >
                    {node.p1 && node.p2 ? (
                        <>
                        <p>Question</p>
                        <table className="car-table">
                            <thead>
                                <tr>
                                    <th>Car</th>
                                    <th>Year</th>
                                    <th>Price</th>
                                    <th>Mileage</th>
                                    <th>Tax</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>1</td>
                                    {node.p1.map((attribute, index) => (
                                        <td key={index}>{attribute}</td>
                                    ))}
                                </tr>
                                <tr>
                                    <td>2</td>
                                    {node.p2.map((attribute, index) => (
                                        <td key={index}>{attribute}</td>
                                    ))}
                                </tr>
                            </tbody>
                        </table>
                        </>
                    ) : node.p1 ? (
                        <>
                        <p>Terminate <br/>Returned Car</p>
                        <table className="car-table">
                            <thead>
                                <tr>
                                    <th>Car</th>
                                    <th>Year</th>
                                    <th>Price</th>
                                    <th>Mileage</th>
                                    <th>Tax</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>1</td>
                                    {node.p1.map((attribute, index) => (
                                        <td key={index}>{attribute}</td>
                                    ))}
                                </tr>
                            </tbody>
                        </table>
                        </>
                    ) : (
                        "No Car Information Available"
                    )}
                    </a>
                </li>
            </ul>
        );
    }
};



class InteractionPage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            description: "No description available yet.",
            analysis: "No analysis available yet.",
            distanceHistory: [],
            treeData: null,
            highlightPath: [],
            dataGroup1: [],
            dataGroup2: [],
            questionNumber: null,
            path: [],
            extVec: [],
            inferenceComplete: false, 
            hyperplanes: []
        };
        this.distanceChartRef = React.createRef();
        this.utilityRangeChartRef = React.createRef();
        this.treeContainerRef = React.createRef();
        this.socket = null;
        this.distanceChart = null;
        this.descriptionRef = React.createRef();
        this.analysisRef = React.createRef();
    }

    componentDidUpdate(prevProps, prevState) {
        // 滚动 description 到底部
        if (this.state.description !== prevState.description && this.descriptionRef.current) {
            this.descriptionRef.current.scrollTop = this.descriptionRef.current.scrollHeight;
        }
        
        // 滚动 analysis 到底部
        if (this.state.analysis !== prevState.analysis && this.analysisRef.current) {
            this.analysisRef.current.scrollTop = this.analysisRef.current.scrollHeight;
        }
    }

    componentDidMount() {
        this.startWebSocket();
    }

    componentWillUnmount() {
        if (this.socket) this.socket.close();
    }

    startWebSocket = () => {
        this.socket = new WebSocket("ws://localhost:8000/");
        this.socket.onopen = () => {
            console.log("Connected to the server");
            this.socket.send(JSON.stringify({ type: "inference" }));
        };
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketData(data);
        };
        this.socket.onclose = () => console.log("Connection closed");
    };

    handleWebSocketData = (data) => {
        if (data.dist_history) this.updateChart(data.dist_history);

        if (data.tree) {
            this.setState({ treeData: data.tree });
        }

        // 如果接收到路径信息，则高亮路径
        if (data.path) {
            this.setState({ highlightPath: data.path });
        }

        if (data.vertices)
        {
            console.log(data.vertices)
            this.setState({ extVec: data.vertices }, () => {
                    this.props.updateConvexHull(this.state.extVec);
            });
        }

        if (data.hyperplanes)
        {
            this.setState({ hyperplanes: data.hyperplanes });
            if (this.state.questionNumber >= 1)
            {
                this.setState({  analysis: "" });             
            }
        }

        if (data.data_group_1 || data.data_group_2) {
            this.setState({
                dataGroup1: data.data_group_1 || [],
                dataGroup2: data.data_group_2 || [],
                questionNumber: data.integer_value, 
                description: ""
            });
        }

        if (data.description_update) {
            this.setState((prevState) => ({
                description: prevState.description + data.description_update
            }));
        }

        if (data.analysis_update) {
            this.setState((prevState) => ({
                analysis: prevState.analysis + data.analysis_update
            }));
        }

        if (data.analysis_finish) this.setState({ inferenceComplete: true });
        //if (data.path) this.highlightPath(data.path);
    };

    startAgain = () =>
    {
        this.socket.close()
        this.props.restartedAgain();
    }

    updateChart = (distanceHistory) => {
        if (!this.distanceChart) {
            this.distanceChart = new Chart(this.distanceChartRef.current, {
                type: "line",
                data: {
                    labels: distanceHistory.map((_, i) => i),
                    datasets: [{
                        label: "Distance",
                        data: distanceHistory,
                        borderColor: "rgba(75, 192, 192, 1)",
                        borderWidth: 2, 
                        fill: false,
                        tension: 0
                    }]
                },
                options: { 
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Number of Questions' // X轴标签
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Distance' // Y轴标签
                            }
                        }
                    } 
                }
            });
        } else {
            this.distanceChart.data.labels = distanceHistory.map((_, i) => i);
            this.distanceChart.data.datasets[0].data = distanceHistory;
            this.distanceChart.update();
        }
    };

    sendChoiceToServer = (choice) => {
        if (this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ choice }));
        }
    };

    render() {
        const { description, analysis, treeData, highlightPath, dataGroup1, dataGroup2, questionNumber, inferenceComplete, hyperplanes } = this.state;

        return (
            <div>
                <h1 className="cool-title">INFERENCE</h1>
                
                <div className="content-wrapper">
                    {/* 第一列 */}
                    <div className="column left-column">
                        <div className="section-title">Interaction</div>
                        <p style={{ 'width': '1000px', 'background': 'gainsboro', 'borderRadius': '5px',  'padding': '10px',
                            'fontSize': '16px', 'textAlign': 'left', 'margin-left': '5px', 'margin-top': '-30px', 'margin-bottom': '0px'}}>
                            <strong>NOTE: </strong>There are three parts showing the interaction process. In the first part, you can see 
                            the interactive question. The second part provides a description of the two cars in the interactive 
                            question (including similarities and differences). The third part shows a summary of the learned
                            information based on the interaction with you. 
                        </p>
                        <div className="info-title">Interactive Question</div>
                        <div className="question-section">
                        <div className="question-title">{questionNumber ? `Question ${questionNumber}: Please choose the car you prefer.` : "Here is the returned car."}</div>
                        <table className="group-table">
                            {dataGroup2.length > 0 ? (
                                <>
                                    <thead>
                                        <tr>
                                            <th>Car</th>
                                            <th>Year</th>
                                            <th>Price</th>
                                            <th>Mileage</th>
                                            <th>Tax</th>
                                            <th></th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>1</td>
                                            {dataGroup1.map((item, index) => (
                                                <td key={index}>{item}</td>
                                            ))}
                                            <td>
                                                <button
                                                    className="styled-button"
                                                    onClick={() => this.sendChoiceToServer(1)}
                                                >
                                                    Choose
                                                </button>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>2</td>
                                            {dataGroup2.map((item, index) => (
                                                <td key={index}>{item}</td>
                                            ))}
                                            <td>
                                                <button
                                                    className="styled-button"
                                                    onClick={() => this.sendChoiceToServer(2)}
                                                >
                                                    Choose
                                                </button>
                                            </td>
                                        </tr>
                                    </tbody>
                                </>
                            ) : (
                                <>
                                    <thead>
                                        <tr>
                                            <th>Car</th>
                                            <th>Year</th>
                                            <th>Price</th>
                                            <th>Mileage</th>
                                            <th>Tax</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>1</td>
                                            {dataGroup1.map((item, index) => (
                                                <td key={index}>{item}</td>
                                            ))}
                                        </tr>
                                    </tbody>
                                </>
                            )}
                        </table>
                        </div>

                        <div className="info-title">Description of Cars</div>
                        <div id="infoContainer">
                            <div
                                className="info-block"
                                ref={this.descriptionRef}
                                style={{ overflowY: 'auto', maxHeight: '200px' }}
                            >
                                <div>{description}</div>
                            </div>
                        </div>

                        <div class="info-title">Summary of Learned Information</div>
                        <div id="infoContainer">
                            <div
                                className="info-block"
                                ref={this.analysisRef}
                                style={{ overflowY: 'auto', maxHeight: '200px' }}
                            >
                                <div>{analysis}</div>
                            </div>
                        </div>
                    </div>

                    {/* 第二列 */}
                    <div className="column right-column">
                        <div className="section-title">I-Tree</div>
                        <p style={{ 'width': '800px', 'background': 'gainsboro', 'borderRadius': '5px',  'padding': '10px',
                            'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-30px', 'margin-bottom': '0px'}}>
                            <strong>NOTE: </strong>This part displays the I-Tree, which illustrates all possible interaction processes 
                            (i.e., interaction paths). Each internal node represents an interactive question, while each leaf node 
                            contains the final returned car.
                        </p>
                        <div className="tree"> 
                            {treeData && <TreeNode node={treeData} path={[]} highlightPath={highlightPath} />}
                        </div>
                    </div>
                </div>

                <br/>
                <div className="middle-results-section">
                    <h4 className="section-title">Middle Results</h4>
                    <p style={{ 'width': '1880px', 'background': 'gainsboro', 'borderRadius': '5px',  'padding': '10px',
                            'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-6px', 'margin-bottom': '0px'}}>
                            <strong>NOTE: </strong>This part shows the details of the utility range that contains your utility vector. 
                            The first and third figures offer critical information about the utility range, while the second figure 
                            shows a visualization of the utility range. 
                    </p>
                    <div className="three-column-section">
                        <div className="column">
                            <div className="info-title">Distance</div><br/>
                            <p style={{ 'width': '440px', 'background': 'white', 'borderRadius': '5px',  'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-20px', 'margin-bottom': '0px'}}>
                                The outer rectangle is the smallest axis-aligned rectangle containing the utility range. This figure
                                shows the largest Euclidean distance between two vertices of the outer rectangle after each interactive round.
                                The interaction process stops if the distance is smaller than 1.2.
                            </p>
                            <canvas ref={this.distanceChartRef} id="distanceChart"></canvas>
                        </div>

                        
                        <div className="column">
                            <div className="info-title">Utility Range</div><br/>
                            <p style={{ 'width': '800px', 'background': 'white', 'borderRadius': '5px',  'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-20px', 'margin-bottom': '0px'}}>
                                The utility range is the possible domain of your utility vector. Adjust the dimensions below to 
                                visualize it. 
                            </p>
                            <PreferenceSpace />
                        </div>

                        
                        <div className="column">
                            <div className="info-title">Hyperplanes</div><br/>
                            <p style={{ 'width': '350px', 'background': 'white', 'borderRadius': '5px',  'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-20px', 'margin-bottom': '0px'}}>
                                This table shows the hyperplanes that bounds the utility range. 
                            </p>
                            <div className="table-container">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Dim 1</th>
                                            <th>Dim 2</th>
                                            <th>Dim 3</th>
                                            <th>Dim 4</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {hyperplanes.map((entry, index) => (
                                            <tr key={index}>
                                                {entry.map((value, i) => (
                                                    <td key={i} title={value}>{value.toFixed(4)}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                <button className="styled-button" onClick={() => this.startAgain()}>
                    Return
                </button>
            </div>
        );
    }
}

const mapStateToProps = ({}) => ({});

const mapDispatchToProps = dispatch => ({
    restartedAgain: () => {
        dispatch(setActiveComponent("Welcome"));
    },
    updateConvexHull: vertices => dispatch(updateConvexHull(vertices))
});

export default connect(mapStateToProps, mapDispatchToProps)(InteractionPage);
