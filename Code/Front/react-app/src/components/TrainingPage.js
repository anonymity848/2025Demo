// src/components/TrainingPage.js
import React, { Component } from "react";
import Chart from "chart.js";
import "../css/TrainingPage.css";
import { setActiveComponent } from "../actions";
import { connect } from "react-redux";

class TrainingPage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            params: {
                threshold: 0.25,
                gamma: 0.8,
                epsilonTrain: 0.5,
                alpha: 0.05,
                maxMemorySize: 5000,
                batchSize: 64,
                actionSpaceSize: 5,
                trainingSize: 1000,
                utility_vector: { Year: 0.25, Price: 0.25, Mileage: 0.25, Tax: 0.25 }
            },
            memory: [],
            timeHistory: [],
            numQuestionHistory: [],
            isTraining: false
        };
        this.timeChartRef = React.createRef();
        this.questionChartRef = React.createRef();
        this.tableContainerRef = React.createRef();
        this.socket = null;
    }

    componentDidUpdate(_, prevState) {
        if (this.state.isTraining && prevState.isTraining !== this.state.isTraining) {
            this.startWebSocket();
        }

        // Initialize charts if they haven't been created
        if (!this.timeChart && !this.questionChart && this.timeChartRef.current && this.questionChartRef.current) {
            this.initializeCharts();
        }

        // Update charts if timeHistory or numQuestionHistory has changed
        if (this.timeChart && this.questionChart) {
            if (prevState.timeHistory !== this.state.timeHistory) {
                this.updateChart(this.timeChart, this.state.timeHistory, "Training Time");
            }
            if (prevState.numQuestionHistory !== this.state.numQuestionHistory) {
                this.updateChart(this.questionChart, this.state.numQuestionHistory, "Number of Questions");
            }
        }

        if (prevState.memory !== this.state.memory) {
            const container = this.tableContainerRef.current;
            container.scrollTop = container.scrollHeight;
        }
        
    }

    componentWillUnmount() {
        if (this.socket) {
            this.socket.close();
        }
    }

    startWebSocket = () => {
        this.socket = new WebSocket("ws://localhost:8000/");
        this.socket.onopen = this.sendTrainingParams;
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "training_data") {
                this.setState({
                    timeHistory: data.time_history,
                    numQuestionHistory: data.num_question_history,
                    memory: data.memory
                });
            }
        };
        this.socket.onclose = () => this.setState({ isTraining: false });
    };

    sendTrainingParams = () => {
        const { params } = this.state;
        const trainingParams = {
            ...params,
            utility_vector: Object.values(params.utility_vector)
        };
        this.socket.send(JSON.stringify({ type: "training", params: trainingParams }));
    };

    initializeCharts = () => {
        const ctxTime = this.timeChartRef.current.getContext("2d");
        const ctxQuestion = this.questionChartRef.current.getContext("2d");

        this.timeChart = new Chart(ctxTime, {
            type: "line",
            data: {
                labels: this.state.timeHistory.map((_, i) => i + 1),
                datasets: [{ label: "Training Time", 
                    data: this.state.timeHistory,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0 }]
            },
            options: { responsive: true }
        });

        this.questionChart = new Chart(ctxQuestion, {
            type: "line",
            data: {
                labels: this.state.numQuestionHistory.map((_, i) => i + 1),
                datasets: [{ label: "Number of Questions", 
                    data: this.state.numQuestionHistory,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0 }]
            },
            options: { responsive: true }
        });
    };

    updateChart = (chart, data, label) => {
        chart.data.labels = data.map((_, i) => i + 1);
        chart.data.datasets[0].data = data;
        chart.data.datasets[0].label = label;
        chart.update();
    };

    handleChange = (e, param, isUtility = false) => {
        const value = parseFloat(e.target.value);
        this.setState((prevState) => ({
            params: isUtility
                ? {
                    ...prevState.params,
                    utility_vector: { ...prevState.params.utility_vector, [param]: value }
                }
                : { ...prevState.params, [param]: value }
        }));
    };

    startTraining = () => {
        this.setState({ isTraining: true });
    };

    stopTraining = () => {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ type: "stop_training" }));
            this.socket.close();
        }
        this.setState({ isTraining: false });
    };

    startAgain = () =>
    {
        if(this.socket)
            this.socket.close()
        this.props.restartedAgain();
    };

    render() {
        const { params, memory, isTraining } = this.state;

        return (
            <div>
                <h1 className="cool-title">TRAINING</h1>

                <div class="settings-container">
                    <h4 class="section-title">Dataset Car</h4>
                    <p style={{ 'width': '1600px', 'background': 'gainsboro', 'borderRadius': '5px',  'padding': '10px',
                        'fontSize': '16px', 'textAlign': 'left', 'margin-left': '5px', 'margin-top': '-10px', 'margin-bottom': '0px'}}>
                        Dataset Car contains numerous cars, each of which is descriped by four attributes, namely year, price, 
                        mileage, and tax. <br/>The value range of each attributes is as follows: year (1997-2020), price (1490-137995), 
                        mileage (1-323000), and tax (1-580).
                    </p>
                </div>
                <div class="settings-container">
                    <h4 class="section-title">Algorithm Parameter Setting</h4>
                    <p style={{ 'width': '1600px', 'background': 'gainsboro', 'borderRadius': '5px',  'padding': '10px',
                        'fontSize': '16px', 'textAlign': 'left', 'margin-left': '5px', 'margin-top': '-10px', 'margin-bottom': '0px'}}>
                        <strong>NOTE: </strong>There are three parts for setting parameters. The first part defines the quality criteria
                        of the returned car. The second part is the utility vector input, which is the training data. The third 
                        part contains training parameters. 
                    </p>
                    <div class="row">
                        <div class="half-section-left">
                            <div className="info-title2">Threshold</div>
                            <div class="vector-row2">
                                <div class="input-group">
                                <label>Regret Ratio Threshold:</label>
                                <input
                                    type="number"
                                    value={params.threshold}
                                    onChange={(e) => this.handleChange(e, "threshold")}
                                    min="0.05"
                                    max="0.3"
                                    step="0.01"
                                />
                                </div>
                            </div>
                        </div>

                        <div class="half-section-right">
                            <div className="info-title2">Utility Vector</div>
                            <div class="vector-row3">
                                {Object.keys(params.utility_vector).map((key) => (
                                    <div key={key} className="input-group">
                                        <label>{key}:</label>
                                        <input
                                            type="number"
                                            value={params.utility_vector[key]}
                                            onChange={(e) => this.handleChange(e, key, true)}
                                            min="0"
                                            max="1"
                                            step="0.05"
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="info-title2">Training Parameter Settings</div>
                    <div className="parameter-back">
                    <div class="vector-row">
                        <div class="input-group">
                            <label>Gamma:</label>
                            <input type="number" value={params.gamma} onChange={(e) => this.handleChange(e, "gamma")} min="0" max="1" step="0.01" />
                        </div>
                        <div class="input-group">
                            <label>Alpha:</label>
                            <input type="number" value={params.alpha} onChange={(e) => this.handleChange(e, "alpha")} min="0" max="1" step="0.01" />
                        </div>
                        <div class="input-group">
                            <label>Max Memory Size:</label>
                            <input type="number" value={params.maxMemorySize} onChange={(e) => this.handleChange(e, "maxMemorySize")} min="1000" max="10000" step="1000" />
                        </div>
                        <div class="input-group">
                            <label>Training Epoch:</label>
                            <input type="number" value={params.trainingSize} onChange={(e) => this.handleChange(e, "trainingSize")} min="100" max="10000" step="100" />
                        </div>
                    </div>
                    <div class="vector-row">
                        <div class="input-group">
                            <label>Epsilon:</label>
                            <input type="number" value={params.epsilonTrain} onChange={(e) => this.handleChange(e, "epsilonTrain")} min="0" max="1" step="0.01" />
                        </div>
                        <div class="input-group">
                            <label>Batch Size:</label>
                            <input type="number" value={params.batchSize} onChange={(e) => this.handleChange(e, "batchSize")} min="16" max="256" step="16" />
                        </div>
                        <div class="input-group">
                            <label>Action Space Size:</label>
                            <input type="number" value={params.actionSpaceSize} onChange={(e) => this.handleChange(e, "actionSpaceSize")} min="1" max="20" step="1" />
                        </div>
                        <div class="input-group">
                        </div>
                    </div>
                    </div>
                </div>

                <div className="button-group">
                    <button className="styled-button" onClick={this.startTraining} disabled={isTraining}>Start Training</button>
                    <button className="styled-button" onClick={this.stopTraining} disabled={!isTraining}>Stop Training</button>
                </div>

                <br/>
                <div class="settings-container">
                    <h4 className="section-title">Training Visualization</h4>
                    <p style={{ 'width': '1600px', 'background': 'gainsboro', 'borderRadius': '5px',  'padding': '10px',
                        'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-5px', 'margin-bottom': '0px'}}>
                        <strong>NOTE: </strong>This part displays the middle results during the training process. The first and 
                        second charts show the middle results related to the interaction. The third table shows the middle 
                        results of the reinforcement training. 
                    </p>
                    <div className="three-column-section2">
                        <div className="column2">
                            <h4 className="info-title2">Time Chart</h4>
                            <p style={{ 'width': '500px', 'background': 'none', 'borderRadius': '5px',  'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-10px', 'margin-bottom': '0px'}}>
                                The total training time after each training epoch. 
                            </p>
                            <canvas ref={this.timeChartRef} id="timeChart"></canvas>
                        </div>

                        <div className="column2">
                            <div className="info-title2">Question Chart</div>
                            <p style={{ 'width': '500px', 'background': 'none', 'borderRadius': '5px',  'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-10px', 'margin-bottom': '0px'}}>
                                The interactive questions needed after each training epoch. 
                            </p>
                            <canvas ref={this.questionChartRef} id="questionChart"></canvas>
                        </div>

                        <div className="column2">
                            <div className="info-title2">Replay Memory</div>
                            <p style={{ 'width': '500px', 'background': 'none', 'borderRadius': '5px',  'padding': '10px',
                                'fontSize': '16px', 'textAlign': 'left', 'margin-top': '-10px', 'margin-bottom': '0px'}}>
                                We use deep Q-learning to train the interactive agent. The table shows
                                the transitions stored in the replay memory. 
                            </p>
                            <div className="table-container2" ref={this.tableContainerRef}>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>State</th>
                                            <th>Action</th>
                                            <th>Reward</th>
                                            <th>Next State</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {memory.map((entry, index) => (
                                            <tr key={index}>
                                                <td>{JSON.stringify(entry[0]).substring(0, 10)}...</td>
                                                <td>{entry[1] + 1}</td>
                                                <td>{entry[2]}</td>
                                                <td>{JSON.stringify(entry[3]).substring(0, 10)}...</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <br/>
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
    }
});

export default connect(mapStateToProps, mapDispatchToProps)(TrainingPage);