import { createStore } from "redux";
import rootReducer from "./reducers";
import React from "react";
import { render } from "react-dom";
import { Provider } from "react-redux";
import App from "./components/App";

window.Module = {
  onRuntimeInitialized: async function() {

    const store = createStore(
      rootReducer
      // window.__REDUX_DEVTOOLS_EXTENSION__ && window.__REDUX_DEVTOOLS_EXTENSION__()
    );

    render(
      <Provider store={store}>
        <App />
      </Provider>,
      document.getElementById("root")
    );
  }
};

//////动态创建了一个 <script> 标签，并将其添加到文档的 <body> 元素中，用于异步加载一个名为 "run.js" 的 JavaScript 文件。这样可以在页面加载时并行地加载 JavaScript 代码，从而提高页面的性能和响应速度。
const script = document.createElement("script");
script.src = "run.js";
script.async = true;
document.body.appendChild(script);
