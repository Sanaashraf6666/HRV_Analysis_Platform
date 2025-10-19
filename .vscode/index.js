// filepath: c:\Users\hp\Desktop\hrv\.vscode\index.js
import React from "react";
import { createRoot } from "react-dom/client";
import HRVDashboardApp from "../src/hrv-dashboard-app.jsx";

const root = createRoot(document.getElementById("root"));
root.render(<HRVDashboardApp />);