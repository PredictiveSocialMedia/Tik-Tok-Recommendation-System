import React from "react";
import ReactDOM from "react-dom/client";
import { AuthProvider } from "./app/AuthContext";
import { NavigationProvider } from "./app/NavigationContext";
import { App } from "./app/App";
import "./styles/globals.css";
import "./styles/responsive.css";
import "./keyboard";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AuthProvider>
      <NavigationProvider>
        <App />
      </NavigationProvider>
    </AuthProvider>
  </React.StrictMode>
);
