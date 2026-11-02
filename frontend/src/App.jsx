import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Overview from "./views/Overview.jsx";
import Federated from "./views/Federated.jsx";
import YieldPrediction from "./views/YieldPrediction.jsx";
import Explainable from "./views/Explainable.jsx";
import Climate from "./views/Climate.jsx";
import ByState from "./views/ByState.jsx";
import Predict from "./views/Predict.jsx";

const VIEWS = [
  { path: "/overview", label: "Overview", el: <Overview /> },
  { path: "/predict", label: "Predict from CSV", el: <Predict /> },
  { path: "/federated", label: "Federated learning", el: <Federated /> },
  { path: "/yield", label: "Yield prediction", el: <YieldPrediction /> },
  { path: "/explainable", label: "Explainable AI", el: <Explainable /> },
  { path: "/climate", label: "Climate conditions", el: <Climate /> },
  { path: "/state", label: "By state", el: <ByState /> },
];

export default function App() {
  return (
    <>
      <nav className="topnav" aria-label="Views">
        <div className="wrap nav-inner">
          <span className="brand">fedcrop</span>
          <div className="links">
            {VIEWS.map((v) => (
              <NavLink key={v.path} to={v.path}
                className={({ isActive }) => (isActive ? "active" : "")}>
                {v.label}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>
      <main className="wrap">
        <Routes>
          {VIEWS.map((v) => <Route key={v.path} path={v.path} element={v.el} />)}
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </main>
    </>
  );
}
