import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import { AppLayout } from "./components/layout/AppLayout";
import { SkeletonBlock } from "./components/ui/States";
import Landing from "./pages/Landing";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Customers = lazy(() => import("./pages/Customers"));
const CustomerDetail = lazy(() => import("./pages/CustomerDetail"));
const ModelPage = lazy(() => import("./pages/ModelPage"));
const Actions = lazy(() => import("./pages/Actions"));
const Monitoring = lazy(() => import("./pages/Monitoring"));
const NotFound = lazy(() => import("./pages/NotFound"));

const Fallback = () => (
  <div className="space-y-4" aria-busy="true" aria-label="Carregando">
    <SkeletonBlock className="h-10 w-1/3" />
    <SkeletonBlock className="h-28" />
    <SkeletonBlock className="h-72" />
  </div>
);

export default function App() {
  return (
    <Suspense fallback={<Fallback />}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/clientes" element={<Customers />} />
          <Route path="/clientes/:id" element={<CustomerDetail />} />
          <Route path="/acoes" element={<Actions />} />
          <Route path="/modelo" element={<ModelPage />} />
          <Route path="/monitoramento" element={<Monitoring />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
