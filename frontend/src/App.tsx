// frontend/src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout'; // Import the Layout
import DemoPage from './pages/DemoPage';   // Import the Demo Page
import AboutPage from './pages/AboutPage';  // Import the About Page
import 'react-loading-skeleton/dist/skeleton.css'


function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Wrap all routes within the Layout component */}
        <Route path="/" element={<Layout />}>
          {/* Define child routes - Outlet in Layout will render these */}
          <Route index element={<DemoPage />} /> {/* index route renders at '/' */}
          <Route path="about" element={<AboutPage />} />
          {/* Add more routes here later if needed */}
           {/* <Route path="*" element={<NotFoundPage />} /> */}
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;