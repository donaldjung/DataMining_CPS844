import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Home from './pages/Home';
import Module1_Introduction from './pages/modules/Module1_Introduction';
import Module2_DataPreprocessing from './pages/modules/Module2_DataPreprocessing';
import Module3_ClassificationBasics from './pages/modules/Module3_ClassificationBasics';
import Module4_ClassificationAdvanced from './pages/modules/Module4_ClassificationAdvanced';
import Module5_AssociationAnalysis from './pages/modules/Module5_AssociationAnalysis';
import Module6_ClusterAnalysis from './pages/modules/Module6_ClusterAnalysis';
import Module7_AnomalyDetection from './pages/modules/Module7_AnomalyDetection';
import Playground from './pages/Playground';
import CheatSheet from './pages/CheatSheet';
import Glossary from './pages/Glossary';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="module/1" element={<Module1_Introduction />} />
          <Route path="module/2" element={<Module2_DataPreprocessing />} />
          <Route path="module/3" element={<Module3_ClassificationBasics />} />
          <Route path="module/4" element={<Module4_ClassificationAdvanced />} />
          <Route path="module/5" element={<Module5_AssociationAnalysis />} />
          <Route path="module/6" element={<Module6_ClusterAnalysis />} />
          <Route path="module/7" element={<Module7_AnomalyDetection />} />
          <Route path="playground" element={<Playground />} />
          <Route path="cheatsheet" element={<CheatSheet />} />
          <Route path="glossary" element={<Glossary />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
