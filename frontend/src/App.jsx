import React, { useState } from 'react';
import PDFUpload from './components/PDFUpload';
import DiffVisualizer from './components/DiffVisualizer';
import SummaryView from './components/SummaryView';
import ExportPDF from './components/ExportPDF';

function App() {
  const [comparisonData, setComparisonData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleUploadComplete = (data) => {
    setComparisonData(data);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-blue-600 text-white shadow-md">
        <div className="container mx-auto py-6 px-4">
          <h1 className="text-3xl font-bold">PDF Diff Tool</h1>
          <p className="mt-2">Compare PDF documents and visualize differences</p>
        </div>
      </header>

      <main className="container mx-auto py-6 px-4">
        <div className="grid grid-cols-1 gap-6">
          {!comparisonData && (
            <PDFUpload
              onUploadComplete={handleUploadComplete}
              setIsLoading={setIsLoading}
            />
          )}

          {isLoading && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
              <p className="mt-4 text-gray-600">Processing documents, please wait...</p>
            </div>
          )}

          {comparisonData && (
            <>
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold">Comparison Results</h2>
                <button
                  onClick={() => setComparisonData(null)}
                  className="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded"
                >
                  Upload New Documents
                </button>
              </div>

              <SummaryView comparisonResult={comparisonData.comparisonResult} />

              <DiffVisualizer
                comparisonResult={comparisonData.comparisonResult}
                doc1Url={comparisonData.doc1Url}
                doc2Url={comparisonData.doc2Url}
              />

              <ExportPDF
                comparisonResult={comparisonData.comparisonResult}
                doc1Name={comparisonData.doc1Name}
                doc2Name={comparisonData.doc2Name}
              />
            </>
          )}
        </div>
      </main>

      <footer className="bg-gray-800 text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>PDF Diff Tool - Interview Assignment</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
