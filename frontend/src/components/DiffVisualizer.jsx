import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';

// Set worker path for pdf.js
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;

const DiffVisualizer = ({ comparisonResult, doc1Url, doc2Url }) => {
  const [numPages1, setNumPages1] = useState(null);
  const [numPages2, setNumPages2] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Document Comparison</h2>

      <div className="flex justify-between mb-4">
        <button
          onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
          disabled={currentPage <= 1}
          className="bg-gray-200 px-4 py-2 rounded disabled:opacity-50"
        >
          Previous Page
        </button>

        <span>
          Page {currentPage} of {Math.max(numPages1 || 0, numPages2 || 0)}
        </span>

        <button
          onClick={() => setCurrentPage(Math.min(Math.max(numPages1 || 0, numPages2 || 0), currentPage + 1))}
          disabled={currentPage >= Math.max(numPages1 || 0, numPages2 || 0)}
          className="bg-gray-200 px-4 py-2 rounded disabled:opacity-50"
        >
          Next Page
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border rounded p-4">
          <h3 className="font-bold mb-2">Document 1</h3>
          <Document
            file={`http://localhost:5000${doc1Url}`}
            onLoadSuccess={({ numPages }) => setNumPages1(numPages)}
          >
            <Page pageNumber={currentPage} />
          </Document>
        </div>

        <div className="border rounded p-4">
          <h3 className="font-bold mb-2">Document 2</h3>
          <Document
            file={`http://localhost:5000${doc2Url}`}
            onLoadSuccess={({ numPages }) => setNumPages2(numPages)}
          >
            <Page pageNumber={currentPage} />
          </Document>
        </div>
      </div>

      <div className="mt-6 p-4 border rounded bg-gray-50">
        <h3 className="font-bold mb-2">Legend</h3>
        <div className="flex space-x-6">
          <div className="flex items-center">
            <span className="inline-block w-4 h-4 bg-green-200 mr-2"></span>
            <span>Added Content</span>
          </div>
          <div className="flex items-center">
            <span className="inline-block w-4 h-4 bg-red-200 mr-2"></span>
            <span>Removed Content</span>
          </div>
          <div className="flex items-center">
            <span className="inline-block w-4 h-4 bg-yellow-200 mr-2"></span>
            <span>Modified Content</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiffVisualizer;
