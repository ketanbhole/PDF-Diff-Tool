import React from 'react';
import { jsPDF } from 'jspdf';

const ExportPDF = ({ comparisonResult, doc1Name, doc2Name }) => {
  const handleExport = () => {
    const doc = new jsPDF();

    // Add title
    doc.setFontSize(18);
    doc.text('PDF Comparison Results', 14, 22);

    // Add document information
    doc.setFontSize(12);
    doc.text(`Document 1: ${doc1Name || 'First Document'}`, 14, 32);
    doc.text(`Document 2: ${doc2Name || 'Second Document'}`, 14, 38);
    doc.text(`Date: ${new Date().toLocaleDateString()}`, 14, 44);

    // Add summary statistics
    doc.setFontSize(14);
    doc.text('Summary of Changes', 14, 54);

    doc.setFontSize(12);
    doc.text(`Additions: ${comparisonResult.summary.additions}`, 14, 62);
    doc.text(`Deletions: ${comparisonResult.summary.deletions}`, 14, 68);
    doc.text(`Modifications: ${comparisonResult.summary.modifications}`, 14, 74);

    // Save the PDF
    doc.save('pdf-comparison-results.pdf');
  };

  return (
    <div className="mt-6">
      <button
        onClick={handleExport}
        className="bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded flex items-center"
      >
        Export Comparison as PDF
      </button>
    </div>
  );
};

export default ExportPDF;
