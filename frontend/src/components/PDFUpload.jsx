import React, { useState } from 'react';
import axios from 'axios';

const PDFUpload = ({ onUploadComplete }) => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file1 || !file2) {
      setError('Please select two PDF files');
      return;
    }

    // Validate file types
    if (file1.type !== 'application/pdf' || file2.type !== 'application/pdf') {
      setError('Please upload PDF files only');
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file1.size > maxSize || file2.size > maxSize) {
      setError('File size should be less than 10MB');
      return;
    }

    setIsUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    try {
      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      onUploadComplete(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to upload files');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Upload PDF Documents</h2>

      <form onSubmit={handleUpload}>
        <div className="mb-4">
          <label className="block text-gray-700 mb-2">First Document</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile1(e.target.files[0])}
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>

        <div className="mb-4">
          <label className="block text-gray-700 mb-2">Second Document</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile2(e.target.files[0])}
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>

        {error && <p className="text-red-500 mb-4">{error}</p>}

        <button
          type="submit"
          disabled={isUploading}
          className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded disabled:opacity-50"
        >
          {isUploading ? 'Processing...' : 'Compare Documents'}
        </button>
      </form>
    </div>
  );
};

export default PDFUpload;