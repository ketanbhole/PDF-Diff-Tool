import React from 'react';

const SummaryView = ({ comparisonResult }) => {
  const { summary, differences } = comparisonResult;

  // Calculate percentages for changes
  const totalChanges = summary.additions + summary.deletions + summary.modifications;
  const additionPercentage = totalChanges > 0 ? Math.round((summary.additions / totalChanges) * 100) : 0;
  const deletionPercentage = totalChanges > 0 ? Math.round((summary.deletions / totalChanges) * 100) : 0;
  const modificationPercentage = totalChanges > 0 ? Math.round((summary.modifications / totalChanges) * 100) : 0;

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Summary of Changes</h2>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-green-100 rounded text-center">
          <div className="text-3xl font-bold text-green-700">{summary.additions}</div>
          <div className="text-sm mt-1">Additions</div>
          <div className="text-xs mt-1 text-green-700">{additionPercentage}% of changes</div>
        </div>

        <div className="p-4 bg-red-100 rounded text-center">
          <div className="text-3xl font-bold text-red-700">{summary.deletions}</div>
          <div className="text-sm mt-1">Deletions</div>
          <div className="text-xs mt-1 text-red-700">{deletionPercentage}% of changes</div>
        </div>

        <div className="p-4 bg-yellow-100 rounded text-center">
          <div className="text-3xl font-bold text-yellow-700">{summary.modifications}</div>
          <div className="text-sm mt-1">Modifications</div>
          <div className="text-xs mt-1 text-yellow-700">{modificationPercentage}% of changes</div>
        </div>
      </div>

      <h3 className="font-bold text-lg mb-2">Changes Details</h3>

      {differences.length > 0 ? (
        <div className="max-h-96 overflow-y-auto border rounded">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Original Text
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  New Text
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {differences.map((diff, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      diff.type === 'ADDED' ? 'bg-green-100 text-green-800' :
                      diff.type === 'REMOVED' ? 'bg-red-100 text-red-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {diff.type}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900 max-w-md truncate">
                      {diff.doc1Text || '-'}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900 max-w-md truncate">
                      {diff.doc2Text || '-'}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-gray-500 italic">No differences detected between the documents.</p>
      )}
    </div>
  );
};

export default SummaryView;
