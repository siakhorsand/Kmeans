// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const elements = {
        methodSelect: document.getElementById('clusteringMethod'),
        numClustersInput: document.getElementById('numClusters'),
        numClustersValue: document.getElementById('numClustersValue'),
        runButton: document.getElementById('runButton'),
        statusPanel: document.getElementById('statusPanel'),
        statsPanel: document.getElementById('statsPanel'),
        plotArea: document.getElementById('plotArea')
    };

    // Initialize state
    let state = {
        currentData: null,
        currentResults: null,
        plotState: {
            showConfidenceEllipses: true,
            showCentroids: true
        }
    };

    // Initialize event listeners
    function initializeEventListeners() {
        elements.methodSelect.addEventListener('change', onMethodChange);
        elements.numClustersInput.addEventListener('input', onNumClustersChange);
        elements.runButton.addEventListener('click', runClustering);
    }

    // Handle method change
    function onMethodChange() {
        if (state.currentResults) {
            updatePlot();
        }
    }

    // Handle number of clusters change
    function onNumClustersChange() {
        elements.numClustersValue.textContent = elements.numClustersInput.value;
    }

    // Update status display
    function updateStatus(message, type = 'info') {
        elements.statusPanel.innerHTML = `
            <div class="p-3 rounded ${type === 'error' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'}">
                ${message}
            </div>
        `;
    }

    // Fetch initial data
    async function fetchData() {
        updateStatus('Fetching data...');
        try {
            const response = await fetch('/api/data');
            if (!response.ok) throw new Error('Failed to fetch data');
            
            const result = await response.json();
            state.currentData = result.data;
            
            plotData(state.currentData);
            updateStatus('Data loaded. Select parameters and click "Run Clustering"');
            elements.runButton.disabled = false;
            
        } catch (error) {
            updateStatus('Error loading data: ' + error.message, 'error');
            console.error('Error:', error);
        }
    }

    // Run clustering algorithm
    async function runClustering() {
        if (!state.currentData) {
            updateStatus('No data available', 'error');
            return;
        }
        
        updateStatus('Running clustering...');
        elements.runButton.disabled = true;
        
        try {
            const response = await fetch('/api/cluster', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: state.currentData,
                    method: elements.methodSelect.value,
                    n_clusters: parseInt(elements.numClustersInput.value)
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Clustering failed');
            }
            
            state.currentResults = await response.json();
            updatePlot();
            updateStatistics();
            updateStatus('Clustering completed successfully');
            
        } catch (error) {
            updateStatus('Error running clustering: ' + error.message, 'error');
            console.error('Error:', error);
        } finally {
            elements.runButton.disabled = false;
        }
    }

    // Plot data points
    function plotData(data) {
        const trace = {
            x: data.map(d => d[0]),
            y: data.map(d => d[1]),
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 8,
                color: 'rgba(100, 100, 100, 0.5)'
            },
            name: 'Data Points'
        };

        const layout = {
            title: 'PCA Transformed Data',
            xaxis: { title: 'First Principal Component' },
            yaxis: { title: 'Second Principal Component' },
            hovermode: 'closest',
            showlegend: true,
            margin: { t: 50, r: 50, b: 50, l: 50 }
        };
        
        Plotly.newPlot(elements.plotArea, [trace], layout, {responsive: true});
    }

    // Update plot with clustering results
    function updatePlot() {
        if (!state.currentData || !state.currentResults) return;
        
        const method = elements.methodSelect.value;
        const colors = generateColors(Math.max(...state.currentResults.labels) + 1);
        
        // Group data points by clusters
        const clusters = {};
        state.currentResults.labels.forEach((label, idx) => {
            if (!clusters[label]) {
                clusters[label] = { x: [], y: [] };
            }
            clusters[label].x.push(state.currentData[idx][0]);
            clusters[label].y.push(state.currentData[idx][1]);
        });
        
        // Create traces for each cluster
        const traces = Object.keys(clusters).map(label => ({
            x: clusters[label].x,
            y: clusters[label].y,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 8,
                color: colors[label]
            },
            name: `Cluster ${parseInt(label) + 1}`,
            hovertemplate: 'PCA1: %{x:.2f}<br>PCA2: %{y:.2f}'
        }));
        
        // Add centroids if available
        if (state.currentResults.centroids && state.plotState.showCentroids) {
            traces.push({
                x: state.currentResults.centroids.map(c => c[0]),
                y: state.currentResults.centroids.map(c => c[1]),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 12,
                    symbol: 'x',
                    color: 'black',
                    line: { width: 2 }
                },
                name: 'Centroids',
                hovertemplate: 'Centroid<br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}'
            });
        }
        
        // Add confidence ellipses for EM method
        if (method === 'em' && state.currentResults.covariances && state.plotState.showConfidenceEllipses) {
            state.currentResults.centroids.forEach((centroid, i) => {
                const covMatrix = state.currentResults.covariances[i];
                
                if (!covMatrix || covMatrix[0][0] < 1e-8 || covMatrix[1][1] < 1e-8) return;
                
                try {
                    const scaleFactor = 2.447; // chi-square with 2 df, p=0.95
                    
                    const [eigenvalues, eigenvectors] = calculateEigen(covMatrix);
                    
                    if (eigenvalues[0] <= 1e-10 || eigenvalues[1] <= 1e-10) return;
                    
                    const a = Math.sqrt(eigenvalues[0]) * scaleFactor;
                    const b = Math.sqrt(eigenvalues[1]) * scaleFactor;
                    
                    const angle = Math.atan2(eigenvectors[0][1], eigenvectors[0][0]);
                    
                    const ellipsePoints = generateEllipsePoints(centroid, a, b, angle);
                    
                    traces.push({
                        x: ellipsePoints.x,
                        y: ellipsePoints.y,
                        mode: 'lines',
                        type: 'scatter',
                        line: {
                            color: colors[i],
                            width: 2,
                            dash: 'dash'
                        },
                        name: `Cluster ${i+1} Confidence Region`,
                        showlegend: true,
                        hoverinfo: 'none'
                    });
                } catch (e) {
                    console.error("Error generating ellipse for cluster", i, e);
                }
            });
        }

        const layout = {
            title: `${method.toUpperCase()} Clustering Results`,
            xaxis: { title: 'First Principal Component' },
            yaxis: { title: 'Second Principal Component' },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                x: 1.05,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(255, 255, 255, 0.7)',
                bordercolor: 'rgba(0, 0, 0, 0.1)',
                borderwidth: 1
            },
            margin: { t: 50, r: 100, b: 50, l: 50 }
        };

        Plotly.newPlot(elements.plotArea, traces, layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        });
    }
    
    // Function to generate ellipse points
    function generateEllipsePoints(center, a, b, angle, nPoints = 100) {
        const t = Array.from({ length: nPoints + 1 }, (_, i) => (2 * Math.PI * i) / nPoints);
        const x = t.map(ti => center[0] + a * Math.cos(ti) * Math.cos(angle) - b * Math.sin(ti) * Math.sin(angle));
        const y = t.map(ti => center[1] + a * Math.cos(ti) * Math.sin(angle) + b * Math.sin(ti) * Math.cos(angle));
        return { x, y };
    }
    
    // Function to calculate eigenvalues and eigenvectors
    function calculateEigen(covMatrix) {
        const a = covMatrix[0][0];
        const b = covMatrix[0][1];
        const c = covMatrix[1][0];
        const d = covMatrix[1][1];
        
        // Compute eigenvalues
        const trace = a + d;
        const det = a * d - b * c;
        const discriminant = Math.sqrt(trace * trace - 4 * det);
        
        const eigenvalue1 = (trace + discriminant) / 2;
        const eigenvalue2 = (trace - discriminant) / 2;
        
        // Compute eigenvectors
        let eigenvector1, eigenvector2;
        
        if (Math.abs(b) < 1e-10 && Math.abs(c) < 1e-10) {
            // Diagonal matrix case
            eigenvector1 = [1, 0];
            eigenvector2 = [0, 1];
        } else {
            // For non-diagonal matrices
            if (Math.abs(a - eigenvalue1) > Math.abs(d - eigenvalue1)) {
                eigenvector1 = [b, eigenvalue1 - a];
                eigenvector2 = [b, eigenvalue2 - a];
            } else {
                eigenvector1 = [eigenvalue1 - d, c];
                eigenvector2 = [eigenvalue2 - d, c];
            }
            
            // Normalize eigenvectors
            const norm1 = Math.sqrt(eigenvector1[0] * eigenvector1[0] + eigenvector1[1] * eigenvector1[1]);
            const norm2 = Math.sqrt(eigenvector2[0] * eigenvector2[0] + eigenvector2[1] * eigenvector2[1]);
            
            if (norm1 > 1e-10) eigenvector1 = [eigenvector1[0] / norm1, eigenvector1[1] / norm1];
            if (norm2 > 1e-10) eigenvector2 = [eigenvector2[0] / norm2, eigenvector2[1] / norm2];
        }
        
        return [[eigenvalue1, eigenvalue2], [eigenvector1, eigenvector2]];
    }

    // Update statistics panel
    function updateStatistics() {
        if (!state.currentResults) return;
        
        const labels = state.currentResults.labels;
        const numClusters = Math.max(...labels) + 1;
        
        // Calculate cluster sizes
        const clusterSizes = new Array(numClusters).fill(0);
        labels.forEach(label => clusterSizes[label]++);
        
        // Create stats content based on method
        let statsContent = `
            <div class="space-y-2">
                <p><strong>Number of Clusters:</strong> ${numClusters}</p>
                <p><strong>Points per Cluster:</strong></p>
                <ul class="list-disc pl-5">
                    ${clusterSizes.map((size, i) => 
                        `<li>Cluster ${i + 1}: ${size} points (${(size/labels.length*100).toFixed(1)}%)</li>`
                    ).join('')}
                </ul>
            </div>
        `;
        
        // Add method-specific stats
        const method = elements.methodSelect.value;
        if (method === 'em' && state.currentResults.weights) {
            statsContent += `
                <div class="mt-4">
                    <p><strong>EM Component Weights:</strong></p>
                    <ul class="list-disc pl-5">
                        ${state.currentResults.weights.map((weight, i) => 
                            `<li>Component ${i + 1}: ${(weight*100).toFixed(1)}%</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Update stats panel
        elements.statsPanel.innerHTML = statsContent;
    }

    // Generate colors for clusters
    function generateColors(n) {
        const colors = [];
        for (let i = 0; i < n; i++) {
            colors.push(`hsl(${(i * 360) / n}, 70%, 50%)`);
        }
        return colors;
    }

    // Initialize application
    initializeEventListeners();
    fetchData();
});