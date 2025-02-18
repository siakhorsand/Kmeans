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

    // Verify all elements are found
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Could not find element with ID: ${key}`);
            return;
        }
    }

    // Initialize state
    let state = {
        currentData: null,
        currentResults: null,
        plotState: {
            showConfidenceEllipses: true,
            showCentroids: true,
            animation: true
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
        updateMethodInfo(elements.methodSelect.value);
        if (state.currentData) {
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
            
            if (!response.ok) throw new Error('Clustering failed');
            
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
        
        const traces = [];
        const method = elements.methodSelect.value;
        const colors = generateColors(Math.max(...state.currentResults.labels) + 1);
        
        // Add points with cluster colors
        state.currentResults.labels.forEach((label, idx) => {
            traces.push({
                x: [state.currentData[idx][0]],
                y: [state.currentData[idx][1]],
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 8,
                    color: colors[label]
                },
                name: `Cluster ${label + 1}`,
                showlegend: idx === 0
            });
        });

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
                name: 'Centroids'
            });
        }

        const layout = {
            title: `${method.toUpperCase()} Clustering Results`,
            xaxis: { title: 'First Principal Component' },
            yaxis: { title: 'Second Principal Component' },
            hovermode: 'closest',
            showlegend: true,
            margin: { t: 50, r: 50, b: 50, l: 50 }
        };

        Plotly.newPlot(elements.plotArea, traces, layout, {
            responsive: true,
            transition: {
                duration: state.plotState.animation ? 500 : 0
            }
        });
    }

    // Update statistics panel
    function updateStatistics() {
        if (!state.currentResults) return;
        
        const labels = state.currentResults.labels;
        const numClusters = Math.max(...labels) + 1;
        
        // Calculate cluster sizes
        const clusterSizes = new Array(numClusters).fill(0);
        labels.forEach(label => clusterSizes[label]++);
        
        // Update stats panel
        elements.statsPanel.innerHTML = `
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
    }

    // Update method information
    function updateMethodInfo(method) {
        document.querySelectorAll('.method-info').forEach(el => el.classList.add('hidden'));
        const infoElement = document.getElementById(`${method}-info`);
        if (infoElement) {
            infoElement.classList.remove('hidden');
        }
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