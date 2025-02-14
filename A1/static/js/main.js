
let data = [];
let chart;

function initChart() {
    const ctx = document.getElementById('plot').getContext('2d');
    chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                data: [],
                backgroundColor: 'blue'
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'X'
                    }
                },
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Y'
                    }
                }
            }
        }
    });
}

async function selectDataset() {
    const datasetType = document.getElementById('dataset').value;
    try {
        const response = await fetch(`/get-dataset?type=${datasetType}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        
        data = result.points.map(point => ({
            x: point[0],
            y: point[1]
        }));
        
        updateChart(data);
    } catch (error) {
        console.error('Error loading dataset:', error);
    }
}

function updateChart(points, labels = null) {
    const colors = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff'];
    
    if (labels) {
        // Handle clustered data
        const datasets = [];
        const uniqueLabels = [...new Set(labels)];
        uniqueLabels.forEach((label, i) => {
            datasets.push({
                data: points.filter((_, j) => labels[j] === label),
                backgroundColor: colors[i % colors.length],
                label: `Cluster ${label + 1}`
            });
        });
        chart.data.datasets = datasets;
    } else {
        // Handle unclustered data
        chart.data.datasets = [{
            data: points,
            backgroundColor: 'blue',
            label: 'Data Points'
        }];
    }
    
    chart.update();
}

async function runClustering() {
    const k = parseInt(document.getElementById('k').value);
    
    try {
        const response = await fetch('/cluster', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                points: data.map(p => [p.x, p.y]),
                k: k
            })
        });
        
        const result = await response.json();
        
        updateChart(data, result.labels);
        
        // Add centroids
        chart.data.datasets.push({
            data: result.centroids.map(c => ({x: c[0], y: c[1]})),
            backgroundColor: 'black',
            pointRadius: 8,
            pointStyle: 'crossRot',
            label: 'Centroids'
        });
        chart.update();
    } catch (error) {
        console.error('Error running clustering:', error);
    }
}

// Initialize on page load
window.onload = function() {
    initChart();
    selectDataset(); // Load initial dataset
    
    // Add event listener for k slider
    document.getElementById('k').addEventListener('input', function(e) {
        document.getElementById('k-value').textContent = e.target.value;
    });
};