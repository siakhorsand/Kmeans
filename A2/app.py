from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import logging
from models import KMeansModel, EMModel, HierarchicalModel
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def load_and_preprocess_data():
    """Load and preprocess the student data"""
    try:
        # Get the current directory of the app.py file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to the data file
        file_path = os.path.join(current_dir, 'PA2_data.csv')
        logger.info(f"Loading data from: {file_path}")
        
        data = pd.read_csv(file_path)
        X = data.iloc[:, 1:].to_numpy()  # Skip the ID column
        X = X / X.max(axis=0)  # Scale to [0,1] range
        return X, data.columns[1:]
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get PCA transformed data"""
    try:
        X, columns = load_and_preprocess_data()
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X)
        return jsonify({
            'data': pca_data.tolist(),
            'columns': columns.tolist()
        })
    except Exception as e:
        logger.error(f"Error in get_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    """Perform clustering using specified algorithm"""
    try:
        data = request.json
        X = np.array(data['data'])
        method = data.get('method', 'kmeans')
        n_clusters = int(data.get('n_clusters', 3))
        
        logger.info(f"Clustering method: {method}, n_clusters: {n_clusters}")
        
        if n_clusters < 2 or n_clusters > 10:
            return jsonify({'error': 'Number of clusters must be between 2 and 10'}), 400
            
        if method == 'kmeans':
            model = KMeansModel(n_clusters=n_clusters)
            result = model.fit_predict(X)
            return jsonify({
                'labels': result['labels'].tolist(),
                'centroids': result['centroids'].tolist()
            })
            
        elif method == 'em':
            model = EMModel(k=n_clusters, max_iters=100)
            result = model.fit_predict(X)
            return jsonify({
                'labels': result['labels'].tolist(),
                'centroids': result['centroids'].tolist(),
                'covariances': result['covariances'].tolist(),
                'weights': result['weights'].tolist()
            })
            
        elif method == 'hierarchical':
            model = HierarchicalModel()
            result = model.fit_predict(X, n_clusters=n_clusters)
            
            # Calculate centroids for hierarchical clustering (for visualization)
            centroids = []
            for i in range(n_clusters):
                cluster_points = X[result['labels'] == i]
                if len(cluster_points) > 0:
                    centroids.append(np.mean(cluster_points, axis=0))
                    
            return jsonify({
                'labels': result['labels'].tolist(),
                'linkage_matrix': result['linkage_matrix'].tolist(),
                'centroids': [c.tolist() for c in centroids]
            })
            
        else:
            return jsonify({'error': 'Invalid clustering method'}), 400
            
    except Exception as e:
        logger.error(f"Error in cluster_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)