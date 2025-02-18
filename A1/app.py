from flask import Flask, request, jsonify, render_template
import numpy as np
from kmeans_model import KmeansModel
from sklearn.datasets import make_blobs

app = Flask(__name__)

def get_X1():
    """Get X1 dataset from notebook implementation"""
    X, _ = make_blobs(cluster_std=1.5, random_state=20, n_samples=500, centers=3)
    X1 = np.dot(X, np.random.RandomState(0).randn(2, 2))
    return X1

def get_X2():
    """Get X2 dataset from notebook implementation"""
    centers = [[4, 7], [9, 9], [9, 2]]
    X2, _ = make_blobs(cluster_std=1.5, random_state=20, n_samples=500, centers=centers)
    X2 = np.dot(X2, np.random.RandomState(0).randn(2, 2))
    return X2

def get_X3():
    """Get X3 dataset from notebook implementation"""
    centers = [[5, 5]]
    X31, _ = make_blobs(cluster_std=1.5, random_state=20, n_samples=200, centers=centers)
    X31 = np.dot(X31, np.array([[1.0, 0], [0, 5.0]]))
    
    X32, _ = make_blobs(cluster_std=1.5, random_state=20, n_samples=200, centers=centers)
    X32 = np.dot(X32, np.array([[5.0, 0], [0, 1.0]]))
    
    centers = [[7, 7]]
    X33, _ = make_blobs(cluster_std=1.5, random_state=20, n_samples=100, centers=centers)
    X33 = np.dot(X33, np.random.RandomState(0).randn(2, 2))
    
    X3 = np.vstack((X31, X32, X33))
    return X3

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get-dataset', methods=['GET'])
def get_dataset():
    try:
        dataset_type = request.args.get('type', 'X1')
        
        if dataset_type == 'X1':
            X = get_X1()
        elif dataset_type == 'X2':
            X = get_X2()
        elif dataset_type == 'X3':
            X = get_X3()
        else:
            X = get_X1()
            
        return jsonify({
            'points': X.tolist()
        })
    except Exception as e:
        print("Error getting dataset:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        data = request.json
        X = np.array(data['points'])
        k = data['k']
        max_iters = data.get('max_iters', 100)
        
        kmeans = KmeansModel(X, k, max_iters)
        labels = kmeans.run()
        
        return jsonify({
            'labels': labels.tolist(),
            'centroids': kmeans.centroids.tolist()
        })
    except Exception as e:
        print("Error in clustering:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)