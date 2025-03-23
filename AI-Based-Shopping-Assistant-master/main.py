from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter
import speech_recognition as sr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
from sklearn.cluster import MeanShift

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-for-shopping-assistant")

# Load dataset
try:
    dataset = pd.read_csv('attached_assets/person.csv')
    logger.info("Successfully loaded person.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv('attached_assets/dataset.csv')
        logger.info("Successfully loaded dataset.csv")
    except FileNotFoundError:
        logger.error("Could not find dataset files. Please ensure person.csv or dataset.csv exists.")
        dataset = None

# Extract features if dataset is available
if dataset is not None:
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, [0]].values
    names = dataset['Item_names'].tolist()

    # Clustering model
    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    label_counts = Counter(labels)
else:
    X, y, names, labels, cluster_centers, label_counts = None, None, None, None, None, None

# Functions for recommendation
def find_max_cluster():
    """Find the cluster with maximum items"""
    if label_counts is None:
        return 0
        
    max_val = label_counts[0]
    max_cluster = 0
    for i in range(len(label_counts)):
        if label_counts[i] > max_val:
            max_val = label_counts[i]
            max_cluster = i
    return max_cluster

def get_recommendations():
    """Get product recommendations based on clustering"""
    if labels is None or y is None or names is None:
        return []
        
    max_cluster = find_max_cluster()
    suggest_ids = [int(str(y[i])[1:-1]) for i in range(len(labels)) if labels[i] == max_cluster]
    suggest_names = [names[i] for i in range(len(labels)) if labels[i] == max_cluster]
    
    # Create a list of recommendations with IDs and names
    recommendations = []
    for i in range(min(10, len(suggest_ids))):
        recommendations.append({
            'id': suggest_ids[i],
            'name': suggest_names[i]
        })
    
    return recommendations

def get_cluster_visualization():
    """Generate cluster visualization plot as base64 image"""
    if X is None or labels is None or cluster_centers is None:
        return None
        
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    
    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                marker="x", s=150, linewidths=5, zorder=10)
    
    plt.title("Shopping Patterns Cluster Visualization")
    plt.xlabel("Frequency")
    plt.ylabel("Recency")
    
    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_str}"

def get_product_images(keyword):
    """Get paths to product images based on keyword"""
    images = []
    data_dir = "data/"
    
    try:
        # Check if data directory exists
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist")
            # Try attached_assets directory as fallback
            data_dir = os.path.join("attached_assets", "data")
            if not os.path.exists(data_dir):
                return images
                
        # Look through categories
        for category in os.listdir(data_dir):
            if category.lower() == keyword.lower():
                category_path = os.path.join(data_dir, category)
                if os.path.isdir(category_path):
                    for image_file in os.listdir(category_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            images.append(os.path.join(category, image_file))
    except Exception as e:
        logger.error(f"Error searching for product images: {e}")
    
    return images

def get_product_categories():
    """Get list of available product categories"""
    if names is None:
        return ["dress", "shoes", "watch", "laptop", "shirt"]
        
    # Get unique product names
    unique_names = list(set(names))
    return sorted(unique_names)

# Routes
@app.route('/')
def index():
    """Render the main application page"""
    return send_file('templates/index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Handle search requests"""
    data = request.get_json()
    keyword = data.get('query', '').strip()
    
    if not keyword:
        return jsonify({
            'status': 'error',
            'message': 'Empty search query'
        })
    
    # Get product images
    images = get_product_images(keyword)
    
    # Get recommendations
    recommendations = get_recommendations()
    
    # Get visualization
    visualization = get_cluster_visualization()
    
    return jsonify({
        'status': 'success',
        'query': keyword,
        'images': images,
        'imageBaseUrl': '/data',
        'recommendations': recommendations,
        'visualization': visualization
    })

@app.route('/api/categories', methods=['GET'])
def categories():
    """Return available product categories"""
    return jsonify({
        'status': 'success', 
        'categories': get_product_categories()
    })

@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """Serve files from the data directory"""
    return send_from_directory('data', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)