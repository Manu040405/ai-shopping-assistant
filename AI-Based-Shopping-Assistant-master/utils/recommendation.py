import numpy as np
from collections import Counter
from sklearn.cluster import MeanShift


class ShoppingRecommender:
    """Class for generating product recommendations"""
    
    def __init__(self, dataset):
        """Initialize the recommender with a dataset"""
        self.dataset = dataset
        self.X = dataset.iloc[:, [2, 3]].values  # Frequency and Recency
        self.y = dataset.iloc[:, [0]].values  # Item IDs
        self.names = dataset['Item_names'].tolist()  # Item names
        
        # Initialize and fit the clustering model
        self.model = MeanShift()
        self.model.fit(self.X)
        self.labels = self.model.labels_
        self.cluster_centers = self.model.cluster_centers_
        self.label_counts = Counter(self.labels)
    
    def find_max_cluster(self):
        """Find the cluster with the most items"""
        max_val = self.label_counts[0]
        max_cluster = 0
        for i in range(len(self.label_counts)):
            if self.label_counts[i] > max_val:
                max_val = self.label_counts[i]
                max_cluster = i
        return max_cluster
    
    def get_recommendations(self, limit=10):
        """Get top N recommendations based on clustered patterns"""
        max_cluster = self.find_max_cluster()
        
        # Extract IDs and names for items in the largest cluster
        suggest_ids = [int(str(self.y[i])[1:-1]) for i in range(len(self.labels)) 
                      if self.labels[i] == max_cluster]
        suggest_names = [self.names[i] for i in range(len(self.labels)) 
                        if self.labels[i] == max_cluster]
        
        # Create a list of recommendations with IDs and names
        recommendations = []
        for i in range(min(limit, len(suggest_ids))):
            recommendations.append({
                'id': suggest_ids[i],
                'name': suggest_names[i]
            })
        
        return recommendations
    
    def get_cluster_data(self):
        """Get data for visualization"""
        return {
            'data_points': self.X,
            'labels': self.labels,
            'centers': self.cluster_centers
        }
