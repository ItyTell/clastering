# mean-shift clustering algorithm implementation in Python
import numpy as np


def mean_shift(dots: list, bandwidth: float, max_iter: int = 300):
    """
    Perform mean shift clustering on a python list of points.
    """
    dots = np.array(dots)
    n_samples, n_features = dots.shape
    
    # Масив для збереження фінальних позицій (центрів) для кожної точки
    final_positions = np.zeros_like(dots)
    
    # Для кожної точки знаходимо її фінальну позицію (центр кластера)
    for i in range(n_samples):
        current_point = dots[i].copy()
        
        # Ітеративний зсув до локального максимуму щільності  
        for iteration in range(max_iter):
            # Знаходимо точки в межах bandwidth
            distances = np.linalg.norm(dots - current_point, axis=1)
            in_bandwidth = dots[distances <= bandwidth]
            
            if len(in_bandwidth) <= 1:  # Якщо тільки поточна точка або менше
                break
            
            # Зсув до центру мас (зважене середнє)
            new_point = np.mean(in_bandwidth, axis=0)
            
            # Перевірка збіжності
            if np.allclose(current_point, new_point, atol=1e-4):
                break
                
            current_point = new_point
        
        final_positions[i] = current_point
    
    # Групуємо точки за їх фінальними позиціями
    # Використовуємо менший поріг для групування центрів
    cluster_threshold = bandwidth * 0.3
    labels = np.full(n_samples, -1)
    cluster_id = 0
    
    for i in range(n_samples):
        if labels[i] != -1:  # Точка вже призначена
            continue
            
        # Знаходимо всі точки, що збіглися до подібного центру
        current_center = final_positions[i]
        distances_to_center = np.linalg.norm(final_positions - current_center, axis=1)
        
        # Групуємо точки з близькими фінальними позиціями
        similar_points = distances_to_center <= cluster_threshold
        
        # Призначаємо їм однакову мітку
        labels[similar_points] = cluster_id
        cluster_id += 1
    
    return labels