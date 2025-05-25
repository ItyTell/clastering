
import numpy as np


def forel(dots: list, bandwidth: float, max_iter: int = 300):
    """
    Perform Forel clustering on a python list of points with a specified bandwidth and maximum number of iterations.
    The algorithm iteratively shifts points towards the mean of points within the bandwidth until convergence or max_iter is reached.
    """
    dots = np.array(dots)
    n_samples, n_features = dots.shape
    labels = np.full(n_samples, -1)
    cluster_centers = []
    cluster_id = 0

    for i in range(n_samples):
        if labels[i] != -1:
            continue
        
        # Починаємо з поточної точки як початкового центру
        current_center = dots[i].copy()
        prev_center = None
        
        # Ітеративний пошук оптимального центру кластера
        for iteration in range(max_iter):
            # Знаходимо всі точки в межах bandwidth від поточного центру
            distances = np.linalg.norm(dots - current_center, axis=1)
            points_in_sphere = distances <= bandwidth
            
            if not np.any(points_in_sphere):
                break
            
            # Обчислюємо новий центр як середнє точок в сфері
            points_in_bandwidth = dots[points_in_sphere]
            new_center = np.mean(points_in_bandwidth, axis=0)
            
            # Перевіряємо збіжність
            if prev_center is not None and np.allclose(new_center, prev_center, atol=1e-6):
                break
            
            prev_center = current_center.copy()
            current_center = new_center
        
        # Після знаходження стабільного центру, призначаємо мітки
        final_distances = np.linalg.norm(dots - current_center, axis=1)
        points_to_assign = final_distances <= bandwidth
        
        # Призначаємо мітки тільки точкам, які ще не належать жодному кластеру
        unassigned_mask = labels == -1
        assignment_mask = points_to_assign & unassigned_mask
        
        if np.any(assignment_mask):
            labels[assignment_mask] = cluster_id
            cluster_centers.append(current_center)
            cluster_id += 1

    return labels