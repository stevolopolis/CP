import torch
import numpy as np

from scorers.sim_score import *


def ramer_douglas_peucker(points, n_segments):
    """
    Simplify a polyline using the Ramer-Douglas-Peucker algorithm.
    """
    def distance(p1, p2, p):
        """
        Compute the perpendicular distance between point p and the line segment defined by points p1 and p2.
        """
        x1, y1 = p1
        x2, y2 = p2
        x, y = p
        
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        triangle_area = np.abs((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1))
        
        return triangle_area / line_length

    def max_distance_point(points, start, end):
        """
        Find the point with the maximum distance from the line segment defined by points p1 and p2.
        """
        max_distance = -1
        max_point_idx = 0
        
        for i in range(start+1, end):
            d = distance(points[start], points[end], points[i])
            if d > max_distance:
                max_distance = d
                max_point_idx = i
        
        return max_point_idx, max_distance

    if len(points) <= n_segments+1:
        return points
    
    weights = {}
    simplified = []

    def rdp_recursive(start, end, depth):
        if depth not in weights:
            weights[depth] = []
        if end > start + 1:
            max_index, max_weight = max_distance_point(points, start, end)
            weights[depth].append((max_index, max_weight))
            rdp_recursive(start, max_index, depth+1)
            rdp_recursive(max_index, end, depth+1)
        
    rdp_recursive(0, len(points) - 1, 0)

    while len(simplified) < n_segments+1:
        max_depth = min(weights.keys())
        max_weight = max(weights[max_depth], key=lambda x: x[1])
        simplified.append(points[max_weight[0]])
        weights[max_depth].remove(max_weight)
        if len(weights[max_depth]) == 0:
            del weights[max_depth]
    
    simplified = sorted(simplified, key=lambda x: x[0])
    simplified.insert(0, points[0])
    simplified.append(points[-1])

    return simplified


def ramer_douglas_peucker_double_line(points1, points2, n_segments):
    """
    Simplify a polyline using the Ramer-Douglas-Peucker algorithm.
    """
    def distance(p1, p2, p):
        """
        Compute the perpendicular distance between point p and the line segment defined by points p1 and p2.
        """
        x1, y1 = p1
        x2, y2 = p2
        x, y = p
        
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        triangle_area = np.abs((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1))
        
        return triangle_area / line_length, (y - x * (y2-y1) / (x2-x1)) > 0

    def max_distance_point(points1, points2, start, end):
        """
        Find the point with the maximum distance from the line segment defined by points p1 and p2.
        """
        max_distance = -1
        max_point_idx = 0

        end_point = (points2[end] + points1[end]) / 2
        start_point = (points2[start] + points1[start]) / 2
        
        for i in range(start+1, end):
            d1, sign1 = distance(start_point, end_point, points1[i])
            d1, sign1 = distance(start_point, end_point, points1[i])
            if d > max_distance:
                max_distance = d
                max_point_idx = i
        
        return max_point_idx, max_distance


    if len(points) < n_segments+1:
        return points
    
    weights = {}
    simplified = []

    def rdp_recursive(start, end, depth):
        if depth not in weights:
            weights[depth] = []
        if end > start + 1:
            max_index, max_weight = max_distance_point(points, start, end)
            weights[depth].append((max_index, max_weight))
            rdp_recursive(start, max_index, depth+1)
            rdp_recursive(max_index, end, depth+1)
        
    rdp_recursive(0, len(points) - 1, 0)

    while len(simplified) < n_segments+1:
        max_depth = min(weights.keys())
        max_weight = max(weights[max_depth], key=lambda x: x[1])
        simplified.append(points[max_weight[0]])
        weights[max_depth].remove(max_weight)
        if len(weights[max_depth]) == 0:
            del weights[max_depth]
    
    simplified = sorted(simplified, key=lambda x: x[0])
    simplified.insert(0, points[0])
    simplified.append(points[-1])

    return torch.tensor(simplified)


def rdp_scorer(sample: torch.tensor, signal: torch.tensor, n_samples: int,
               slopes: torch.tensor, b: torch.tensor, knot_idx: torch.tensor,
               n_pieces: int, return_points=False):
    """
    Get similarity score with Ramer-Douglas-Peucker.
    
    Parameters:
    - sample (torch.tensor): Sample data points.
    - signal (torch.tensor): Signal data points.
    - n_samples (int): Number of samples.
    - slopes (torch.tensor): Slopes of each segment of the signal.
    - b (torch.tensor): y-intercepts of each segment of the signal.
    - knot_idx (torch.tensor): Indices of the knots. Knots are the endpoints of each segment.
    - n_pieces (int): Number of segments to divide the signal into.
    - return_points (bool): Whether to return the turning points.
    """
    # Convert knot indices to knot x-coordinates
    knots = sample[knot_idx]
    points = [(sample[knot_idx[i]].item(), signal[knot_idx[i]].item()) for i in range(len(knot_idx))]

    # Get turning points from Ramer-Douglas-Peucker
    rdp_points = ramer_douglas_peucker(points, n_pieces)
    # Get h and b from turning points
    rdp_h, rdp_b = get_hb_from_turning_points(rdp_points)
    # Extract x-coordinates from the turning points
    rdp_knots = torch.tensor([int(p[0] * n_samples) for p in rdp_points])
    # Convert knot x-coordinates to indices that correspond to the h(s) and b(s)
    rdp_knots_idx = convert_data_point_to_piece_idx(rdp_knots, knot_idx)

    # Return the similarity score
    score, _, _ = signal_similarity_score_with_turning_points(slopes, b, knots, rdp_knots_idx, pred_hs=rdp_h, pred_bs=rdp_b)
    analytical_score, analytical_h, analytical_b = signal_similarity_score_with_turning_points(slopes, b, knots, rdp_knots_idx)

    if return_points:
        return (score, analytical_score), (rdp_h, analytical_h), (rdp_b, analytical_b), rdp_points

    return (score, analytical_score), (rdp_h, analytical_h), (rdp_b, analytical_b), None