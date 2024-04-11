import torch


def convert_data_point_to_piece_idx(data_points, knot_idx):
    piece_idx = []
    knot_counter = 0
    
    for data_point in data_points:
        while not (data_point >= knot_idx[knot_counter] and data_point < knot_idx[knot_counter+1]):
            if knot_counter == len(knot_idx)-2:
                break
            knot_counter += 1

        piece_idx.append(knot_counter)

    if len(piece_idx) < len(data_points):
        raise ValueError("Some data points are not within the range of the knots")
        
    return piece_idx


def get_analytical_preds(slopes, b, knots):
    weights = [knots[i+1] - knots[i] for i in range(len(knots)-1)]
    sq_weights = [knots[i+1]**2 - knots[i]**2 for i in range(len(knots)-1)]
    cube_weights = [knots[i+1]**3 - knots[i]**3 for i in range(len(knots)-1)]

    slopes = slopes.to("cuda")
    b = b.to("cuda")
    weights = torch.tensor(weights).to("cuda")
    sq_weights = torch.tensor(sq_weights).to("cuda")
    cube_weights = torch.tensor(cube_weights).to("cuda")

    mean_b = b @ weights
    sq_mean_b = b @ sq_weights
    sq_mean_slope = slopes @ sq_weights
    cube_mean_slope = slopes @ cube_weights
    norm_factor = 1 - (3/4 * sq_weights.sum()**2 / weights.sum() / cube_weights.sum())

    slope_remainders = (mean_b + 1/2*sq_mean_slope) / weights.sum()
    b_remainders = (cube_mean_slope + 3/2*sq_mean_b) / cube_weights.sum()

    pred_slopes = (cube_mean_slope + 3/2*((b - slope_remainders) @ sq_weights)) / cube_weights.sum() / norm_factor
    pred_b = (mean_b + 1/2*((slopes - b_remainders) @ sq_weights)) / weights.sum() / norm_factor 

    return pred_slopes, pred_b


def analytical_loss(slopes, b, pred_slopes, pred_b, knots):
    weights = [abs(knots[i+1] - knots[i]) for i in range(len(knots)-1)]
    sq_weights = [abs(knots[i+1]**2 - knots[i]**2) for i in range(len(knots)-1)]
    cube_weights = [abs(knots[i+1]**3 - knots[i]**3) for i in range(len(knots)-1)]

    slopes = slopes.to("cuda")
    b = b.to("cuda")
    pred_slopes = pred_slopes.to("cuda")
    pred_b = pred_b.to("cuda")
    weights = torch.tensor(weights).to("cuda")
    sq_weights = torch.tensor(sq_weights).to("cuda")
    cube_weights = torch.tensor(cube_weights).to("cuda")

    e1 = (slopes - pred_slopes)**2 @ cube_weights / 3
    e2 = ((slopes - pred_slopes) * (b - pred_b)) @ sq_weights
    e3 = (b - pred_b)**2 @ weights

    return e1 + e2 + e3



def signal_similarity_score(slopes, b, knots, n_pieces):
    knots_per_pieces = len(slopes) // n_pieces
    score = []
    ranges = []
    for i in range(n_pieces):
        if i == n_pieces-1:
            pred_h, pred_b = get_analytical_preds(slopes[i*knots_per_pieces:], b[i*knots_per_pieces:], knots[i*knots_per_pieces:])
            score.append(analytical_loss(slopes[i*knots_per_pieces:],
                                      b[i*knots_per_pieces:],
                                      pred_h, pred_b,
                                      knots[i*knots_per_pieces:]))
            ranges.append(knots[-1] - knots[i*knots_per_pieces])
        else:
            pred_h, pred_b = get_analytical_preds(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                                 b[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                                 knots[i*knots_per_pieces:(i+1)*knots_per_pieces+1])
            score.append(analytical_loss(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                      b[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                      pred_h, pred_b,
                                      knots[i*knots_per_pieces:(i+1)*knots_per_pieces+1]))
            ranges.append(knots[(i+1)*knots_per_pieces] - knots[i*knots_per_pieces])

    score_tensor = torch.tensor(score).to("cuda")
    ranges = torch.tensor(ranges).to("cuda")

    mean_score = score_tensor @ ranges / ranges.sum()

    return mean_score.item()


def signal_similarity_score_with_turning_points(slopes, b, knots, turning_points, pred_hs=None, pred_bs=None):
    score = []
    ranges = []

    new_pred_hs = []
    new_pred_bs = []
    
    for i in range(len(turning_points)-1):
        if i == len(turning_points)-1:
            if pred_hs is None or pred_bs is None:
                pred_h, pred_b = get_analytical_preds(slopes[turning_points[i]:], b[turning_points[i]:], knots[turning_points[i]:])
            else:
                pred_h = pred_hs[i]
                pred_b = pred_bs[i]
            score.append(analytical_loss(slopes[turning_points[i]:],
                                      b[turning_points[i]:],
                                      pred_h, pred_b,
                                      knots[turning_points[i]:]))
            ranges.append(knots[-1] - knots[turning_points[i]])
        else:
            if pred_hs is None or pred_bs is None:
                pred_h, pred_b = get_analytical_preds(slopes[turning_points[i]:turning_points[i+1]],
                                                 b[turning_points[i]:turning_points[i+1]],
                                                 knots[turning_points[i]:turning_points[i+1]+1])
            else:
                pred_h = pred_hs[i]
                pred_b = pred_bs[i]

            new_pred_hs.append(pred_h.cpu())
            new_pred_bs.append(pred_b.cpu())
            score.append(analytical_loss(slopes[turning_points[i]:turning_points[i+1]],
                                      b[turning_points[i]:turning_points[i+1]],
                                      pred_h, pred_b,
                                      knots[turning_points[i]:turning_points[i+1]+1]))
            ranges.append(knots[turning_points[i+1]] - knots[turning_points[i]])

    score_tensor = torch.tensor(score).to("cuda")
    ranges = torch.tensor(ranges).to("cuda")

    mean_score = score_tensor @ ranges / ranges.sum()

    return mean_score.item(), new_pred_hs, new_pred_bs


def get_hb_from_turning_points(turning_points: torch.tensor):
    """
    Get h and b from turning points.

    turning_points.shape = (n_turning_points, 2)
    """
    pred_hs = []
    pred_bs = []
    
    for i in range(len(turning_points)-1):
        # get slope of line segments of two points
        h = (turning_points[i+1, 1] - turning_points[i, 1]) / (turning_points[i+1, 0] - turning_points[i, 0])
        b = turning_points[i, 1] - h * turning_points[i, 0]
        pred_hs.append(h)
        pred_bs.append(b)

    return torch.tensor(pred_hs), torch.tensor(pred_bs)