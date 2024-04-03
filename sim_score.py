import torch


def get_weighted_preds(slopes, b, knots):
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


def weighted_var(slopes, b, pred_slopes, pred_b, knots):
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
            pred_h, pred_b = get_weighted_preds(slopes[i*knots_per_pieces:], b[i*knots_per_pieces:], knots[i*knots_per_pieces:])
            score.append(weighted_var(slopes[i*knots_per_pieces:],
                                      b[i*knots_per_pieces:],
                                      pred_h, pred_b,
                                      knots[i*knots_per_pieces:]))
            ranges.append(knots[-1] - knots[i*knots_per_pieces])
        else:
            pred_h, pred_b = get_weighted_preds(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                                 b[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                                 knots[i*knots_per_pieces:(i+1)*knots_per_pieces+1])
            score.append(weighted_var(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                      b[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                      pred_h, pred_b,
                                      knots[i*knots_per_pieces:(i+1)*knots_per_pieces+1]))
            ranges.append(knots[(i+1)*knots_per_pieces] - knots[i*knots_per_pieces])

    score_tensor = torch.tensor(score).to("cuda")
    ranges = torch.tensor(ranges).to("cuda")

    mean_score = score_tensor @ ranges / ranges.sum()

    return mean_score.item()


def signal_similarity_score_with_turning_points(slopes, b, knots, turning_points):
    score = []
    ranges = []

    pred_hs = []
    pred_bs = []
    
    for i in range(len(turning_points)-1):
        if i == len(turning_points)-1:
            pred_h, pred_b = get_weighted_preds(slopes[turning_points[i]:], b[turning_points[i]:], knots[turning_points[i]:])
            score.append(weighted_var(slopes[turning_points[i]:],
                                      b[turning_points[i]:],
                                      pred_h, pred_b,
                                      knots[turning_points[i]:]))
            ranges.append(knots[-1] - knots[turning_points[i]])
        else:
            pred_h, pred_b = get_weighted_preds(slopes[turning_points[i]:turning_points[i+1]],
                                                 b[turning_points[i]:turning_points[i+1]],
                                                 knots[turning_points[i]:turning_points[i+1]+1])
            pred_hs.append(pred_h.item())
            pred_bs.append(pred_b.item())
            score.append(weighted_var(slopes[turning_points[i]:turning_points[i+1]],
                                      b[turning_points[i]:turning_points[i+1]],
                                      pred_h, pred_b,
                                      knots[turning_points[i]:turning_points[i+1]+1]))
            ranges.append(knots[turning_points[i+1]] - knots[turning_points[i]])

    score_tensor = torch.tensor(score).to("cuda")
    ranges = torch.tensor(ranges).to("cuda")

    mean_score = score_tensor @ ranges / ranges.sum()

    return mean_score.item(), pred_hs, pred_bs