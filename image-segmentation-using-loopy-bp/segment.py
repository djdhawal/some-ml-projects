import sys
import numpy as np
import skimage.io
from scipy.io import loadmat
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from Rgb2Luv import Rgb2Luv

# fixing the random seed
np.random.seed(0)


def prettyPrintArray(array):
    """
    Pretty-prints a numpy array.

    Based on:
    https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
    """
    s = [[str(e) for e in row] for row in array]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:>{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def printGMMComponents(foregroundGMM, backgroundGMM):
    """Prints the means and diagonal covariances for both GMMs."""
    print('Foreground: Gaussian Mixture Model Means')
    prettyPrintArray(foregroundGMM.means_)

    print('\nForeground: Gaussian Mixture Model Covariances (diagonal)')
    prettyPrintArray(foregroundGMM.covariances_)

    print('\nBackground: Gaussian Mixture Model Means')
    prettyPrintArray(backgroundGMM.means_)

    print('\nBackground: Gaussian Mixture Model Covariances (diagonal)')
    prettyPrintArray(backgroundGMM.covariances_)


# (a): fit gmm

def fitGMM(foreground, background):
    """
    Fit a Gaussian Mixture Model to foreground and background pixels.

    Args:
        foreground : (N, 3) array of LUV pixel values labeled as foreground
        background : (M, 3) array of LUV pixel values labeled as background

    Returns:
        foregroundGMM : fitted sklearn GaussianMixture for the foreground
        backgroundGMM : fitted sklearn GaussianMixture for the background
    """
    foregroundGMM = GaussianMixture(
        n_components=5, covariance_type='diag', random_state=0
    ).fit(foreground)
    backgroundGMM = GaussianMixture(
        n_components=5, covariance_type='diag', random_state=0
    ).fit(background)
    return foregroundGMM, backgroundGMM


# (b): superpixel adjacency matrix
def buildAdjacencyMatrix(superpixelMap):
    """
    Build a binary adjacency matrix for the superpixel neighborhood graph.

    Args:
        superpixelMap : (H, W) integer array of superpixel indices (0-indexed)

    Returns:
        adjacency : (S, S) symmetric binary matrix where adjacency[i, j] = 1
                    if superpixels i and j are spatially adjacent, 0 otherwise.
                    S is the total number of superpixels.
    """
    S = int(superpixelMap.max()) + 1
    adjacency = np.zeros((S, S), dtype=np.uint8)

    # a pixel and its right neighbor whose
    # superpixel labels differ are an adjacency.
    left  = superpixelMap[:, :-1]
    right = superpixelMap[:,  1:]
    h_mask = left != right
    a, b = left[h_mask], right[h_mask]
    adjacency[a, b] = 1
    adjacency[b, a] = 1

    # a pixel and its bottom neighbor whose
    # superpixel labels differ are an adjacency.
    top    = superpixelMap[:-1, :]
    bottom = superpixelMap[1:,  :]
    v_mask = top != bottom
    a, b = top[v_mask], bottom[v_mask]
    adjacency[a, b] = 1
    adjacency[b, a] = 1

    # zeroing the diagonal
    np.fill_diagonal(adjacency, 0)
    return adjacency


def saveAdjacencyOverlay(rgbImage, superpixelMap, adjacency, outputPath):
    """
    Save a visualization of the superpixel neighborhood graph overlaid
    on the original image.

    Each superpixel is drawn as a dot at its (row, col) centroid; each
    pair of adjacent superpixels is connected by a line.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    S = adjacency.shape[0]
    H, W = superpixelMap.shape

    # Compute the (row, col) centroid of every superpixel.
    rows, cols = np.indices((H, W))
    flat_idx  = superpixelMap.reshape(-1)
    flat_rows = rows.reshape(-1).astype(np.float64)
    flat_cols = cols.reshape(-1).astype(np.float64)

    sum_r = np.zeros(S); sum_c = np.zeros(S); cnt = np.zeros(S)
    np.add.at(sum_r, flat_idx, flat_rows)
    np.add.at(sum_c, flat_idx, flat_cols)
    np.add.at(cnt,   flat_idx, 1.0)
    cnt = np.maximum(cnt, 1.0)
    cy = sum_r / cnt
    cx = sum_c / cnt

    # Edges (i, j) with i < j drawn from upper triangle of adjacency.
    iu, ju = np.triu_indices(S, k=1)
    mask   = adjacency[iu, ju] > 0
    ei, ej = iu[mask], ju[mask]
    segs = np.stack([
        np.column_stack([cx[ei], cy[ei]]),
        np.column_stack([cx[ej], cy[ej]]),
    ], axis=1)

    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=150)
    ax.imshow(rgbImage)
    ax.add_collection(LineCollection(segs, colors='yellow', linewidths=0.6, alpha=0.85))
    ax.scatter(cx, cy, s=4, c='red', edgecolors='none')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(outputPath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# (c): loopy belief propagation
def computeSuperpixelFeatures(luvImage, superpixelMap, numSuperpixels):
    """
    Compute the mean LUV color vector for each superpixel.

    Args:
        luvImage      : (H, W, 3) LUV image
        superpixelMap : (H, W) integer array of superpixel indices (0-indexed)
        numSuperpixels: total number of superpixels S

    Returns:
        features : (S, 3) array where row i is the mean LUV color of
                   superpixel i.
    """
    flat   = luvImage.reshape(-1, luvImage.shape[2])
    idx    = superpixelMap.reshape(-1)
    sums   = np.zeros((numSuperpixels, flat.shape[1]), dtype=np.float64)
    counts = np.zeros(numSuperpixels, dtype=np.float64)
    np.add.at(sums, idx, flat)
    np.add.at(counts, idx, 1.0)
    counts = np.maximum(counts, 1.0)  # guard against empty superpixels
    return sums / counts[:, None]


def computeNodePotentials(features, foregroundGMM, backgroundGMM):
    """
    Compute the unary (node) potential for each superpixel.

        phi_i(X_i=fg, Y_i) = P(Y_i | GMM_fg)
        phi_i(X_i=bg, Y_i) = P(Y_i | GMM_bg)

    Args:
        features      : (S, 3) array of per-superpixel mean LUV colors
        foregroundGMM : fitted GaussianMixture for the foreground
        backgroundGMM : fitted GaussianMixture for the background

    Returns:
        nodePotentials : (S, 2) array; column 0 = P(Y_i | GMM_fg),
                                       column 1 = P(Y_i | GMM_bg)
    """
    log_p_fg = foregroundGMM.score_samples(features)
    log_p_bg = backgroundGMM.score_samples(features)
    return np.column_stack([np.exp(log_p_fg), np.exp(log_p_bg)])


def runLBP(adjacency, nodePotentials, beta, tol=1e-5, max_iters=1000):
    """
    Run loopy belief propagation on the superpixel cluster graph.

    Edge potential:
        phi_{i,j}(X_i, X_j) = exp(-beta * I(X_i != X_j))

    Messages are computed in log-space and normalized to sum to 1 (in
    probability space) at every step.

    Args:
        adjacency      : (S, S) binary adjacency matrix
        nodePotentials : (S, K) unary potentials (K=2: fg, bg)
        beta           : edge potential parameter (scalar)
        tol            : convergence threshold on max absolute message change
        max_iters      : maximum number of synchronous LBP iterations

    Returns:
        beliefs : (S, K) normalized belief for each superpixel.
    """
    S, K = nodePotentials.shape

    # Log node potentials 
    log_phi = np.log(np.maximum(nodePotentials, 1e-300))

    # Log edge potential: 0 on diagonal, -beta off-diagonal
    log_edge = -beta * (1.0 - np.eye(K))            # (K, K)

    
    src_list, dst_list = [], []
    edge_idx = {}                                   # (i, j) -> directed-edge index
    for i in range(S):
        for j in np.flatnonzero(adjacency[i]):
            edge_idx[(i, int(j))] = len(src_list)
            src_list.append(i)
            dst_list.append(int(j))
    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    E   = src.size
    rev = np.array([edge_idx[(int(dst[e]), int(src[e]))] for e in range(E)],
                   dtype=np.int64)

    if E == 0:
        # No edges: beliefs are just the normalized node potentials.
        beliefs = nodePotentials / nodePotentials.sum(axis=1, keepdims=True)
        return beliefs

   
    # initializing all messages to 1
    
    log_msg = np.full((E, K), -np.log(K))

    for _ in range(max_iters):
        # Sum of all incoming log-messages at each node:
        #   sum_in[i, x] = sum_{k in ne(i)} log_msg[(k -> i), x]
        sum_in = np.zeros((S, K))
        np.add.at(sum_in, dst, log_msg)

        # for edge e = (i -> j):
        #   q[e, x_i] = log_phi[i, x_i] + sum_in[i, x_i] - log_msg[rev[e], x_i]
        # which is the contribution of node i excluding the message from j.
        q = log_phi[src] + sum_in[src] - log_msg[rev]                   # (E, K)

        # new_log_msg[e, x_j]
        #   = logsumexp_{x_i} ( q[e, x_i] + log_edge[x_i, x_j] )
        vals = q[:, :, None] + log_edge[None, :, :]                     # (E, K, K)
        new_log_msg = logsumexp(vals, axis=1)                           # (E, K)

        # normalizing each message
        new_log_msg -= logsumexp(new_log_msg, axis=1, keepdims=True)
        diff = np.max(np.abs(np.exp(new_log_msg) - np.exp(log_msg)))
        log_msg = new_log_msg
        if diff < tol:
            break

    # beliefs: b_i(x) ∝ phi_i(x) * prod_{k in ne(i)} m_{k -> i}(x)
    sum_in = np.zeros((S, K))
    np.add.at(sum_in, dst, log_msg)
    log_belief = log_phi + sum_in
    log_belief -= logsumexp(log_belief, axis=1, keepdims=True)
    return np.exp(log_belief)



def segmentImage(beliefs, superpixelMap):
    """
    Produce a pixel-level label image from superpixel beliefs.

    Args:
        beliefs       : (S, 2) belief array (column 0 = fg, column 1 = bg)
        superpixelMap : (H, W) integer array of superpixel indices (0-indexed)

    Returns:
        labelImage : (H, W) uint8 image with pixel values:
                       128 (foreground)
                       255 (background)
                     (using the visibility-friendly mapping mentioned in
                     the problem statement; for ties, we break toward
                     foreground.)
    """
    # argmax along the K axis -> 0 (fg) or 1 (bg) per superpixel.
    superpixelLabels = np.argmin(-beliefs, axis=1)        # 0 = fg, 1 = bg
    pixelLabels      = superpixelLabels[superpixelMap]    # broadcast via fancy indexing

    labelImage = np.where(pixelLabels == 0, 128, 255).astype(np.uint8)
    return labelImage

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: The function %s is called as follows" % sys.argv[0])
        print("")
        print("    %s originalImage.png superpixelMap.mat"
              " scribbleMask.mat beta" % sys.argv[0])
        sys.exit(0)

    rgbImageFile      = sys.argv[1]
    superpixelMapFile = sys.argv[2]
    scribbleMaskFile  = sys.argv[3]
    beta              = float(sys.argv[4])

    # Load the image and convert to LUV color space
    rgbImage = skimage.io.imread(rgbImageFile)
    luvImage = Rgb2Luv().convert(rgbImage)

    # Load the scribble mask (values: 1=foreground, 2=background, 0=unlabeled)
    scribbleMask   = loadmat(scribbleMaskFile)['scribble_mask']
    foregroundMask = scribbleMask == 1
    backgroundMask = scribbleMask == 2

    foreground = luvImage[foregroundMask]
    background = luvImage[backgroundMask]

    # (a): fit gMMs 
    foregroundGMM, backgroundGMM = fitGMM(foreground, background)
    printGMMComponents(foregroundGMM, backgroundGMM)

    # (b): build the superpixel adjacency matrix
    superpixelMap  = loadmat(superpixelMapFile)['labels']
    numSuperpixels = int(superpixelMap.max()) + 1

    adjacency = buildAdjacencyMatrix(superpixelMap)

    # adjacency-graph visualization overlaid on the original image.
    saveAdjacencyOverlay(rgbImage, superpixelMap, adjacency,
                         'adjacency-overlay.png')

    # (c): lbp
    features       = computeSuperpixelFeatures(luvImage, superpixelMap, numSuperpixels)
    nodePotentials = computeNodePotentials(features, foregroundGMM, backgroundGMM)

    beliefs   = runLBP(adjacency, nodePotentials, beta)
    segmented = segmentImage(beliefs, superpixelMap)
    skimage.io.imsave('segmented-img.png', segmented)

    # (d): beta sweep across {0, 2, 4, 6, 8, 10}
    for b in range(0, 11, 2):
        beliefs_b   = runLBP(adjacency, nodePotentials, float(b))
        segmented_b = segmentImage(beliefs_b, superpixelMap)
        skimage.io.imsave('segmented-beta%d.png' % b, segmented_b)
        print('Saved segmented-beta%d.png' % b)
