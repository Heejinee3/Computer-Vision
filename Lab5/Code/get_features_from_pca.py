import numpy as np

def get_features_from_pca(feat_num, feature):

    """
    This function loads 'vocab_*.npy' file and
	returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
	:param feature: 'HoG' or 'SIFT'

    :return: an N x feat_num matrix
    """

    vocab = np.load(f'vocab_{feature}.npy')

    # Your code here. You should also change the return value.
    mean = np.mean(vocab, axis = 0)
    shifted_vocab = vocab - mean
    cov = np.cov(shifted_vocab.T)
    v, e=np.linalg.eig(cov)
    i = np.argsort(-v)[:feat_num]
    e = e[:,i]

    return shifted_vocab @ e


