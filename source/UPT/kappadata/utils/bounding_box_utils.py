def intersection_area_ijkl(i0, j0, k0, l0, i1, j1, k1, l1):
    # i/j: coordinates of top left corner
    # k/l: coordinates of bottom right corner
    intersection_height = max(0, min(k0, k1) - max(i0, i1))
    intersection_width = max(0, min(l0, l1) - max(j0, j1))
    return intersection_height * intersection_width


def intersection_area_ijhw(i0, j0, h0, w0, i1, j1, h1, w1):
    # i/j: coordinates of top left corner
    # h/w: height/width
    intersection_height = max(0, min(i0 + h0, i1 + h1) - max(i0, i1))
    intersection_width = max(0, min(j0 + w0, j1 + w1) - max(j0, j1))
    return intersection_height * intersection_width
