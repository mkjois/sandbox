def edit_dist(s1, s2):
    l1, l2 = len(s1)+1, len(s2)+1
    matrix = [];
    for i in range(l1):
        matrix.append([])
        for j in range(l2):
            matrix[i].append(l1+l2)
    for i in range(l1):
        for j in range(l2):
            if i==0 or j==0:
                matrix[i][j] = i if j==0 else j
            else:
                d_ins = 1+matrix[i-1][j]
                d_del = 1+matrix[i][j-1]
                d_sub = matrix[i-1][j-1] + (0 if s1[i-1]==s2[j-1] else 1)
                matrix[i][j] = min(d_ins, d_del, d_sub)
    return matrix[l1-1][l2-1]
