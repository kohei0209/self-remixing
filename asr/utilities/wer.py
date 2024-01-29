import numpy

wer_config = {"correct": 0, "substitution": 1, "insertion": 2, "deletion": 3}


def go_backtrace(backtrace, i, j):
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0

    while i > 0 or j > 0:
        if backtrace[i][j] == wer_config["correct"]:
            numCor += 1
            i -= 1
            j -= 1
        elif backtrace[i][j] == wer_config["substitution"]:
            numSub += 1
            i -= 1
            j -= 1
        elif backtrace[i][j] == wer_config["insertion"]:
            numIns += 1
            j -= 1
        elif backtrace[i][j] == wer_config["deletion"]:
            numDel += 1
            i -= 1

    return numCor, numSub, numIns, numDel


def get_word_error_rate(r, h):
    """
    Given two list of strings how many word error rate(insert, delete or substitution).
    """

    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    backtrace = numpy.zeros((len(r) + 1, len(h) + 1), dtype=numpy.uint16)
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
                backtrace[0][j] = wer_config["insertion"]
            elif j == 0:
                d[i][0] = i
                backtrace[i][0] = wer_config["deletion"]

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                backtrace[i][j] = wer_config["correct"]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

                if d[i][j] == substitution:
                    backtrace[i][j] = wer_config["substitution"]
                elif d[i][j] == insertion:
                    backtrace[i][j] = wer_config["insertion"]
                elif d[i][j] == deletion:
                    backtrace[i][j] = wer_config["deletion"]

    corr, sub, ins, deli = go_backtrace(backtrace, len(r), len(h))

    wer = float(d[len(r)][len(h)]) / len(r) * 100

    results = {
        "wer": round(wer, 3),
        "cor": corr,
        "sub": sub,
        "ins": ins,
        "del": deli,
    }

    return results


def get_wer(r, h, return_n_errors=False):
    """
    Given two list of strings how many word error rate(insert, delete or substitution).
    """

    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    backtrace = numpy.zeros((len(r) + 1, len(h) + 1), dtype=numpy.uint16)
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
                backtrace[0][j] = wer_config["insertion"]
            elif j == 0:
                d[i][0] = i
                backtrace[i][0] = wer_config["deletion"]

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                backtrace[i][j] = wer_config["correct"]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

                if d[i][j] == substitution:
                    backtrace[i][j] = wer_config["substitution"]
                elif d[i][j] == insertion:
                    backtrace[i][j] = wer_config["insertion"]
                elif d[i][j] == deletion:
                    backtrace[i][j] = wer_config["deletion"]

    if return_n_errors:
        return d[len(r)][len(h)]
    else:
        wer = float(d[len(r)][len(h)]) / len(r) * 100
        return wer
