def unzip(iterable):
    return zip(*iterable)


def single(list):
    first = list[0]

    assert (len(list) == 1)

    return first