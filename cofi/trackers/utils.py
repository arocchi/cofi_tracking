__author__ = 'Alessio Rocchi'

SAT_TOL = 0.2   # +-20%

def hs_raw_to_cv(raw_data):
    """
    takes a list of H,S thresholds from a json file and transforms
    the raw hue and saturation values into cv values, i.e. maps the ranges
    from 0-360 to 0-180 and from 0-100 to 0-255
    :param raw_data:
    :return: opencv hs threshold
    """
    cv_hsv = raw_data
    for i in range(len(raw_data)):
        raw_data[i]['H'] = list(raw_data[i]['H'])
        raw_data[i]['S'] = list(raw_data[i]['S'])
        for j in range(2):
            raw_data[i]['H'][j] /= 2.0
            raw_data[i]['H'][j] = int(round(raw_data[i]['H'][j]))
            raw_data[i]['S'][j] *= 2.55
            raw_data[i]['S'][j] = int(round(raw_data[i]['S'][j]))
        raw_data[i]['H'] = tuple(raw_data[i]['H'])
        raw_data[i]['S'] = tuple(raw_data[i]['S'])
    return cv_hsv

def hs_optimize(hs_list):
    """
    hs_optimize takes a list of (hue,saturation) tuples, and returns
    an optimized list of hue and saturation ranges.
    Notice these ranges are not to be used by opencv, since they are
    in absolute ranges and not opencv ranges (hue goes from 0 to 360,
    saturation goes from 0 to 100)
    :param hs_list: a list of tuples (hue, saturation) to optimize
    :return: a list of dictionaries {'H':(h_min,h_max),'S':(s_min,s_max)}
    """
    from operator import itemgetter
    import numpy

    # let's find the mean distance between "close" colors:
    # we first need to sort the colors by hue value,
    # so that "similar" colors are close by
    hs = hs_list
    hs.sort(key=(itemgetter(0)))

    # we then compute the distance between all "similar" hues
    h_distance = numpy.array(hs)[:,0] - numpy.array(hs[-1:]+hs[:-1])[:,0]
    h_distance[h_distance < 0]+=360

    # we compute the best distances by removing distances smaller than 0.5*standard deviation
    h_distance_mask = h_distance > (h_distance.mean() - 0.5*h_distance.std())
    n_to_keep = len(h_distance[h_distance_mask])
    print "hue mean spacing:", h_distance.mean()
    # and we sort according to the distances so that cutting the last n elements also
    # results in cutting those hues that would result in better hue spacing
    hs_ = zip(hs,h_distance)
    hs_.sort(key=(itemgetter(1)))
    hs = [data[0] for data in hs_]
    hs = hs[:n_to_keep]

    # we sort again by hue similarity
    hs.sort(key=(itemgetter(0)))

    # we compute again the distance between all "similar" hues
    h_distance = numpy.array(hs)[:,0] - numpy.array(hs[-1:]+hs[:-1])[:,0]
    h_distance[h_distance < 0]+=360
    print "hue mean spacing after filtering:", h_distance.mean()

    hs_range = list()
    for i in range(len(h_distance)):
        h_min, h_max = (hs[i][0]-h_distance[i]/2.0, hs[i][0]+h_distance[i]/2.0)
        if h_min < 0:
            h_min += 360
        if h_max > 360:
            h_max -= 360
        s_min, s_max = (hs[i][1]-SAT_TOL*hs[i][1],hs[i][1]+SAT_TOL*hs[i][1])
        hs_range.append({'H':(h_min,h_max), 'S':(s_min,s_max)})
    return hs_range

if __name__ == "__main__":
    from operator import itemgetter
    import numpy
    import json

    with open('../../markers_v0_raw.json') as data_file:
        hs_raw = json.load(data_file)

    hs = [(data['H'][0], data['S'][0]) for data in hs_raw]

    hs_optimized = hs_optimize(hs)

    print json.dumps(hs_optimized)
    print json.dumps(hs_raw_to_cv(hs_optimized))
    json.dump(hs_optimized, file('../../markers_v0.json', 'w+'))

