import os

def format_libsvm_feature(feature, label, ofname, work_dir='./'):
    if os.path.isfile(os.path.join(work_dir, ofname)):
        print 'file: %s exit' % ofname
        return
    fout = open(os.path.join(work_dir, ofname), 'w')
    #for index, line in enumerate(feature):
    for index in range(feature.shape[0]):
        line = feature.getrow(index)
        line = line.toarray()
        if label is None:
            # fout.write(_libsvm_feature(label[index], line) + '\n')
            # aaa = _libsvm_feature(label[index], line[0])
            fout.write(_libsvm_feature(label[index], line[0]) + '\n')
        else:
            fout.write(_libsvm_feature(0, line[0]) + '\n')
    fout.close()


def _libsvm_feature(label, feature):
    rl_str = str(label)
    for fid, fea_value in enumerate(feature):
        if not fea_value == 0:
            rl_str += ' ' + str(fid + 1) + ':' + str(fea_value)
    return rl_str