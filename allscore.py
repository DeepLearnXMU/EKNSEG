import subprocess
import os


def f1():
    datapath = '/home/midkiser/workspace/sent_order/sind/sis'

    for dirpath, dirnames, filenames in os.walk('models'):
        print(filenames)
        for f in filenames:
            if 'best' in f:
                f = f.replace('.best.pt', '')
                s, o = subprocess.getstatusoutput('CUDA_VISIBLE_DEVICES=0 python main.py --mode test '
                                                  '--load_from models/{f} '
                                                  '--test {d}/test.raw --writetrans decoding/{f}.order '
                                                  '--beam_size 64 >{f}.tranlog 2>&1'.format(d=datapath, f=f))

import sys
import itertools
import numpy as np
from scipy.stats import kendalltau


def fscore(p, r):
    p = np.mean(p)
    r = np.mean(r)
    return 2 * p * r / (p + r)


def ktau(p, t):
    s_t = set([i for i in itertools.combinations(t, 2)])
    s_p = set([i for i in itertools.combinations(p, 2)])

    cn_2 = len(p) * (len(p) - 1) / 2
    pairs = len(s_p) - len(s_p.intersection(s_t))
    tau = 1 - 2 * pairs / cn_2
    print(tau)


def sta():
    with open(sys.argv[1], 'r') as fr:
        pm_p, pm_r = [], []
        lsr_p, lsr_r = [], []
        head, tail = [], []
        taus = []

        sptaus = []

        for l in fr:
            p, t = l.strip().split('|||')
            p = list(map(int, p.split()))
            t = list(map(int, t.split()))


            # pm
            
            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])
            pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
            pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2*pairs/cn_2
            taus.append(tau)

            t1, _ = kendalltau(t, p)
            sptaus.append(t1)

            # lsr
            if p == t:
                lsr_p.append(1)
                lsr_r.append(1)
            else:
                for sublen in range(len(p)-1, 0, -1):
                    s_t = set([i for i in itertools.combinations(t, sublen)])
                    s_p = set([i for i in itertools.combinations(p, sublen)])
                    match = len(s_t.intersection(s_p))
                    if match != 0:
                        lsr_p.append(sublen/len(p))
                        lsr_r.append(sublen/len(t))
                        break

            # head, tail
            if p[0] == t[0]:
                head.append(1)
            else:
                head.append(0)

            if p[-1] == t[-1]:
                tail.append(1)
            else:
                tail.append(0)

        pm = fscore(pm_p, pm_r)
        lsr = fscore(lsr_p, lsr_r)
        print('pm', pm)
        print('lsr', lsr)
        print('tau {:.2f}'.format(np.mean(taus)))
        print('sptau {:.2f}'.format(np.mean(sptaus)))

        print('head:{:.2%} tail:{:.2%}'.format(np.mean(head), np.mean(tail)))

def headtail():
    with open(sys.argv[1], 'r') as fr:
        head, tail = [], []

        for l in fr:
            p, t = l.strip().split('|||')
            p = list(map(int, p.split()))
            t = list(map(int, t.split()))

            # head, tail
            if p[0] == t[0]:
                head.append(1)
            else:
                head.append(0)

            if p[-1] == t[-1]:
                tail.append(1)
            else:
                tail.append(0)

        print('head:{:.2%} tail:{:.2%}'.format(np.mean(head), np.mean(tail)))



if __name__ == '__main__':
    sta()
    #headtail()

