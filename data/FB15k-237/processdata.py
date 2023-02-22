import json



e2id = json.load(open('ent2id.json'))
r2id = json.load(open('rel2id.json'))
#
# f_n_11 = open('name1-1.txt', 'r').read().split('\n')
# f_n_11.remove(f_n_11[-1])
# f_n_1n = open('name1-n.txt', 'r').read().split('\n')
# f_n_1n.remove(f_n_1n[-1])
# f_n_n1 = open('namen-1.txt', 'r').read().split('\n')
# f_n_n1.remove(f_n_n1[-1])
# f_n_nn = open('namen-n.txt', 'r').read().split('\n')
# f_n_nn.remove(f_n_nn[-1])
#
# f_id_11 = open('1-1.txt', 'w')
# f_id_1n = open('1-n.txt', 'w')
# f_id_n1 = open('n-1.txt', 'w')
# f_id_nn = open('n-n.txt', 'w')
#
# for line in f_n_11:
#     s = line.split('\t')
#     f_id_11.write('{}'.format(e2id[s[0]]) + '\t' + '{}'.format(r2id[s[1]]) + '\t' + '{}'.format(e2id[s[2]]) + '\n')
# for line in f_n_1n:
#     s = line.split('\t')
#     f_id_1n.write('{}'.format(e2id[s[0]]) + '\t' + '{}'.format(r2id[s[1]]) + '\t' + '{}'.format(e2id[s[2]]) + '\n')
# for line in f_n_n1:
#     s = line.split('\t')
#     f_id_n1.write('{}'.format(e2id[s[0]]) + '\t' + '{}'.format(r2id[s[1]]) + '\t' + '{}'.format(e2id[s[2]]) + '\n')
# for line in f_n_nn:
#     s = line.split('\t')
#     f_id_nn.write('{}'.format(e2id[s[0]]) + '\t' + '{}'.format(r2id[s[1]]) + '\t' + '{}'.format(e2id[s[2]]) + '\n')
#
#
# f_id_11.close()
# f_id_1n.close()
# f_id_n1.close()
# f_id_nn.close()

f1 = open('train.txt', 'r').read().split('\n')
f1.remove(f1[-1])

f2 = open('train2id.txt', 'w')

for line in f1:
    s = line.split('\t')
    f2.write('{}'.format(e2id[s[0]]) + ',' + '{}'.format(r2id[s[1]]) + ',' + '{}'.format(e2id[s[2]]) + '\n')
f2.close()
