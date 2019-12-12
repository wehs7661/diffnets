import enspara
import enspara.cluster as cluster
import enspara.info_theory as infotheor
import enspara.msm as msm
import enspara.cluster as cluster
import enspara.info_theory as infotheor
import enspara.msm as msm
import mdtraj as md
import nnutils
import numpy as np
import os
import pickle
import scipy.sparse
import sys
import torch
#import umap
import whiten

from pylab import *
from scipy.stats import pearsonr
from torch.autograd import Variable


def split_vars(d, vars):
    n = len(d)
    n_vars = len(vars)
    n_per_var = int(len(d)/n_vars)
    lst = {}
    for i in range(n_vars):
        v = vars[i]
        lst[v] = d[i*n_per_var:(i+1)*n_per_var]
    return lst


def get_extrema(lst_lst):
    my_min = np.inf
    my_max = -np.inf
    for lst in lst_lst:
        my_min = np.min((my_min, np.min(lst)))
        my_max = np.max((my_max, np.max(lst)))
    return my_min, my_max


def common_hist(lst_lst, labels, bins):
    my_min, my_max = get_extrema(lst_lst)
    n_lst = len(lst_lst)
    all_h = {}
    for i in range(n_lst):
        h, x = np.histogram(lst_lst[i], bins=bins, range=(my_min, my_max))
        all_h[labels[i]] = h
    return all_h, x


def calc_overlap(d1, d2, bins):
    n_feat = d1.shape[1]
    js = np.zeros(n_feat)
    ent1 = np.zeros(n_feat)
    ent2 = np.zeros(n_feat)
    for i in range(n_feat):
        h, x = common_hist([d1[:, i], d2[:, i]], ["d1", "d2"], bins)
        h1 = h["d1"]
        h2 = h["d2"]
        p1 = np.array(h1) / h1.sum()
        p2 = np.array(h2) / h2.sum()
        js[i] = infotheor.js_divergence(p1, p2)
        ent1[i] = infotheor.shannon_entropy(p1)
        ent2[i] = infotheor.shannon_entropy(p2)
    return js, ent1, ent2


def project(enc, lab, vars, i1, i2, bins, my_title, cutoff=0.8):
    subsample = 100

    all_act_inds = np.where(lab>cutoff)[0]
    act_i1_mu = enc[all_act_inds, i1].mean()
    act_i1_std = enc[all_act_inds, i1].std()
    act_i2_mu = enc[all_act_inds, i2].mean()
    act_i2_std = enc[all_act_inds, i2].std()

    n_vars = len(vars)
    enc_dict = split_vars(enc, vars)
    lab_dict = split_vars(lab, vars)
    i1_dict = {}
    i2_dict = {}
    act_inds = {}
    for v in vars:
        i1_dict[v] = enc_dict[v][:, i1]
        i2_dict[v] = enc_dict[v][:, i2]
        act_inds[v] = np.where(lab_dict[v]>cutoff)[0]
    # i1_min, i1_max = get_extrema(i1_dict.values())
    # i2_min, i2_max = get_extrema(i2_dict.values())

    # drop outliers by only show data within const*std
    const = 3
    i1_mu = enc[:, i1].mean()
    i1_std = enc[:, i1].std()
    i1_min = np.max((i1_mu-const*i1_std, enc[:, i1].min()))
    i1_max = np.min((i1_mu+const*i1_std, enc[:, i1].max()))
    i2_mu = enc[:, i2].mean()
    i2_std = enc[:, i2].std()
    i2_min = np.max((i2_mu-const*i2_std, enc[:, i2].min()))
    i2_max = np.min((i2_mu+const*i2_std, enc[:, i2].max()))

    # get min/max of z dim
    cmin = np.inf
    cmax = -np.inf
    for i in range(n_vars):
        v = vars[i]
        tmp, x, y = np.histogram2d(i1_dict[v], i2_dict[v], range=([i1_min, i1_max], [i2_min, i2_max]), bins=n_bins)
        tmp /= tmp.sum()
        h = np.zeros(tmp.shape)
        inds = np.where(tmp>0)
        h[inds] = -np.log(tmp[inds])
        #inds = np.where(np.isnan(h))
        #h[inds] = 0
        cmin = np.min((cmin, h[inds].min()))
        cmax = np.max((cmax, h[inds].max()))

    height = 4
    width = height*n_vars
    fig = figure(figsize=(width, height))
    fig.suptitle(my_title)
    bins = 20
    dot_size = 0.1
    for i in range(n_vars):
        v = vars[i]
        ax = fig.add_subplot(1, n_vars, i+1, aspect='auto', xlim=x[[0, -1]], ylim=y[[0, -1]])
        #scatter(i1_dict[v], i2_dict[v], s=dot_size, c='b', alpha=0.1)
        tmp, x, y = np.histogram2d(i1_dict[v], i2_dict[v], range=([i1_min, i1_max], [i2_min, i2_max]), bins=n_bins)
        tmp /= tmp.sum()
        h = cmax*np.ones(tmp.shape)
        inds = np.where(tmp>0)
        h[inds] = -np.log(tmp[inds])
        h -= cmax
        delta_x = (x[1]-x[0])/2.0
        delta_y = (y[1]-y[0])/2.0
        #imshow(h, interpolation='bilinear', aspect='auto', origin='low', extent=[x[0]+delta_x, x[-1]+delta_x, y[0]+delta_y, y[-1]+delta_y], vmin=cmin-cmax, vmax=0, cmap=get_cmap('Blues_r'))
        # transpose to put first dimension (i1) on x axis
        #imshow(h.T, interpolation='bilinear', aspect='auto', origin='low', extent=[y[0]+delta_y, y[-1]+delta_y, x[0]+delta_x, x[-1]+delta_x], vmin=cmin-cmax, vmax=0, cmap=get_cmap('Blues_r'))
        imshow(h.T, interpolation='bilinear', aspect='auto', origin='low', extent=[x[0]+delta_x, x[-1]+delta_x, y[0]+delta_y, y[-1]+delta_y], vmin=cmin-cmax, vmax=0, cmap=get_cmap('Blues_r'))
        colorbar()
        
        #im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
        #xcenters = (x[:-1] + x[1:]) / 2
        #ycenters = (y[:-1] + y[1:]) / 2
        #im.set_data(xcenters, ycenters, h)
        #im.set_cmap(cm)
        #ax.images.append(im)
        
        lines = []
        line_labels = []
        for v2 in vars:
            i1_mu = i1_dict[v2].mean()
            i1_std = i1_dict[v2].std()
            i2_mu = i2_dict[v2].mean()
            i2_std = i2_dict[v2].std()
            #print(v, "x", i1_mu, i1_std)
            #print(v, "y", i2_mu, i2_std)
            line, _, _ = errorbar([i1_mu], [i2_mu], xerr=[i1_std], yerr=[i2_std], label=v2)
            lines.append(line)
            line_labels.append(v2)

            # inds = act_inds[v2]
            # if inds.shape[0] > subsample:
            #     inds = inds[::subsample]
            # print(inds.shape)
            # if inds.shape[0] > 0:
            #     scatter(i1_dict[v2][inds], i2_dict[v2][inds], s=dot_size, c='k')
        
        line, _, _ = errorbar([act_i1_mu], [act_i2_mu], xerr=[act_i1_std], yerr=[act_i2_std], label='act', ecolor='k', fmt='k')
        lines.append(line)
        line_labels.append('act')
        #legend()

        title(v)
    # scatter([0], [0], s=dot_size*10, c='k')
    # scatter([6], [0], s=dot_size*10, c='k')
    # scatter([6], [6], s=dot_size*10, c='k')
    fig.legend(lines, line_labels)
    show()


def morph_conditional(nn_dir, data_dir, n_frames=10):
    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    uwm_fn = os.path.join(data_dir, "uwm.npy")
    uwm = np.load(uwm_fn)
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)
    enc = load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    n_latent = int(enc.shape[1])
    morph_dir = os.path.join(nn_dir, "morph")
    if not os.path.exists(morph_dir):
        os.mkdir(morph_dir)

    for i in range(n_latent):
        my_min, my_max = get_extrema([enc[:, i]])
        print(i, my_min, my_max)
        morph_enc = np.zeros((n_frames, n_latent))
        vals = np.linspace(my_min, my_max, n_frames)
        delta = (vals[1] - vals[0]) * 0.5
        for j in range(n_frames):
            val = vals[j]

            # set each latent variable to most probable value given latent(ind) within delta of selected value
            inds = np.where(np.logical_and(enc[:,i]>=val-delta, enc[:,i]<=val+delta))[0]
            for k in range(n_latent):
                n, x = np.histogram(enc[inds, k], bins=20)
                offset = (x[1] - x[0]) * 0.5
                morph_enc[j, k] = x[n.argmax()] + offset

            # fix ref latent variable to val
            morph_enc[j, i] = val

        morph_enc = Variable(torch.from_numpy(morph_enc).type(torch.FloatTensor))
        try:
            outputs, labs = net.decode(morph_enc)
        except:
            print("single")
            outputs = net.decode(morph_enc)
        outputs = outputs.data.numpy()
        coords = whiten.apply_unwhitening(outputs, uwm, cm)
        print("shape", coords.shape)
        recon_trj = md.Trajectory(coords.reshape((n_frames, n_atoms, 3)), ref_s.top)
        out_fn = os.path.join(morph_dir, "m%d.pdb" % i)
        recon_trj.save(out_fn)

def morph_cond_mean(nn_dir,data_dir,n_frames=10):
    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    uwm_fn = os.path.join(data_dir, "uwm.npy")
    uwm = np.load(uwm_fn)
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)
    enc = load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    n_latent = int(enc.shape[1])
    morph_dir = os.path.join(nn_dir, "morph_bin_mean")
    if not os.path.exists(morph_dir):
        os.mkdir(morph_dir)

    for i in range(n_latent):
        my_min, my_max = get_extrema([enc[:, i]])
        print(i, my_min, my_max)
        morph_enc = np.zeros((n_frames, n_latent))
        vals = np.linspace(my_min, my_max, n_frames)
        delta = (vals[1] - vals[0]) * 0.5
        for j in range(n_frames):
            val = vals[j]

            # set each latent variable to most probable value given latent(ind) within delta of selected value
            inds = np.where(np.logical_and(enc[:,i]>=val-delta, enc[:,i]<=val+delta))[0]
            for k in range(n_latent):
                x  = np.mean(enc[inds,k])
                morph_enc[j, k] = x

            # fix ref latent variable to val
            morph_enc[j, i] = val

        morph_enc = Variable(torch.from_numpy(morph_enc).type(torch.FloatTensor))
        traj = nnutils.recon_traj(morph_enc,net,ref_s.top,cm)
        rmsf = get_rmsf(traj)

        out_fn = os.path.join(outdir, "m%d.pdb" % i)
        traj.save_pdb(out_fn, bfactors=rmsf)

def morph_std(nn_dir, data_dir, enc):
    outdir = os.path.join(nn_dir, "morph_std")
    nnutils.mkdir(outdir)
    n_frames = 10

    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)

    n_latent = int(enc.shape[1])
    ave_enc = enc.mean(axis=0)
    std_enc = enc.std(axis=0)
    max_enc = enc.max(axis=0)
    min_enc = enc.min(axis=0)

    # want vary between mean +/- 2*std but not go out of range   
    for i in range(n_latent):
        #my_min = np.max((ave_enc[i]-5*std_enc[i], min_enc[i]))
        #my_max = np.min((ave_enc[i]+5*std_enc[i], max_enc[i]))
        my_min = min_enc[i]
        my_max = max_enc[i]

        morph_enc = np.zeros((n_frames, n_latent)) + ave_enc
        morph_enc[:, i] = np.linspace(my_min, my_max, n_frames)
        traj = nnutils.recon_traj(morph_enc, net, ref_s.top, cm)

        rmsf = get_rmsf(traj)

        out_fn = os.path.join(outdir, "m%d.pdb" % i)
        traj.save_pdb(out_fn, bfactors=rmsf)


def get_rmsf(traj):
    x_mean = traj.xyz.mean(axis=0)
    delta = traj.xyz - x_mean
    d2 = np.einsum('ijk,ijk->ij', delta, delta)
    p = 1.0*np.ones(len(traj)) / len(traj)
    msf = np.einsum('ij,i->j', d2, p)
    return np.sqrt(msf)


def get_act_inact(nn_dir, data_dir, enc, labels):
    """Save most active/inactive sturctures with RMSDs from target less than 2 Angstroms."""
    outdir = os.path.join(nn_dir, "act_and_inact")
    nnutils.mkdir(outdir)
    n_extreme = 1000

    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    ca_inds = ref_s.top.select('name CA')
    n_atoms = ref_s.top.n_atoms
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)
    
    rmsd_cutoff = 0.2
    rmsd_fn = os.path.join(nn_dir, "rmsd.npy")
    rmsd = np.load(rmsd_fn)
    good_inds = np.where(rmsd<rmsd_cutoff)
    enc = enc[good_inds]
    labels = labels[good_inds]

    inds = np.argsort(labels.flatten())

    act_traj = nnutils.recon_traj(enc[inds[-n_extreme:]], net, ref_s.top, cm)
    out_fn = os.path.join(outdir, "active.xtc")
    act_traj.save(out_fn)
    for i in range(10):
        out_fn = os.path.join(outdir, "act%d.pdb" % i)
        act_traj[i].save(out_fn)
    act_traj = act_traj.atom_slice(ca_inds)
    act_rmsf = 10*get_rmsf(act_traj)
    out_fn = os.path.join(outdir, "act_rmsf.npy")
    np.save(out_fn, act_rmsf)

    inact_traj = nnutils.recon_traj(enc[inds[:n_extreme]], net, ref_s.top, cm)
    out_fn = os.path.join(outdir, "inactive.xtc")
    inact_traj.save(out_fn)
    for i in range(10):
        out_fn = os.path.join(outdir, "inact%d.pdb" % i)
        inact_traj[i].save(out_fn)
    inact_traj = inact_traj.atom_slice(ca_inds)
    inact_rmsf = 10*get_rmsf(inact_traj)
    out_fn = os.path.join(outdir, "inact_rmsf.npy")
    np.save(out_fn, inact_rmsf)

    #all_h, x = common_hist([act_rmsf, inact_rmsf], ['act', 'inact'], 20)
    fig = figure(figsize=(4, 8))
    title
    #plot(x, all_h['act'], label='act')
    #plot(x, all_h['inact'], label='inact')
    res_nums = []

    for r in act_traj.top.residues:
        res_nums.append(r.resSeq)

    ax = fig.add_subplot(211)
    plot(res_nums, act_rmsf, label='act')
    plot(res_nums, inact_rmsf, label='inact')
    legend()

    ax = fig.add_subplot(212)
    d = act_rmsf-inact_rmsf
    plot(res_nums, d, 'k')
    out_fn = os.path.join(outdir, "act_minus_inact.npy")
    np.save(out_fn, d)
    show()

    out_fn = os.path.join(outdir, "act_minus_inact.pdb")
    ref_s = ref_s.atom_slice(ca_inds)
    ref_s.save_pdb(out_fn, bfactors=d)
    print("rmsf delta extrema", d.min(), d.mean(), d.max())


def enc_corr(enc):
    n_latent = enc.shape[1]
    corr = []
    for i in range(n_latent):
        for j in range(i+1, n_latent):
            c = pearsonr(enc[:,i], enc[:,j])[0]
            corr.append(c)
    return np.array(corr)


def project_act(lab_v, vars, my_title):
    n_vars = len(vars)
    print(my_title)
    fig = figure(figsize=(4, 4))
    fig.suptitle(my_title)
    for i in range(n_vars):
        v = vars[i]
        n, x = np.histogram(lab_v[v], range=(0, 1), bins=50)
        plot(x[:-1], n, label=v)
        print(v, lab_v[v].mean())
    legend()
    show()


def check_loss(nn_dir):
    i = 2
    fn = os.path.join(nn_dir, "test_loss_%d.npy" % i)
    while os.path.exists(fn):
        d = np.load(fn)
        plot(d, label=str(i))
        i += 1
        fn = os.path.join(nn_dir, "test_loss_%d.npy" % i)
    fn = os.path.join(nn_dir, "test_loss_polish.npy")
    d = load(fn)
    plot(d, label='p')
    legend()
    show()


def euc_dist(trj, frame):
    diff = np.abs(trj - frame)
    try:
        d = np.sqrt(np.sum(diff * diff, axis=1))
    except:
        d = np.array([np.sqrt(np.sum(diff * diff))])
    return d


def clust_encod(nn_dir, n_clusters, vars, lag_times,n_traj_per_var):
    msm_dir = os.path.join(nn_dir, "msm_%d" % n_clusters)
    nnutils.mkdir(msm_dir)

    enc = nnutils.load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    enc_v = split_vars(enc, vars)
    n_vars = len(vars)
    #n_traj_per_var = 5

    clusters = cluster.hybrid.hybrid(enc, euc_dist, n_clusters=n_clusters, n_iters=1)
    # clusters.assignments and clusters.centers most relevant vars
    cluster_fn = os.path.join(msm_dir, "clusters.pkl")
    pickle.dump(clusters, open(cluster_fn, 'wb'))

    # assuming 5 traj of equal length per variant, divide into traj
    assigns = clusters.assignments.reshape((n_vars*n_traj_per_var, -1))

    height = 4
    width = height*n_vars
    fig = figure(figsize=(width, height))
    fig.suptitle(nn_dir)
    for i in range(n_vars):
        v = vars[i]
        print("Getting impolied timescales for", v)
        v_assians = assigns[i*n_traj_per_var:(i+1)*n_traj_per_var]
        
        f = lambda c: msm.builders.normalize(c, prior_counts=1.0/n_clusters, calculate_eq_probs=True)
        imp_times = msm.implied_timescales(v_assians, lag_times, f)
        imp_fn = os.path.join(msm_dir, "%s_imp_norm.npy" % v)
        np.save(imp_fn, imp_times)

        ax = fig.add_subplot(1, n_vars, i+1, aspect='auto')
        for i, t in enumerate(lag_times):
            scatter(t*np.ones(imp_times.shape[1]), imp_times[i])
        title(v)
        ax.set_yscale('log')

        markov_lag = 10
        c = msm.assigns_to_counts(v_assians, 1, max_n_states=n_clusters)
        c_fn = os.path.join(msm_dir, "%s_c_raw_lag%s.npz" % (v, markov_lag))
        scipy.sparse.save_npz(c_fn, c)
        C, T, p = msm.builders.normalize(c, prior_counts=1.0/n_clusters, calculate_eq_probs=True)
        p_fn = os.path.join(msm_dir, "%s_p_norm_lag%d.npy" % (v, markov_lag))
        np.save(p_fn, p)
        T_fn = os.path.join(msm_dir, "%s_T_norm_lag%d.npy" % (v, markov_lag))
        np.save(T_fn, T)
        C_fn = os.path.join(msm_dir, "%s_C_norm_lag%d.npy" % (v, markov_lag))
        np.save(C_fn, C)
    out_fn = os.path.join(msm_dir, "imp_times.png")
    savefig(out_fn)
    show()


#n_bins = 100
#vars = ["m", "em", "gm", "gem"]
#n_vars = len(vars)
#data_dir = "data/m_em_gm_gem_cabcn"
#lag_times = [1, 10, 50, 100, 200, 500] # in frames, 10ps per frame
#
#nn_dir = "em50_sae_e51_lr0.000100_lat25_r0"
##nn_dir = "ae_e51_lr0.000100_lat25_r0"
##nn_dir = "sae_e51_lr0.000100_lat25_r0"
#nn_dir = "em30_sae_e51_lr0.000100_lat100_r0"
##nn_dir = "em20_sae_e51_lr0.000100_lat100_r0"
##nn_dir = "ae_e51_lr0.000100_lat10_r0"
##lab_dir = os.path.join(nn_dir, "em50_labels")
#lab_dir = os.path.join(nn_dir, "labels")
#lab = nnutils.load_npy_dir(lab_dir, "lab*npy")
#lab_v = split_vars(lab, vars)
#enc = nnutils.load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
#enc_v = split_vars(enc, vars)
#
##clust_encod(nn_dir, 100, vars, lag_times)
##clust_encod("ae_e51_lr0.000100_lat100_r0", 100, vars, lag_times)
##sys.exit(0)
#
#check_loss(nn_dir)
##sys.exit(0)
#
#js, ent1, ent2 = calc_overlap(enc_v["m"], enc_v["gem"], n_bins)
#inds = np.argsort(js)
#for ind in inds:
#    print(ind, js[ind], ent1[ind], ent2[ind])
#
#
#morph_std(nn_dir, data_dir, enc)
#sys.exit(0)
#
#project(enc, lab, vars, inds[-2], inds[-1], n_bins, "sae")
#project_act(lab_v, vars, "sae")
#
#get_act_inact(nn_dir, data_dir, enc, lab)
#
##sys.exit(0)
#
#print("umap")
#subsample = 10
#src_dir = os.path.join(data_dir, "labels")
#src = nnutils.load_npy_dir(src_dir, "*npy")
#src = src[::subsample]
#nn_map = umap.UMAP().fit_transform(enc[::subsample])
#
#act_inds = np.where(lab[::subsample]>0.8)[0]
#fig = figure(figsize=(4, 4))
#dot_size = 0.5
#plot(nn_map[act_inds,0], nn_map[act_inds,1], 'ko', label='atc', markersize=dot_size*3, markeredgewidth=0)
#for i in range(n_vars):
#    inds = np.where(src==i)[0]
#    plot(nn_map[inds,0], nn_map[inds,1], 'o', label=vars[i], markersize=dot_size, alpha=0.1, markeredgewidth=0)
#leg = legend(loc='lower left')
#for lh in leg.legendHandles: 
#    lh._legmarker.set_alpha(1)
#    lh._legmarker.set_markersize(dot_size*3)
#out_fn = os.path.join(nn_dir, "umap.png")
#savefig(out_fn)
#show()
#sys.exit(0)
#
#methods = ["ae", "em"]
#if False:
#    h, x = common_hist([ae_js, em_js], methods, 10)
#    for mthd in methods:
#        plot(x[:-1], h[mthd], label=mthd)
#    legend()
#    show()




