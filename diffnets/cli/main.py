#Native to python
import pickle
import os
import shutil
import click
import yaml
import multiprocessing as mp
#third-party libraries
import numpy as np
import mdtraj as md
#diffnets libraries
from diffnets.analysis import Analysis
from diffnets.data_processing import ProcessTraj, WhitenTraj
from diffnets.training import Trainer
from diffnets.utils import get_fns
from diffnets import nnutils

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

@click.group()
def cli():
    pass

@cli.command(name='process')
@click.argument('sim_dirs')
@click.argument('pdb_fns')
@click.argument('outdir')
@click.option('-a','--atom-sel',default=None)
@click.option('-s','--stride',default=None)
def preprocess_data(sim_dirs,pdb_fns,outdir,atom_sel=None,stride=1):
    """ sim_dirs: Path to an np.array containing directory names. The 
               array needs one directory name for each variant where each
               directory contains all trajectories for that variant.

        pdb_fns: Path to an np.array containing pdb filenames. The 
               array needs one pdb filename for each variant. The order of 
               variants should match the order of sim_dirs.

        atom_sel: (optional) Path to an np.array containing a list of indices for 
              each variant, which operates on the pdbs supplied. The indices
              need to select equivalent atoms across variants.

        stride: (optional) Path to an np.array containing an integer for
               each variant.

        outdir: Path you would like processed data to live.
 """
    try:
        var_dir_names = np.load(sim_dirs)
    except:
        click.echo(f'Incorrect input for sim_dirs. Use --help flag for '
               'information on the correct input for sim_dirs.')
        raise

    try:
        var_pdb_fns = np.load(pdb_fns)
    except:
        click.echo(f'Incorrect input for pdb_fns. Use --help flag for '
               'information on the correct input for pdb_fns.')
        raise

    if stride:
        try:
            stride = np.load(stride)
        except:
            click.echo(f'Incorrect input for stride. User must supply a ' 
               'path to a np.array that has a stride value for each variant.')
            raise

    if atom_sel:
        try:
            atom_sel = np.load(atom_sel)
            #Add a check to make sure atom_sel is not same
            n_atoms = [md.load(fn).atom_slice(atom_sel[i]).n_atoms for i,fn in enumerate(var_pdb_fns)]
            if len(np.unique(n_atoms)) != 1:
                raise ImproperlyConfigured(
                    f'atom_sel needs to choose equivalent atoms across variants. '
                     'After performing atom_sel, pdbs have different numbers of '
                     'atoms.')
        except:
            click.echo(f'Incorrect input for atom_sel. Use --help flag for '
               'information on the correct input for atom_sel.')
            raise

    else:
        n_resis = []
        for fn in var_pdb_fns:
            pdb = md.load(fn)
            n_resis.append(pdb.top.n_residues)
        print(f'unique: {len(np.unique(n_resis))}')
        if len(np.unique(n_resis)) != 1:  # np.unique remove the repeptitive numbers and sort the rest
            raise ImproperlyConfigured(
                f'The PDBs supplied have different numbers of residues. The '
                 'default atom selection does not work in this case. Please '
                 'use the --atom-sel option to choose equivalent atoms across  '
                 'different variant pdbs.')

    if len(var_dir_names) != len(var_pdb_fns):
        raise ImproperlyConfigured(
            f'pdb_fns and sim_dirs must point to np.arrays that have '
             'the same length')

    for vd,fn in zip(var_dir_names, var_pdb_fns):
        traj_fns = get_fns(vd, "*.xtc")
        n_traj = len(traj_fns)
        click.echo("Found %s trajectories in %s" % (n_traj,vd))
        if n_traj == 0:
            raise ImproperlyConfigured(
                "Found no trajectories in %s" % vd)
        """
        try:  # This is too memory intensive and traj is not used later. 
            traj = md.load(traj_fns[0],top=fn)
        except:
            click.echo(f'Order of pdb_fns and sim_dirs need to '
                'correspond to each other.')
            raise
        """
    proc_traj = ProcessTraj(var_dir_names,var_pdb_fns,outdir,stride=stride,
                            atom_sel=atom_sel)
    proc_traj.run()
    print("\nFinished aligning trajectories!")
    whiten_traj = WhitenTraj(outdir)
    print("\nStart whitening the input trajectories ...")
    whiten_traj.run()
    print("\nFinished data whitening!\n")
    
nn_d = {
    'nnutils.split_sae': nnutils.split_sae,
    'nnutils.sae': nnutils.sae,
    'nnutils.ae': nnutils.ae,
    'nnutils.split_ae': nnutils.split_ae
}

@cli.command(name='train')
@click.argument('config')
def train(config):
    """ config: YML config file. See train_sample.yml for an example and
                train_sample.txt for parameter descriptions.
    """
    with open(config) as f:
        job = yaml.load(f, Loader=yaml.FullLoader)
 
    required_keys = ['data_dir','n_epochs','act_map','lr','n_latent',
                     'hidden_layer_sizes','em_bounds','do_em','em_batch_size',
                     'nntype','batch_size','batch_output_freq',
                     'epoch_output_freq','test_batch_size','frac_test',
                     'subsample','outdir','data_in_mem']
    optional_keys = ["close_inds_fn","label_spreading","w_loss"]

    if hasattr(job['nntype'], 'split_inds'):
        required_keys.append("close_inds_fn")

    if "label_spreading" in job.keys():
        if job["label_spreading"] not in ["gaussian","uniform","bimodal"]:
            raise ImproperlyConfigured(
                f'label_spreading must be set to gaussian or uniform')
    
    # Added by WTH: w_loss=[w_cls, w_corr] relative to the weight of recon (alsways set as 1)
    if "w_loss" in job.keys():
        if len(job['w_loss']) != 2:
            raise ImproperlyConfigured(
                    'w_loss should have 2 elements!')
        job['w_loss'] = np.array(job['w_loss'])
    else:
        job['w_loss'] = np.array([1, 1])  # equally weighted

    for key in job.keys():
        try:
            required_keys.remove(key)
        except:
            if key in optional_keys:
                continue
            else:
                raise ImproperlyConfigured(
                f'{key} is not a valid parameter. Check yaml file.')
    
    if len(required_keys) != 0:  # If the parameter is in config.yml, hence in job.keys(), it would have been removed from required_keys.
        raise ImproperlyConfigured(
                f'Missing the following parameters in {config} '
                 '{required_keys} ')
   
    data_dir  = job['data_dir']   # e.g. ./whitened_data
    data_fns = get_fns(data_dir,"*.npy")  # e.g. wm.npy, uwm.npy, traj_lens.npy, cm.npy, c00.npy
    wm_fn = os.path.join(data_dir,"wm.npy")
    if wm_fn not in data_fns:
        raise ImproperlyConfigured(
                f'Cannot find wm.npy in preprocessed data directory. Likely '
                 'need to re-run data preprocessing step.')

    xtc_fns = os.path.join(data_dir,"aligned_xtcs")
    data_fns = get_fns(xtc_fns,"*.xtc")   # e.g. 000000.xtc, 000001.xtc, ...
    ind_fns = os.path.join(data_dir,"indicators")
    inds = get_fns(ind_fns,"*.npy")       # e.g. 000000.npy, 000001.npy, ...

    if (len(inds) != len(data_fns)) or len(inds)==0:
        raise ImproperlyConfigured(
                f'Number of files in aligned_xtcs and indicators should be '
                  'equal. Likely need to re-run data preprocessing step.')
    last_indi = np.load(inds[-1])   # the last indicator file
  
    n_cores = mp.cpu_count()
    
    # WTH: Lines below are added to prevent UserWarning: This DataLoader will create xx worker processes in total. Our suggested max number of worker in current system is xx, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.

    max_cores = len(os.sched_getaffinity(0))
    if n_cores > max_cores:
        n_cores = max_cores
    
    master_fn = os.path.join(job['data_dir'], "master.pdb")
    master = md.load(master_fn)
    n_atoms = master.top.n_atoms
    n_features = 3 * n_atoms

    job['layer_sizes'] =[n_features,n_features]   # Original layer size
    if len(job['hidden_layer_sizes']) == 0:       
        job['layer_sizes'].append(int(n_features/4))  # 4-fold reduction by default
    else: 
        for layer in job['hidden_layer_sizes']:
            job['layer_sizes'].append(layer)

    job['layer_sizes'].append(job['n_latent'])
    job['act_map'] = np.array(job['act_map'],dtype=float)
    job['em_bounds'] = np.array(job['em_bounds'])
    job['em_n_cores'] = n_cores
    job['nntype'] = nn_d[job['nntype']]
       
    if len(job['act_map']) != last_indi[0]+1:
        raise ImproperlyConfigured(
                f'act_map needs to contain a value for each variant.')

    if len(job['act_map']) != len(job['em_bounds']):
        raise ImproperlyConfigured(
                f'act_map and em_bounds should be the same length since '
                 'each variant needs an initial classification label and '
                 'a range for the EM update')

    if n_features != job['layer_sizes'][0]:
        raise ImproperlyConfigured(
                f'1st layer size does not match the number of xyz coordinates')  

    if job['layer_sizes'][0]!=job['layer_sizes'][1]:
        raise ImproperlyConfigured(
                f'1st and 2nd layer size need to be equal.')
  
    if job['layer_sizes'][-1]!=job['n_latent']:
        raise ImproperlyConfigured(
                f'Last layer size needs to equal number of latent variables')

    if 'close_inds_fn' in job.keys():
        if hasattr(job['nntype'], 'split_inds'):
            inds = np.load(job['close_inds_fn'])  # Should be relative to master.pdb
            close_xyz_inds = []
            for i in inds:
                close_xyz_inds.append(i*3)      # index of x coord of atom i in all_inds
                close_xyz_inds.append((i*3)+1)  # index of y coord of atom i in all_inds
                close_xyz_inds.append((i*3)+2)  # index of z coord of atom i in all_inds
            all_inds = np.arange((master.n_atoms*3))
            non_close_xyz_inds = np.setdiff1d(all_inds,close_xyz_inds)
            job['inds1'] = np.array(close_xyz_inds)   # for encoder A
            job['inds2'] = non_close_xyz_inds         # for encoder B
        else:
            raise ImproperlyConfigured(
                f'Indices chosen for a split autoencoder architecture '
                 '(close_inds_fn), but  a split autoencoder architecture '
                 'was not chosen (nntype)')

    if not os.path.exists(job['outdir']):
        cmd = "mkdir %s" % job['outdir']
        os.system(cmd)
        shutil.copyfile(config,os.path.join(job['outdir'],config))
        #raise ImproperlyConfigured(
        #        f'outdir already exists. Rename and try again. ')

    trainer = Trainer(job)
    net = trainer.run(data_in_mem=job['data_in_mem'])
    print("\nFinished training the DiffNets! " + "\U0001F37A" * 3 + '\n')

@cli.command(name='analyze')
@click.argument('data_dir')
@click.argument('net_dir')
@click.option('-i', '--inds',
              help=f'Path to a np.array that contains indices with respect '
                    'to data_dir/master.pdb. These indices will be used'
                    'to find features that distinguish variants by looking at '
                    'a subset of the protein instead of the whole protein')
@click.option('-c','--cluster-number', default=1000,
              help=f'Number of clusters desired for clustering on latent space')
@click.option('-n','--n-distances', default=100,
              help=f'Number of distances to plot. Takes the n distances that '
                    'are most correlated with the diffnet classification score.')
def analyze(data_dir,net_dir,inds=None,cluster_number=1000,n_distances=100):
    """ data_dir: Path to directory with processed and whitened data.

        net_dir: path to directory with output from training.
    """
    net_fn = os.path.join(net_dir,"nn_best_polish.pkl")

    try: 
        with open(net_fn, "rb") as f:
            net = pickle.load(f)
    except:
        click.echo(f'net_dir supplied either does not exist or does not '
                    'contain a trained DiffNet.')
        raise

    try:
        pdb = md.load(os.path.join(data_dir,"master.pdb"))
        n = pdb.n_atoms
    except:
        click.echo(f'data_dir supplied should contain the processed/whitened '
                    'data including master.pdb')
        raise

    net.cpu()
    a = Analysis(net,net_dir,data_dir)

    #this method generates encodings (latent space) for all frames,
    #produces reconstructed trajectories, produces final classification
    #labels for all frames, and calculates an rmsd between the DiffNets
    #reconstruction and the actual trajectories
    a.run_core()

    #This produces a clustering based on the latent space and then
    # finds distances that are correlated with the DiffNets classification
    # score and generates a .pml that can be opened with master.pdb
    # to generate a figure showing what the diffnet learned.
    #Indices for feature analysis
    print('\nFinding representative features distiniguishing the ensembles ...')
    if inds is None:
        print("    Note: No indices with respect to the reference PDB structure are provided.")
        inds = np.arange(n)
    else:
        try:
            inds = np.load(inds)
            print(inds.shape)
        except:
            click.echo(f'Inds needs to be a path to a np.array')
            raise
    a.find_feats(inds,"rescorr-%s.pml" % n_distances,n_states=cluster_number,
                 num2plot=n_distances)

    #Generate a morph of structures along the DiffNets classification score
    print('\nGenerating representative structures for classification labels from 0 to 1 ...')
    a.morph()
    #print("analysis done")

if __name__=="__main__":
    cli()
